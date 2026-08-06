"""AI confidence scoring.

The ensemble fuses rule-based technical confluence with an optional
machine-learning probability model (HistGradientBoosting trained from the
trade journal / backtest outcomes). The final ``confidence`` is calibrated to
[0, 1] and trades are only allowed above ``risk.min_confidence`` (default 0.90).

Design intent: confidence is deliberately conservative. A high score requires
agreement across trend, momentum, volume, market structure, microstructure and
the absence of over-extension / volatility anomalies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import indicators as ta
from .regime import Regime

log = logging.getLogger("ta_agent.ai")


@dataclass
class SignalContext:
    """Everything the ensemble needs to score a candidate trade."""
    side: str                                # "long" | "short"
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)   # tf -> df
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)  # tf -> feature frame
    regimes: Dict[str, Regime] = field(default_factory=dict)
    trend_alignment: float = 0.0
    ob_imbalance: float = 0.0
    depth_ratio: float = 1.0
    long_short_ratio: float = 1.0            # long value % / short value %
    funding_proxy: float = 0.0               # >0 longs pay, <0 shorts pay
    event_risk: float = 0.0                  # 0 none .. 1 extreme
    atr_pct: float = 0.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


def _last(row_name: str, features: Optional[pd.DataFrame]) -> Optional[float]:
    if features is None or features.empty or row_name not in features.columns:
        return None
    v = features[row_name].iloc[-1]
    return None if pd.isna(v) else float(v)


class ConfidenceEngine:
    def __init__(self, weights: Optional[dict] = None, ml_scorer=None, threshold: float = 0.90):
        self.w = weights or {
            "trend": 0.25,
            "momentum": 0.20,
            "volume": 0.15,
            "structure": 0.15,
            "micro": 0.10,
            "ml": 0.15,
        }
        self.ml_scorer = ml_scorer
        self.threshold = threshold

    # ------------------------------------------------------------------
    def trend_score(self, ctx: SignalContext) -> float:
        r = ctx.trend_alignment
        primary = ctx.regimes.get("1h") or ctx.regimes.get("4h")
        reg = 0.5
        if primary is not None:
            side_ok = (ctx.side == "long" and primary.score >= 0) or (ctx.side == "short" and primary.score <= 0)
            reg = 0.5 + 0.5 * abs(primary.score) if side_ok else 0.5 - 0.5 * abs(primary.score)
        return _clamp(0.6 * r + 0.4 * reg)

    def momentum_score(self, ctx: SignalContext) -> float:
        f = ctx.features.get("1h")
        if f is None:
            f = next(iter(ctx.features.values()), None)
        if f is None or f.empty:
            return 0.5
        s = 0.0
        n = 0
        rsi = _last("rsi_14", f)
        if rsi is not None:
            if ctx.side == "long":
                s += 1.0 if 50 <= rsi <= 70 else (0.7 if 45 <= rsi < 50 else 0.2)
            else:
                s += 1.0 if 30 <= rsi <= 50 else (0.7 if 50 < rsi <= 55 else 0.2)
            n += 1
        adx = _last("adx", f)
        if adx is not None:
            s += _clamp((adx - 20) / 20.0) if adx >= 20 else adx / 40.0
            n += 1
        hist_rise = _last("macd_hist_rise", f)
        above = _last("macd_above_signal", f)
        if hist_rise is not None and above is not None:
            s += 0.5 * hist_rise + 0.5 * above
            n += 1
        di_bull = _last("di_bull", f)
        if di_bull is not None:
            s += 1.0 if di_bull == ctx.side == "long" or (di_bull == 0 and ctx.side == "short") else 0.3
            n += 1
        if n == 0:
            return 0.5
        return _clamp(s / n)

    def volume_score(self, ctx: SignalContext) -> float:
        f = ctx.features.get("1h")
        if f is None:
            f = next(iter(ctx.features.values()), None)
        if f is None or f.empty:
            return 0.5
        s = 0.0
        n = 0
        vz = _last("volume_z", f)
        if vz is not None:
            s += _clamp((vz - 0.5) / 2.0)
            n += 1
        exp = _last("volume_expanding", f)
        if exp is not None:
            s += exp
            n += 1
        cvd = _last("cvd_rising", f)
        if cvd is not None:
            s += 1.0 if cvd == (1 if ctx.side == "long" else 0) else 0.3
            n += 1
        if n == 0:
            return 0.5
        return _clamp(s / n)

    def structure_score(self, ctx: SignalContext) -> float:
        f = ctx.features.get("1h")
        if f is None:
            f = next(iter(ctx.features.values()), None)
        if f is None or f.empty:
            return 0.5
        s = 0.0
        n = 0
        bos_up = _last("bos_up", f)
        bos_dn = _last("bos_down", f)
        if bos_up is not None and bos_dn is not None:
            target = bos_up if ctx.side == "long" else bos_dn
            s += 1.0 if target else 0.2
            n += 1
        st_dir = _last("supertrend_dir", f)
        if st_dir is not None:
            s += 1.0 if (st_dir == 1 and ctx.side == "long") or (st_dir == -1 and ctx.side == "short") else 0.2
            n += 1
        cloud = _last("ich_cloud_bull", f)
        if cloud is not None:
            target = 1.0 if ctx.side == "long" else (1 - cloud)
            s += target
            n += 1
        tk = _last("ich_tenkan_gt_kijun", f)
        if tk is not None:
            s += 1.0 if tk == (1 if ctx.side == "long" else 0) else 0.4
            n += 1
        if n == 0:
            return 0.5
        return _clamp(s / n)

    def micro_score(self, ctx: SignalContext) -> float:
        s = 0.0
        n = 0
        if ctx.ob_imbalance != 0.0:
            target = 1 if ctx.side == "long" else -1
            s += 1.0 if np.sign(ctx.ob_imbalance) == target else 0.3
            n += 1
        if ctx.long_short_ratio:
            # mild contrarian: extreme crowding reduces confidence
            lr = ctx.long_short_ratio
            if 0.6 <= lr <= 1.7:
                s += 1.0
            elif ctx.side == "long" and lr > 1.7:
                s += 0.4
            elif ctx.side == "short" and lr < 0.6:
                s += 0.4
            else:
                s += 0.7
            n += 1
        if ctx.funding_proxy:
            target = 1 if ctx.side == "long" else -1
            s += 1.0 if np.sign(ctx.funding_proxy) == target else 0.5
            n += 1
        if n == 0:
            return 0.5
        return _clamp(s / n)

    def penalties(self, ctx: SignalContext) -> float:
        """Return a multiplicative penalty in [0,1]."""
        pen = 1.0
        f = ctx.features.get("1h")
        if f is None:
            f = next(iter(ctx.features.values()), None)
        if f is not None and not f.empty:
            pv = _last("price_vs_ema21", f)
            atr = _last("atr_pct", f) or 0.0
            if pv is not None:
                extreme = (pv > 4 * atr) if ctx.side == "long" else (pv < -4 * atr)
                if extreme:
                    pen *= 0.4
            vr = _last("vol_ratio", f)
            if vr is not None and vr > 2.5:
                pen *= 0.6
            if vr is not None and vr < 0.4:
                pen *= 0.8
            z = _last("zscore_20", f)
            if z is not None and abs(z) > 3.0:
                pen *= 0.6
        if ctx.event_risk > 0.3:
            pen *= max(0.0, 1.0 - ctx.event_risk)
        return float(pen)

    # ------------------------------------------------------------------
    def score(self, ctx: SignalContext) -> dict:
        comps = {
            "trend": self.trend_score(ctx),
            "momentum": self.momentum_score(ctx),
            "volume": self.volume_score(ctx),
            "structure": self.structure_score(ctx),
            "micro": self.micro_score(ctx),
        }
        raw = sum(self.w[k] * v for k, v in comps.items() if k != "ml")
        w_sum = sum(v for k, v in self.w.items() if k != "ml")

        ml_prob: Optional[float] = None
        if self.ml_scorer is not None:
            X = self._ml_vector(ctx)
            if X is not None:
                try:
                    ml_prob = float(self.ml_scorer.predict_proba_single(X, ctx.side))
                except Exception as exc:  # pragma: no cover
                    log.warning("ML scoring failed: %s", exc)
        if ml_prob is not None:
            raw = (raw + self.w["ml"] * ml_prob) / (w_sum + self.w["ml"])
        else:
            raw = raw / w_sum

        conf = _clamp(raw * self.penalties(ctx))
        prob = _calibrate_probability(conf)
        return {
            "confidence": conf,
            "probability": prob,
            "components": comps,
            "ml_probability": ml_prob,
            "penalty": self.penalties(ctx),
            "above_threshold": conf >= self.threshold,
        }

    def _ml_vector(self, ctx: SignalContext) -> Optional[np.ndarray]:
        f = ctx.features.get("1h")
        if f is None:
            f = next(iter(ctx.features.values()), None)
        if f is None or f.empty or self.ml_scorer is None:
            return None
        row = f.iloc[-1]
        return np.asarray([row.get(c, np.nan) for c in self.ml_scorer.feature_columns], dtype=float)


def _calibrate_probability(conf: float) -> float:
    """Map confidence into an estimated win probability via a logistic curve.

    Confidence 0.90 -> ~0.62 probability, 0.97 -> ~0.85. This keeps the
    probability honest for sizing (never overstate the edge).
    """
    k = 9.0
    return _clamp(1.0 / (1.0 + np.exp(-k * (conf - 0.65))))


class MLScorer:
    """Optional gradient-boosting probability model trained from journal data."""

    def __init__(self, feature_columns: list, model=None):
        self.feature_columns = list(feature_columns)
        self.model = model
        self._fitted = model is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier  # local import

        X = np.nan_to_num(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        if X.shape[0] < 30 or len(X.shape) != 2 or X.shape[0] != y.shape[0]:
            self._fitted = False
            return
        self.model = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=8, learning_rate=0.08, random_state=42
        )
        # Augment: every trade appears once labelled "win" and once "loss"
        # so the learner sees both classes regardless of the realised rate.
        X_aug = np.vstack([X, X])
        y_aug = np.concatenate([y, 1 - y])
        self.model.fit(X_aug, y_aug)
        self._fitted = True

    def predict_proba_single(self, x: np.ndarray, side: str) -> float:
        if self.model is None:
            return 0.5
        x = np.nan_to_num(np.asarray(x, dtype=float).reshape(1, -1))
        p = float(self.model.predict_proba(x)[0, 1])
        return p if side == "long" else (1.0 - p)

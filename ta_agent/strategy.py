"""Strategy layer: multi-timeframe entry triggers and dynamic exits.

Entry rules
-----------
* Only trade in the direction of the higher-timeframe trend (never against it).
* Require trend alignment >= 0.6 across configured timeframes.
* Trigger: pullback-to-support, breakout/BOS, or momentum continuation
  with volume confirmation.
* R:R must be >= ``risk.min_rr`` and confidence >= ``risk.min_confidence``.
* ATR-based stop, structure/momentum-based target.

Exit rules
----------
* Target, stop, trend reversal, trailing stop, breakeven, volatility spike,
  news event, signal reversal, time-based exit, emergency ATR breach.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import indicators as ta
from . import timestamps
from .ai_ensemble import ConfidenceEngine, SignalContext
from .regime import Regime, trend_alignment

log = logging.getLogger("ta_agent.strategy")

TRIGGER_PULLBACK = "pullback"
TRIGGER_BREAKOUT = "breakout"
TRIGGER_BOS = "bos"
TRIGGER_MOMENTUM = "momentum"


@dataclass
class TradePlan:
    coin: str
    pair: str
    side: str                       # long | short
    entry: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    probability: float
    confidence: float
    expected_return: float
    position_size: float            # qty (base units)
    notional: float
    reason: str
    technical_signals: Dict = field(default_factory=dict)
    ai_signals: Dict = field(default_factory=dict)
    news_impact: str = "none"
    risk_assessment: str = ""
    expected_holding_time: str = ""
    trigger: str = ""
    rr: float = 0.0
    timeframe: str = ""
    timestamp: Optional[int] = None

    def as_report(self) -> dict:
        return {
            "coin": self.coin,
            "pair": self.pair,
            "side": self.side,
            "entry": round(self.entry, 8),
            "stop_loss": round(self.stop_loss, 8),
            "take_profit": round(self.take_profit, 8),
            "risk_%": round(self.risk_pct * 100, 2),
            "probability_%": round(self.probability * 100, 1),
            "confidence_%": round(self.confidence * 100, 1),
            "expected_return": round(self.expected_return, 4),
            "rr": round(self.rr, 2),
            "position_size": round(self.position_size, 8),
            "notional": round(self.notional, 2),
            "reason": self.reason,
            "trigger": self.trigger,
            "timeframe": self.timeframe,
            "technical_signals": self.technical_signals,
            "ai_signals": self.ai_signals,
            "news_impact": self.news_impact,
            "risk_assessment": self.risk_assessment,
            "expected_holding_time": self.expected_holding_time,
        }


class StrategyEngine:
    def __init__(self, settings, confidence_engine: ConfidenceEngine):
        self.s = settings
        self.conf = confidence_engine
        self.conf.threshold = settings.min_confidence
        self.trade_tf = settings.trade_timeframe
        self.trend_tfs = settings.trend_timeframes
        self.min_rr = settings.min_rr
        self.min_conf = settings.min_confidence
        self.atr_mult = float(settings.risk.get("atr_mult_sl", 1.5))

    # ------------------------------------------------------------------
    def analyze(self, coin: str, pair: str, frames: Dict[str, pd.DataFrame],
                features: Dict[str, pd.DataFrame], regimes: Dict[str, Regime],
                micro: Optional[dict] = None, event_risk: float = 0.0,
                timestamp: Optional[int] = None) -> Optional[TradePlan]:
        """Return a TradePlan if a high-probability setup exists, else None."""
        if self.trade_tf not in frames or frames[self.trade_tf].empty:
            return None
        df = frames[self.trade_tf]
        last = df.iloc[-1]
        price = float(last["close"])

        regimes = {tf: r for tf, r in (regimes or {}).items() if tf in self.trend_tfs}
        if not regimes:
            return None
        align = trend_alignment(regimes)
        if align < 0.6:
            return None

        primary = regimes.get(self.trend_tfs[-1]) if self.trend_tfs else None
        if primary is None:
            return None
        side = None
        if primary.trend == "up":
            side = "long"
        elif primary.trend == "down":
            side = "short"
        if side is None:
            return None

        atr_v = float(ta.atr(df, 14)[-1])
        if atr_v <= 0 or price <= 0:
            return None

        trigger, trigger_desc = self._detect_trigger(df, features.get(self.trade_tf), side)
        if trigger is None:
            return None

        # --- stop & targets -------------------------------------------
        if side == "long":
            sl = price - self.atr_mult * atr_v
            tp = self._target(df, side, price, atr_v)
            risk_dist = price - sl
        else:
            sl = price + self.atr_mult * atr_v
            tp = self._target(df, side, price, atr_v)
            risk_dist = sl - price
        if risk_dist <= 0 or tp is None:
            return None
        rr = abs(tp - price) / risk_dist
        if rr < self.min_rr:
            return None

        # --- confidence -------------------------------------------------
        ctx = SignalContext(
            side=side,
            frames=frames,
            features=features,
            regimes=regimes,
            trend_alignment=align,
            ob_imbalance=(micro or {}).get("ob_imbalance", 0.0),
            depth_ratio=(micro or {}).get("depth_ratio", 1.0),
            long_short_ratio=(micro or {}).get("long_short_ratio", 1.0),
            funding_proxy=(micro or {}).get("funding_proxy", 0.0),
            event_risk=event_risk,
            atr_pct=float(atr_v / price),
        )
        res = self.conf.score(ctx)
        if not res["above_threshold"] or res["confidence"] < self.min_conf:
            return None

        risk_pct = self.s.per_trade_risk
        expected_return = risk_pct * (res["probability"] * rr - (1 - res["probability"])) / (rr * risk_pct + 1e-12) * rr
        expected_return = round(risk_pct * (res["probability"] * rr - (1 - res["probability"])), 6)

        tech = {
            "trend_alignment": round(align, 2),
            "primary_trend": primary.trend,
            "adx": round(primary.adx, 1),
            "trigger": trigger,
            "trigger_detail": trigger_desc,
            "atr_pct": round(atr_v / price * 100, 2),
            "rr": round(rr, 2),
        }
        return TradePlan(
            coin=coin,
            pair=pair,
            side=side,
            entry=price,
            stop_loss=sl,
            take_profit=tp,
            risk_pct=risk_pct,
            probability=res["probability"],
            confidence=res["confidence"],
            expected_return=expected_return,
            position_size=0.0,  # set by risk manager
            notional=0.0,
            reason=f"{trigger_desc}; higher-TF {primary.trend} trend; confidence {res['confidence']:.0%}",
            technical_signals=tech,
            ai_signals=res,
            trigger=trigger,
            rr=rr,
            timeframe=self.trade_tf,
            timestamp=timestamp or int(timestamps.col_to_datetime(pd.Series([last["time"]])).iloc[0].timestamp() * 1000),
        )

    # ------------------------------------------------------------------
    def _detect_trigger(self, df: pd.DataFrame, feats: Optional[pd.DataFrame],
                        side: str):
        close = df["close"]
        feats = feats if feats is not None and not feats.empty else None
        price = float(close.iloc[-1])
        ema21 = ta.ema(close, 21)[-1]
        ema50 = ta.ema(close, 50)[-1]

        rsi_v = ta.rsi(close, 14)[-1]
        vol_z = ta.zscore(df["volume"], 20)[-1] if len(df) >= 20 else 0.0
        st_dir = ta.supertrend(df, 10, 3.0)[1][-1]
        sw = ta.swing_points(df, 5, 5)
        struct = ta.last_structure(sw)

        if side == "long":
            if feats is not None and "dc_breakout_up" in feats.columns:
                if float(feats["dc_breakout_up"].iloc[-1]) == 1.0 and vol_z > 0.5:
                    return TRIGGER_BREAKOUT, f"Donchian breakout above 20-bar high on volume z={vol_z:.1f}"
            if struct and struct["bos_up"]:
                return TRIGGER_BOS, "Break of structure (higher high) confirmed"
            if price > ema21 > ema50 and 45 <= rsi_v <= 62 and vol_z > 0.0:
                return TRIGGER_PULLBACK, "Pullback within uptrend to rising EMA-21 with momentum intact"
            if price > ema21 and rsi_v >= 62 and vol_z > 1.0 and st_dir == 1:
                return TRIGGER_MOMENTUM, "Momentum continuation above EMA-21 with volume"
        else:
            if feats is not None and "dc_breakout_dn" in feats.columns:
                if float(feats["dc_breakout_dn"].iloc[-1]) == 1.0 and vol_z > 0.5:
                    return TRIGGER_BREAKOUT, f"Donchian breakdown below 20-bar low on volume z={vol_z:.1f}"
            if struct and struct["bos_down"]:
                return TRIGGER_BOS, "Break of structure (lower low) confirmed"
            if price < ema21 < ema50 and 38 <= rsi_v <= 55 and vol_z > 0.0:
                return TRIGGER_PULLBACK, "Pullback within downtrend to falling EMA-21 with momentum intact"
            if price < ema21 and rsi_v <= 38 and vol_z > 1.0 and st_dir == -1:
                return TRIGGER_MOMENTUM, "Momentum continuation below EMA-21 with volume"
        return None, ""

    # ------------------------------------------------------------------
    def _target(self, df: pd.DataFrame, side: str, price: float, atr_v: float) -> Optional[float]:
        """Take profit: at least min_rr * ATR, extended to structure if available."""
        base_dist = self.atr_mult * atr_v * self.min_rr
        sw = ta.swing_points(df, 5, 5)
        highs = [df["high"].iloc[i] for i in np.where(sw == 1)[0]]
        lows = [df["low"].iloc[i] for i in np.where(sw == -1)[0]]
        if side == "long":
            t = price + base_dist
            for h in reversed(highs):
                if h > price + self.atr_mult * atr_v * 0.6:
                    t = h
                    break
            return float(t)
        else:
            t = price - base_dist
            for l_ in reversed(lows):
                if l_ < price - self.atr_mult * atr_v * 0.6:
                    t = l_
                    break
            return float(t)


# ----------------------------------------------------------------------
# Exit management
# ----------------------------------------------------------------------
@dataclass
class PositionState:
    coin: str
    pair: str
    side: str
    entry: float
    qty: float
    notional: float
    stop_loss: float
    take_profit: float
    realized_fraction: float = 0.0
    entry_time_ms: int = 0
    peak_price: float = 0.0
    breakeven_hit: bool = False
    partial_taken: set = field(default_factory=set)
    reason: str = ""
    confidence: float = 0.0


@dataclass
class ExitDecision:
    action: str   # hold | target | stop | trailing_stop | breakeven | signal_reversal | volatility | news | timeout | emergency
    reason: str
    exit_price: Optional[float] = None
    fraction: float = 1.0


class ExitEngine:
    def __init__(self, settings):
        self.s = settings
        self.ex = settings.exit
        self.atr_mult_sl = float(settings.risk.get("atr_mult_sl", 1.5))

    def evaluate(self, pos: PositionState, price: float, high: float, low: float,
                 atr_pct: float, news_event: bool = False,
                 signal_reversal: bool = False, now_ms: Optional[int] = None) -> ExitDecision:
        rr_now = (price - pos.entry) / max(abs(pos.entry - pos.stop_loss), 1e-12) if pos.side == "long" else \
                 (pos.entry - price) / max(abs(pos.entry - pos.stop_loss), 1e-12)

        # Hard stop
        if pos.side == "long" and low <= pos.stop_loss:
            return ExitDecision("stop", f"Stop loss hit @ {pos.stop_loss:.8f}", pos.stop_loss)
        if pos.side == "short" and high >= pos.stop_loss:
            return ExitDecision("stop", f"Stop loss hit @ {pos.stop_loss:.8f}", pos.stop_loss)

        # Target
        if pos.side == "long" and high >= pos.take_profit:
            return ExitDecision("target", f"Target hit @ {pos.take_profit:.8f}", pos.take_profit)
        if pos.side == "short" and low <= pos.take_profit:
            return ExitDecision("target", f"Target hit @ {pos.take_profit:.8f}", pos.take_profit)

        # Emergency volatility breach
        emergency_mult = float(self.s.risk.get("emergency_exit_atr_mult", 6.0))
        if atr_pct > 0 and (price - pos.entry) / price > emergency_mult * atr_pct and pos.side == "long":
            return ExitDecision("emergency", f"Emergency exit: volatility spike {atr_pct:.2%}", price)
        if atr_pct > 0 and (pos.entry - price) / price > emergency_mult * atr_pct and pos.side == "short":
            return ExitDecision("emergency", f"Emergency exit: volatility spike {atr_pct:.2%}", price)

        # News event
        if news_event:
            return ExitDecision("news", "High-impact news in progress - exiting risk", price)

        # Signal reversal
        if signal_reversal:
            return ExitDecision("signal_reversal", "AI signal reversed on trade timeframe", price)

        # Partial profit / trailing
        partials = self.ex.get("partials", [])
        trail_start = float(self.ex.get("trail_start_rr", 1.5))
        be_rr = float(self.ex.get("breakeven_after_rr", 1.0))
        trail_atr = float(self.ex.get("trail_atr_mult", 2.0))

        partial_frac = None
        for p in partials:
            rr_req = float(p.get("rr", 1.0))
            frac = float(p.get("fraction", 0.33))
            if rr_now >= rr_req and rr_req not in pos.partial_taken:
                pos.partial_taken.add(rr_req)
                partial_frac = frac
                break

        if not pos.breakeven_hit and rr_now >= be_rr:
            pos.breakeven_hit = True
            pos.stop_loss = pos.entry

        if rr_now >= trail_start:
            risk_dist = max(abs(pos.entry - pos.stop_loss),
                            self.atr_mult_sl * atr_pct * price)
            if pos.side == "long":
                trail_stop = price - trail_atr * risk_dist
                if trail_stop > pos.stop_loss:
                    pos.stop_loss = trail_stop
            else:
                trail_stop = price + trail_atr * risk_dist
                if trail_stop < pos.stop_loss:
                    pos.stop_loss = trail_stop
            if pos.side == "long" and low <= pos.stop_loss:
                return ExitDecision("trailing_stop", f"Trailing stop hit @ {pos.stop_loss:.8f}", pos.stop_loss)
            if pos.side == "short" and high >= pos.stop_loss:
                return ExitDecision("trailing_stop", f"Trailing stop hit @ {pos.stop_loss:.8f}", pos.stop_loss)

        if partial_frac is not None:
            return ExitDecision("target", "Partial profit booked", None, partial_frac)

        # Timeout
        max_hold = float(self.s.risk.get("max_hold_hours", 48)) * 3600 * 1000
        if now_ms is not None and (now_ms - pos.entry_time_ms) > max_hold:
            return ExitDecision("timeout", "Maximum holding time reached", price)

        return ExitDecision("hold", "")

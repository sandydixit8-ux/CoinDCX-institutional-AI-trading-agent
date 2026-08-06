import numpy as np
import pandas as pd
import pytest

from ta_agent.ai_ensemble import ConfidenceEngine, SignalContext, _calibrate_probability
from ta_agent.features import build_features
from ta_agent.regime import detect_regime


def frame(closes, seed=0, vol_mult=1.0):
    closes = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) * 1.004
    low = np.minimum(open_, closes) * 0.996
    rng = np.random.default_rng(seed)
    volume = rng.normal(100, 10, len(closes)) * vol_mult
    volume = np.abs(volume)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": closes, "volume": volume})
    df["time"] = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return df


def uptrend_ctx(conf: ConfidenceEngine, side="long"):
    df = frame(np.linspace(100, 210, 300), seed=1, vol_mult=1.5)
    feats = {tf: build_features(df) for tf in ["15m", "1h", "4h"]}
    regimes = {"1h": detect_regime(df), "4h": detect_regime(df), "1d": detect_regime(df)}
    return SignalContext(side=side, frames={"1h": df}, features=feats, regimes=regimes,
                         trend_alignment=1.0, ob_imbalance=0.2, long_short_ratio=1.0,
                         atr_pct=0.02)


def range_ctx(conf: ConfidenceEngine, side="long"):
    df = frame(100 + 3 * np.sin(np.linspace(0, 30 * np.pi, 300)), seed=2)
    feats = {tf: build_features(df) for tf in ["15m", "1h", "4h"]}
    regimes = {"1h": detect_regime(df), "4h": detect_regime(df), "1d": detect_regime(df)}
    return SignalContext(side=side, frames={"1h": df}, features=feats, regimes=regimes,
                         trend_alignment=0.0, ob_imbalance=0.0, long_short_ratio=1.0,
                         atr_pct=0.02)


class TestConfidenceEngine:
    def test_score_shape(self):
        eng = ConfidenceEngine()
        res = eng.score(uptrend_ctx(eng))
        assert 0.0 <= res["confidence"] <= 1.0
        assert 0.0 <= res["probability"] <= 1.0
        assert set(res.keys()) >= {"components", "above_threshold"}

    def test_uptrend_beats_range(self):
        eng = ConfidenceEngine()
        up = eng.score(uptrend_ctx(eng))["confidence"]
        rng = eng.score(range_ctx(eng))["confidence"]
        assert up > rng

    def test_range_below_threshold(self):
        eng = ConfidenceEngine()
        res = eng.score(range_ctx(eng))
        assert not res["above_threshold"]

    def test_threshold_configurable(self):
        eng = ConfidenceEngine(threshold=0.0)
        res = eng.score(range_ctx(eng))
        assert res["above_threshold"]

    def test_event_risk_penalty(self):
        eng = ConfidenceEngine()
        ctx = uptrend_ctx(eng)
        clean = eng.score(ctx)["confidence"]
        ctx.event_risk = 1.0
        risked = eng.score(ctx)["confidence"]
        assert risked < clean

    def test_overextension_penalty(self):
        eng = ConfidenceEngine()
        ctx = uptrend_ctx(eng)
        # push price far above EMA-21 -> penalty
        ctx.frames["1h"] = frame(np.linspace(100, 300, 300), seed=3)
        ctx.features = {tf: build_features(ctx.frames["1h"]) for tf in ["15m", "1h", "4h"]}
        res = eng.score(ctx)
        assert 0.0 < res["penalty"] <= 1.0

    def test_calibration(self):
        assert _calibrate_probability(0.90) > _calibrate_probability(0.50)
        assert 0.0 <= _calibrate_probability(0.0) <= 1.0

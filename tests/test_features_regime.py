import numpy as np
import pandas as pd
import pytest

from ta_agent.features import FEATURE_COLUMNS, build_features
from ta_agent.regime import detect_regime, trend_alignment


def make_df(closes, n=300):
    closes = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) * 1.003
    low = np.minimum(open_, closes) * 0.997
    volume = np.abs(np.random.default_rng(5).normal(100, 20, len(closes)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": closes, "volume": volume})
    df["time"] = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return df


def uptrend():
    return make_df(np.linspace(100, 200, 300))


def downtrend():
    return make_df(np.linspace(200, 100, 300))


def range_frame():
    return make_df(100 + 5 * np.sin(np.linspace(0, 20 * np.pi, 300)))


class TestFeatures:
    def test_columns_present(self):
        f = build_features(uptrend())
        for c in FEATURE_COLUMNS:
            assert c in f.columns

    def test_uptrend_indicators(self):
        f = build_features(uptrend())
        last = f.iloc[-1]
        assert last["ema_stack_bull"] == 1.0
        assert last["supertrend_dir"] == 1
        assert last["rsi_14"] > 60
        assert last["di_bull"] == 1.0

    def test_downtrend_indicators(self):
        f = build_features(downtrend())
        last = f.iloc[-1]
        assert last["ema_stack_bear"] == 1.0
        assert last["supertrend_dir"] == -1

    def test_last_row_finite_for_rolling(self):
        f = build_features(uptrend())
        for c in ["ema_9", "rsi_14", "adx", "atr_pct", "bb_pos", "zscore_20"]:
            assert np.isfinite(f[c].iloc[-1]), c


class TestRegime:
    def test_uptrend(self):
        assert detect_regime(uptrend()).trend == "up"

    def test_downtrend(self):
        assert detect_regime(downtrend()).trend == "down"

    def test_range(self):
        assert detect_regime(range_frame()).trend == "range"

    def test_alignment(self):
        regimes = {"1h": detect_regime(uptrend()), "4h": detect_regime(uptrend()),
                   "1d": detect_regime(uptrend())}
        assert trend_alignment(regimes) == 1.0
        mixed = {"1h": detect_regime(uptrend()), "4h": detect_regime(downtrend())}
        assert trend_alignment(mixed) == 0.5

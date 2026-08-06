import numpy as np
import pandas as pd
import pytest

from ta_agent import indicators as ta


def make_df(closes, highs=None, lows=None, volume=None):
    n = len(closes)
    highs = highs if highs is not None else np.maximum(closes, np.roll(closes, 1))
    lows = lows if lows is not None else np.minimum(closes, np.roll(closes, 1))
    volume = volume if volume is not None else np.ones(n)
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": volume})


class TestEmaSma:
    def test_ema_span3(self):
        out = ta.ema(pd.Series([1, 2, 3, 4, 5]), 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert out[2] == pytest.approx(2.25)
        assert out[3] == pytest.approx(3.125)
        assert out[4] == pytest.approx(4.0625)

    def test_sma(self):
        out = ta.sma(pd.Series([1, 2, 3, 4, 5]), 3)
        assert out[-1] == pytest.approx(4.0)


class TestRsi:
    def test_up(self):
        out = ta.rsi(pd.Series(np.arange(1.0, 40.0)), 14)
        assert out[-1] == pytest.approx(100.0)

    def test_down(self):
        out = ta.rsi(pd.Series(np.arange(40.0, 1.0, -1.0)), 14)
        assert out[-1] == pytest.approx(0.0)

    def test_flat(self):
        out = ta.rsi(pd.Series(np.full(30, 10.0)), 14)
        assert out[-1] == pytest.approx(50.0)

    def test_bounded(self):
        rng = np.random.default_rng(0)
        out = ta.rsi(pd.Series(rng.normal(size=200)), 14)
        assert np.nanmax(out) <= 100.0 and np.nanmin(out) >= 0.0


class TestMacd:
    def test_line_equals_diff_ema(self):
        s = pd.Series(np.random.default_rng(1).normal(size=300))
        line, sig, hist = ta.macd(s)
        assert line[-1] == pytest.approx(ta.ema(s, 12)[-1] - ta.ema(s, 26)[-1])
        assert hist[-1] == pytest.approx(line[-1] - sig[-1])


class TestAtrAdx:
    def test_atr_constant_range(self):
        closes = np.linspace(100, 200, 50)
        df = make_df(closes, highs=closes + 5.0, lows=closes - 5.0)
        out = ta.atr(df, 14)
        assert out[-1] == pytest.approx(10.0, abs=1e-6)

    def test_adx_uptrend(self):
        closes = np.linspace(10, 60, 100)
        df = make_df(closes, highs=closes + 2, lows=closes - 2)
        adx, plus_di, minus_di = ta.adx(df, 14)
        assert plus_di[-1] > minus_di[-1]
        assert adx[-1] > 0


def df_highs():
    return np.full(50, 110.0)


def df_lows():
    return np.full(50, 100.0)


class TestBandsChannels:
    def test_bollinger(self):
        s = pd.Series(np.random.default_rng(2).normal(100, 5, size=100))
        mid, upper, lower = ta.bollinger(s, 20, 2.0)
        assert np.all(upper[19:] >= mid[19:])
        assert np.all(mid[19:] >= lower[19:])
        assert upper[-1] - mid[-1] == pytest.approx(2 * s.iloc[-20:].std(ddof=0))

    def test_donchian(self):
        df = make_df(np.random.default_rng(3).normal(100, 5, size=100),
                     highs=np.full(100, 110), lows=np.full(100, 90))
        up, lo, mid = ta.donchian(df, 20)
        assert up[-1] == 110.0 and lo[-1] == 90.0


class TestSupertrend:
    def test_uptrend_long(self):
        closes = np.linspace(10, 60, 100)
        df = make_df(closes, highs=closes + 1.5, lows=closes - 1.5)
        st, direction = ta.supertrend(df, 10, 3.0)
        assert direction[-1] == 1
        assert st[-1] < closes[-1]

    def test_downtrend_short(self):
        closes = np.linspace(60, 10, 100)
        df = make_df(closes, highs=closes + 1.5, lows=closes - 1.5)
        st, direction = ta.supertrend(df, 10, 3.0)
        assert direction[-1] == -1
        assert st[-1] > closes[-1]


class TestMisc:
    def test_zscore_flat(self):
        out = ta.zscore(pd.Series(np.full(30, 5.0)), 20)
        assert out[-1] == pytest.approx(0.0)

    def test_swing_points_zigzag(self):
        closes = np.array([10, 9, 8, 9, 12, 15, 13, 11, 14, 17], dtype=float)
        df = make_df(closes, highs=closes + 0.5, lows=closes - 0.5)
        sw = ta.swing_points(df, 2, 2)
        assert sw[2] == -1 and sw[5] == 1 and sw[7] == -1

    def test_vwap_day(self):
        closes = np.array([100.0, 102.0, 104.0])
        df = make_df(closes)
        df["time"] = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        vwap = ta.vwap_day(df)
        assert vwap[-1] == pytest.approx(np.mean(closes), abs=1e-6)

    def test_cvd(self):
        df = make_df(np.array([102.0, 99.0, 103.0]))
        df["open"] = [100.0, 100.0, 100.0]
        df["volume"] = [10.0, 10.0, 10.0]
        out = ta.cvd(df)
        assert out[-1] == pytest.approx(10.0 - 10.0 + 10.0)

    def test_parabolic_sar_uptrend(self):
        closes = np.linspace(10, 60, 100)
        df = make_df(closes, highs=closes + 1, lows=closes - 1)
        sar, trend = ta.parabolic_sar(df)
        assert trend[-1] == 1
        assert sar[-1] < closes[-1]

    def test_ichimoku_basic(self):
        closes = np.linspace(10, 60, 100)
        df = make_df(closes, highs=closes + 1, lows=closes - 1)
        t, k, sa, sb, ch = ta.ichimoku(df)
        assert t[-1] > 0 and k[-1] > 0 and not np.isnan(sa[-1])

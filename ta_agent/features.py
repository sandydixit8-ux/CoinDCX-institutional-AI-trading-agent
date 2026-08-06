"""Feature engineering: build a flat feature vector from an OHLCV frame
for the AI confidence ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a rich feature frame aligned to ``df``.

    Rows where indicators need warm-up will contain NaN and should be dropped
    before ML use.
    """
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)
    out["ret_24"] = close.pct_change(24)

    out["ema_9"] = ta.ema(close, 9)
    out["ema_21"] = ta.ema(close, 21)
    out["ema_50"] = ta.ema(close, 50)
    out["sma_50"] = ta.sma(close, 50)
    out["sma_200"] = ta.sma(close, 200)
    out["ema_9_slope"] = ta.regression_slope(close, 9)
    out["ema_21_slope"] = ta.regression_slope(close, 21)

    out["price_vs_ema9"] = close / (out["ema_9"] + 1e-10) - 1
    out["price_vs_ema21"] = close / (out["ema_21"] + 1e-10) - 1
    out["price_vs_ema50"] = close / (out["ema_50"] + 1e-10) - 1
    out["ema_stack_bull"] = ((out["ema_9"] > out["ema_21"]) & (out["ema_21"] > out["ema_50"])).astype(float)
    out["ema_stack_bear"] = ((out["ema_9"] < out["ema_21"]) & (out["ema_21"] < out["ema_50"])).astype(float)

    out["rsi_14"] = ta.rsi(close, 14)
    out["rsi_7"] = ta.rsi(close, 7)
    macd_line, macd_sig, macd_hist = ta.macd(close)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    out["macd_hist_rise"] = (out["macd_hist"] > out["macd_hist"].shift(1)).astype(float)
    out["macd_above_signal"] = (macd_line > macd_sig).astype(float)

    adx_v, plus_di, minus_di = ta.adx(out)
    out["adx"] = adx_v
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["di_bull"] = (plus_di > minus_di).astype(float)

    out["atr_14"] = ta.atr(out, 14)
    out["atr_pct"] = out["atr_14"] / close
    out["vol_ratio"] = out["atr_pct"] / (out["atr_pct"].rolling(50, min_periods=10).mean() + 1e-10)

    mid, upper, lower = ta.bollinger(close, 20, 2.0)
    out["bb_width"] = (upper - lower) / (mid + 1e-10)
    out["bb_pos"] = (close.to_numpy() - lower) / (upper - lower + 1e-10)

    dc_hi, dc_lo, dc_mid = ta.donchian(out, 20)
    out["dc_hi_20"] = dc_hi
    out["dc_lo_20"] = dc_lo
    out["dc_pos"] = (close.to_numpy() - dc_lo) / (dc_hi - dc_lo + 1e-10)
    out["dc_breakout_up"] = (close.to_numpy() > pd.Series(dc_hi).shift(1).to_numpy()).astype(float)
    out["dc_breakout_dn"] = (close.to_numpy() < pd.Series(dc_lo).shift(1).to_numpy()).astype(float)

    st, st_dir = ta.supertrend(out, 10, 3.0)
    out["supertrend"] = st
    out["supertrend_dir"] = st_dir

    tenkan, kijun, senkou_a, senkou_b, chikou = ta.ichimoku(out)
    out["ich_tenkan"] = tenkan
    out["ich_kijun"] = kijun
    out["ich_senkou_a"] = senkou_a
    out["ich_senkou_b"] = senkou_b
    out["ich_cloud_bull"] = ((close.to_numpy() > senkou_a) & (close.to_numpy() > senkou_b)).astype(float)
    out["ich_cloud_bear"] = ((close.to_numpy() < senkou_a) & (close.to_numpy() < senkou_b)).astype(float)
    out["ich_tenkan_gt_kijun"] = (tenkan > kijun).astype(float)

    sar, sar_trend = ta.parabolic_sar(out)
    out["sar"] = sar
    out["sar_dir"] = sar_trend

    out["vwap_dist"] = close / (ta.vwap_day(out) + 1e-10) - 1
    out["zscore_20"] = ta.zscore(close, 20)
    out["zscore_50"] = ta.zscore(close, 50)

    vol_z = ta.zscore(out["volume"], 20)
    out["volume_z"] = vol_z
    out["volume_expanding"] = (out["volume"] > out["volume"].rolling(20).mean()).astype(float)

    sw = ta.swing_points(out, 5, 5)
    out["swing"] = sw
    struct = ta.last_structure(sw)
    out["bos_up"] = 0.0
    out["bos_down"] = 0.0
    out["cho_up"] = 0.0
    out["cho_down"] = 0.0
    if struct:
        out.loc[out.index[-1], "bos_up"] = float(struct["bos_up"])
        out.loc[out.index[-1], "bos_down"] = float(struct["bos_down"])
        out.loc[out.index[-1], "cho_up"] = float(struct["cho_up"])
        out.loc[out.index[-1], "cho_down"] = float(struct["cho_down"])

    out["cvd"] = ta.cvd(out)
    out["cvd_20"] = out["cvd"].diff(20)
    out["cvd_rising"] = (out["cvd"].diff(5) > 0).astype(float)

    out["body_pct"] = (close - out["open"]) / (out["high"] - out["low"] + 1e-10)
    out["upper_wick_pct"] = (out["high"] - np.maximum(close, out["open"])) / (out["high"] - out["low"] + 1e-10)
    out["lower_wick_pct"] = (np.minimum(close, out["open"]) - out["low"]) / (out["high"] - out["low"] + 1e-10)

    out["hl_range_pct"] = (out["high"] - out["low"]) / close
    out["overnight_gap"] = out["open"] / out["close"].shift(1) - 1

    return out


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "ema_9_slope", "ema_21_slope",
    "price_vs_ema9", "price_vs_ema21", "price_vs_ema50",
    "ema_stack_bull", "ema_stack_bear",
    "rsi_14", "rsi_7", "macd_hist", "macd_hist_rise", "macd_above_signal",
    "adx", "plus_di", "minus_di", "di_bull",
    "atr_pct", "vol_ratio", "bb_width", "bb_pos",
    "dc_pos", "dc_breakout_up", "dc_breakout_dn",
    "supertrend_dir", "ich_cloud_bull", "ich_cloud_bear", "ich_tenkan_gt_kijun",
    "sar_dir", "vwap_dist", "zscore_20", "zscore_50",
    "volume_z", "volume_expanding",
    "bos_up", "bos_down", "cho_up", "cho_down",
    "cvd_20", "cvd_rising",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "hl_range_pct", "overnight_gap",
]

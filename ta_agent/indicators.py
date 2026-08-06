"""Technical indicators implemented on numpy/pandas (no TA-Lib required).

All functions accept pandas Series / DataFrames and return numpy arrays
(except where noted). Inputs must be in chronological order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import timestamps

EPS = 1e-10


def _s(s) -> pd.Series:
    return pd.Series(s, dtype="float64")


def sma(s, n: int) -> np.ndarray:
    return _s(s).rolling(n, min_periods=n).mean().to_numpy()


def ema(s, n: int) -> np.ndarray:
    return _s(s).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def wma(s, n: int) -> np.ndarray:
    s = _s(s)
    weights = np.arange(1, n + 1, dtype=float)
    out = s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return out.to_numpy()


def rsi(s, n: int = 14) -> np.ndarray:
    """Wilder's RSI."""
    s = _s(s)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    flat = (avg_gain <= EPS) & (avg_loss <= EPS)
    rs = avg_gain / (avg_loss + EPS)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[flat] = 50.0
    return out.to_numpy()


def macd(s, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _s(s)
    line = ema(s, fast) - ema(s, slow)
    line_s = pd.Series(line)
    sig = line_s.ewm(span=signal, adjust=False, min_periods=signal).mean().to_numpy()
    hist = line - sig
    return line, sig, hist


def true_range(df: pd.DataFrame) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.to_numpy()


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    tr = pd.Series(true_range(df), index=df.index)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def adx(df: pd.DataFrame, n: int = 14):
    """Return (adx, +di, -di)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.Series(true_range(df), index=df.index)

    def _wilder(x: pd.Series) -> pd.Series:
        return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    atr_s = _wilder(tr)
    plus_di = 100.0 * _wilder(plus_dm) / (atr_s + EPS)
    minus_di = 100.0 * _wilder(minus_dm) / (atr_s + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = _wilder(pd.Series(dx)).to_numpy()
    return adx, plus_di.to_numpy(), minus_di.to_numpy()


def bollinger(s, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    std = _s(s).rolling(n, min_periods=n).std(ddof=0).to_numpy()
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def bollinger_b(s, n: int = 20, k: float = 2.0) -> np.ndarray:
    s = _s(s)
    mid, upper, lower = bollinger(s, n, k)
    width = (upper - lower) + EPS
    return ((s.to_numpy() - lower) / width).astype(float)


def donchian(df: pd.DataFrame, n: int = 20):
    upper = df["high"].rolling(n, min_periods=n).max().to_numpy()
    lower = df["low"].rolling(n, min_periods=n).min().to_numpy()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """Return (supertrend line, direction) where direction=1 long / -1 short."""
    atr_v = atr(df, n)
    hl2 = (df["high"].to_numpy() + df["low"].to_numpy()) / 2.0
    upper_basic = hl2 + mult * atr_v
    lower_basic = hl2 - mult * atr_v
    close = df["close"].to_numpy()
    upper = np.full(len(df), np.nan)
    lower = np.full(len(df), np.nan)
    direction = np.zeros(len(df), dtype=int)
    first = np.where(np.isfinite(atr_v))[0]
    if len(first) == 0:
        return np.full(len(df), np.nan), np.ones(len(df), dtype=int)
    k = int(first[0])
    upper[k] = upper_basic[k]
    lower[k] = lower_basic[k]
    direction[k] = 1 if close[k] >= lower[k] else -1
    st = np.full(len(df), np.nan)
    st[k] = lower[k] if direction[k] == 1 else upper[k]
    for i in range(k + 1, len(df)):
        if close[i] > upper[i - 1]:
            direction[i] = 1
        elif close[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
        st[i] = lower[i] if direction[i] == 1 else upper[i]
    return st, direction


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan_line = ((high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2).to_numpy()
    kijun_line = ((high.rolling(kijun).max() + low.rolling(kijun).min()) / 2).to_numpy()
    senkou_a = ((tenkan_line + kijun_line) / 2)
    senkou_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).to_numpy()
    senkou_a = pd.Series(senkou_a).shift(kijun).to_numpy()
    senkou_b = pd.Series(senkou_b).shift(kijun).to_numpy()
    chikou = close.shift(-kijun).to_numpy()
    return tenkan_line, kijun_line, senkou_a, senkou_b, chikou


def parabolic_sar(df: pd.DataFrame, af0: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    sar = np.full(n, np.nan)
    trend_arr = np.full(n, np.nan)
    af = af0
    trend = 1
    extreme = high[0]
    sar[0] = low[0]
    trend_arr[0] = 1
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (extreme - sar[i - 1])
        if trend == 1:
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar[i]:
                trend = -1
                sar[i] = extreme
                extreme = low[i]
                af = af0
            else:
                if high[i] > extreme:
                    extreme = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar[i]:
                trend = 1
                sar[i] = extreme
                extreme = high[i]
                af = af0
            else:
                if low[i] < extreme:
                    extreme = low[i]
                    af = min(af + af_step, af_max)
        trend_arr[i] = trend
    return sar, trend_arr


def stoch_rsi(s, n: int = 14, k: int = 3, d: int = 3):
    s = _s(s)
    r = pd.Series(rsi(s, n))
    rmin = r.rolling(n).min()
    rmax = r.rolling(n).max()
    stoch = (r - rmin) / (rmax - rmin + EPS)
    k_line = stoch.rolling(k).mean().to_numpy()
    d_line = stoch.rolling(d).mean().to_numpy()
    return k_line, d_line


def vwap_day(df: pd.DataFrame) -> np.ndarray:
    """VWAP anchored to each UTC calendar day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].clip(lower=0)
    tpv = typical * vol
    date = timestamps.col_to_datetime(df["time"]).dt.date if "time" in df.columns else None
    out = np.full(len(df), np.nan)
    if date is None:
        return out
    cum_tpv = pd.Series(tpv).groupby(date).cumsum()
    cum_vol = pd.Series(vol).groupby(date).cumsum()
    return (cum_tpv / (cum_vol + EPS)).to_numpy()


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].clip(lower=0)
    tpv = (typical * vol).rolling(n, min_periods=n).sum()
    v = vol.rolling(n, min_periods=n).sum()
    return (tpv / (v + EPS)).to_numpy()


def zscore(s, n: int = 20) -> np.ndarray:
    s = _s(s)
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0)
    return ((s - mean) / (std + EPS)).to_numpy()


def regression_slope(s, n: int = 20) -> np.ndarray:
    """Slope of linear regression over a rolling window (normalized by price)."""
    s = _s(s)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def _slope(y):
        if np.isnan(y).any():
            return np.nan
        ym = y.mean()
        return ((x - xm) * (y - ym)).sum() / denom

    slopes = s.rolling(n).apply(_slope, raw=True)
    return (slopes / s).to_numpy()


def swing_points(df: pd.DataFrame, left: int = 5, right: int = 5):
    """Return arrays marking swing highs (1) and swing lows (-1)."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    sw = np.zeros(n, dtype=int)
    for i in range(left, n - right):
        win_high = high[i - left : i + right + 1]
        win_low = low[i - left : i + right + 1]
        if high[i] == win_high.max() and (win_high == high[i]).sum() == 1:
            sw[i] = 1
        if low[i] == win_low.min() and (win_low == low[i]).sum() == 1:
            sw[i] = -1
    return sw


def last_structure(sw: np.ndarray):
    """Derive the last two confirmed swings and BOS/CHoCH from swing markers."""
    idx = np.where(sw != 0)[0]
    if len(idx) < 2:
        return None
    s1 = int(idx[-2])
    s2 = int(idx[-1])
    if sw[s1] == sw[s2]:  # need alternating for simple analysis; take previous distinct
        for j in idx[-3::-1]:
            if sw[int(j)] != sw[s2]:
                s1 = int(j)
                break
    bos_up = sw[s2] == 1 and sw[s1] == -1
    bos_down = sw[s2] == -1 and sw[s1] == 1
    return {
        "last_swing": int(sw[s2]),
        "prev_swing": int(sw[s1]),
        "last_swing_idx": int(s2),
        "prev_swing_idx": int(s1),
        "bos_up": bool(bos_up),
        "bos_down": bool(bos_down),
        "cho_up": bool(sw[s2] == 1 and sw[s1] == 1),
        "cho_down": bool(sw[s2] == -1 and sw[s1] == -1),
    }


def fair_value_gaps(df: pd.DataFrame):
    """Return the most recent bullish and bearish FVG objects (or None)."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    fvgs = []
    for i in range(2, n):
        if low[i] > high[i - 2]:  # bullish gap left by 3-candle imbalance
            fvgs.append({"type": "bullish", "top": low[i], "bottom": high[i - 2], "idx": i})
        if high[i] < low[i - 2]:
            fvgs.append({"type": "bearish", "top": low[i - 2], "bottom": high[i], "idx": i})
    if not fvgs:
        return None, None
    bull = [g for g in fvgs if g["type"] == "bullish"]
    bear = [g for g in fvgs if g["type"] == "bearish"]
    last_bull = bull[-1] if bull else None
    last_bear = bear[-1] if bear else None
    return last_bull, last_bear


def volume_profile(df: pd.DataFrame, bins: int = 50):
    """Simple volume profile. Returns (bin_edges, volume, poc, vah, val)."""
    if len(df) < 5:
        return None, None, None, None, None
    lo = df["low"].min()
    hi = df["high"].max()
    if hi - lo <= EPS:
        return None, None, None, None, None
    edges = np.linspace(lo, hi, bins + 1)
    widths = np.diff(edges)
    prof = np.zeros(bins)
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    vol = df["volume"].clip(lower=0).to_numpy()
    for i in range(len(df)):
        idx = int(np.clip((hl2[i] - lo) / (hi - lo) * bins, 0, bins - 1))
        prof[idx] += vol[i]
    poc_bin = int(np.argmax(prof))
    poc = (edges[poc_bin] + edges[poc_bin + 1]) / 2
    total = prof.sum()
    cum = np.cumsum(prof) / (total + EPS)
    # 70% value area around POC
    i_low = int(poc_bin)
    i_high = int(poc_bin)
    covered = prof[poc_bin]
    while covered / (total + EPS) < 0.70:
        left = prof[i_low - 1] if i_low > 0 else -1.0
        right = prof[i_high + 1] if i_high < bins - 1 else -1.0
        if left < 0 and right < 0:
            break
        if left >= right:
            i_low -= 1
            covered += left
        else:
            i_high += 1
            covered += right
    val = edges[i_low]
    vah = edges[i_high + 1]
    return edges, prof, poc, vah, val


def pivot_points(df: pd.DataFrame):
    """Classic pivots from the prior day's H/L/C. Uses last available day."""
    if "time" not in df.columns or len(df) < 2:
        return {}
    d = timestamps.col_to_datetime(df["time"]).dt.date
    keys = list(dict.fromkeys(d))
    last_day = keys[-1]
    mask = d == last_day
    prior = df[~mask]
    if prior.empty:
        return {}
    prev = prior.iloc[-1]
    h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
    p = (h + l + c) / 3
    r1, s1 = 2 * p - l, 2 * p - h
    r2, s2 = p + (h - l), p - (h - l)
    r3, s3 = h + 2 * (p - l), l - 2 * (h - p)
    return {"pivot": p, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def cvd(df: pd.DataFrame) -> np.ndarray:
    """Cumulative volume delta approximation using close/open position."""
    body = df["close"].to_numpy() - df["open"].to_numpy()
    vol = df["volume"].clip(lower=0).to_numpy()
    delta = np.where(body > 0, vol, np.where(body < 0, -vol, 0.0))
    return np.cumsum(delta)


def order_flow_imbalance(book_bids, book_asks, window: int = 10) -> float:
    """Sum order book imbalance over top ``window`` levels: (bid - ask)/(bid + ask)."""
    bid_vol = sum(q for _, q in book_bids[:window])
    ask_vol = sum(q for _, q in book_asks[:window])
    if bid_vol + ask_vol <= EPS:
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)


def market_depth(book_bids, book_asks) -> dict:
    """Depth pressure around mid price (top 5/10/20 levels)."""
    bid_vol = sum(q for _, q in book_bids)
    ask_vol = sum(q for _, q in book_asks)
    return {
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
        "ratio": bid_vol / (ask_vol + EPS),
        "bid_top5": sum(q for _, q in book_bids[:5]),
        "ask_top5": sum(q for _, q in book_asks[:5]),
    }

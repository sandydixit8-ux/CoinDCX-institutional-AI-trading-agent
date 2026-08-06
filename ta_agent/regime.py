"""Market regime detection (trend / range / volatility state).

Used to (a) block counter-trend trades and (b) adjust risk sizing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ta


@dataclass
class Regime:
    trend: str          # "up" | "down" | "range"
    volatility: str     # "low" | "normal" | "high"
    adx: float
    atr_pct: float
    score: float        # +1 (strong up) .. -1 (strong down)


def detect_regime(df: pd.DataFrame, adx_n: int = 14, atr_pct_lo: float = 0.012, atr_pct_hi: float = 0.04) -> Regime:
    """Regime from the latest bar of ``df``."""
    close = df["close"]
    adx_v, plus_di, minus_di = ta.adx(df, adx_n)
    atr_v = ta.atr(df, adx_n)
    atr_pct = float(atr_v[-1] / close.iloc[-1]) if len(atr_v) else 0.0
    ema21 = ta.ema(close, 21)
    ema50 = ta.ema(close, 50)

    price_above_ema21 = close.iloc[-1] > ema21[-1]
    ema21_above_ema50 = ema21[-1] > ema50[-1]
    di_bull = plus_di[-1] > minus_di[-1]
    slope_up = ta.regression_slope(close, 20)[-1] > 0

    if adx_v[-1] >= 25:
        if (price_above_ema21 and ema21_above_ema50) or (di_bull and slope_up):
            trend = "up"
        elif (not price_above_ema21 and not ema21_above_ema50) or (not di_bull and not slope_up):
            trend = "down"
        else:
            trend = "range"
    else:
        trend = "range"

    if atr_pct <= atr_pct_lo:
        volatility = "low"
    elif atr_pct >= atr_pct_hi:
        volatility = "high"
    else:
        volatility = "normal"

    score = 0.0
    if trend == "up":
        score = float(adx_v[-1]) / 50.0
        score = min(score + (0.2 if di_bull else 0.0) + (0.2 if price_above_ema21 else 0.0), 1.0)
    elif trend == "down":
        score = -float(adx_v[-1]) / 50.0
        score = max(score - (0.2 if not di_bull else 0.0) - (0.2 if not price_above_ema21 else 0.0), -1.0)

    return Regime(trend=trend, volatility=volatility, adx=float(adx_v[-1]),
                  atr_pct=atr_pct, score=score)


def multi_tf_regime(frames: dict) -> dict:
    """Regime across a dict of {timeframe: df}. Returns per-TF regimes."""
    return {tf: detect_regime(df) for tf, df in frames.items()}


def trend_alignment(regimes: dict) -> float:
    """Fraction of timeframes whose trend matches the highest-timeframe trend."""
    tfs = list(regimes.keys())
    if not tfs:
        return 0.0
    primary = regimes[tfs[-1]].trend if tfs else "range"
    if primary == "range":
        return 0.0
    aligned = sum(1 for r in regimes.values() if r.trend == primary)
    return aligned / len(regimes)

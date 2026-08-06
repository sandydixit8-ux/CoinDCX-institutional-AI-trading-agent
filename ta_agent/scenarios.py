"""Regime scenario market generators for validation backtests.

Each generator produces a minute-level price path (with warmup history so
indicators are primed) and resamples it to the configured timeframes.

Scenarios cover the market regimes the spec asks to test against: bull, bear,
sideways, high volatility, low/high liquidity, flash crash, pump & dump,
black swan, and a mixed-regime sequence for adaptation testing.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .datafeed import SyntheticFeed
from .settings import Settings

log = logging.getLogger("ta_agent.scenarios")

# Scenario dynamics.  ``growth`` is the net price multiplier the scenario aims
# for over the *test window* (normalised per duration so a 1-day bull and a
# 365-day bull are comparable).  ``bar_range`` widens/tightens per-bar
# high-low range (liquidity proxy); ``vol_scale`` scales volume depth.
SCENARIOS: Dict[str, dict] = {
    "bull":            {"growth": 3.0,   "sigma": 0.0018, "bar_range": 0.004, "vol_scale": 1.0},
    "bear":            {"growth": 1 / 3.0, "sigma": 0.0018, "bar_range": 0.004, "vol_scale": 1.0},
    "sideways":        {"growth": 1.0,   "sigma": 0.0009, "bar_range": 0.002, "vol_scale": 1.0},
    "high_volatility": {"growth": 1.0,   "sigma": 0.0040, "bar_range": 0.010, "vol_scale": 1.2},
    "low_liquidity":   {"growth": 1.15,  "sigma": 0.0018, "bar_range": 0.012, "vol_scale": 0.25},
    "high_liquidity":  {"growth": 1.15,  "sigma": 0.0015, "bar_range": 0.002, "vol_scale": 3.0},
    "flash_crash":     {"growth": 1.0,   "sigma": 0.0015, "bar_range": 0.004, "vol_scale": 1.0,
                        "event": ("crash", 0.60, 0.28)},
    "pump_dump":       {"growth": 1.0,   "sigma": 0.0015, "bar_range": 0.004, "vol_scale": 1.0,
                        "event": ("pump_dump", 0.50, 0.35)},
    "black_swan":      {"growth": 1.0,   "sigma": 0.0020, "bar_range": 0.005, "vol_scale": 1.1,
                        "event": ("shock", 0.70, 0.35)},
    "mixed_regimes":   {"mode": "mixed"},
}

_MINUTES_PER_DAY = 1440
_WARMUP_SIGMA = 0.0018
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _coin_seed(seed: int, coin: str) -> int:
    digest = hashlib.md5(coin.encode("utf-8")).hexdigest()[:8]
    return seed + int(digest, 16) % 100_000


def _build_returns(test_minutes: int, warmup_minutes: int, params: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Per-minute log-returns implementing the scenario's drift / shocks."""
    total = test_minutes + warmup_minutes
    mu = np.zeros(total)
    sigma = np.full(total, _WARMUP_SIGMA)

    def apply_test(mu, sigma):
        if params.get("mode") == "mixed":
            seg = max(test_minutes // 4, 1)
            seq = [("bull", 1.5), ("bear", 0.7), ("sideways", 1.0), ("high_volatility", 1.0)]
            pos = warmup_minutes
            for q, (name, growth) in enumerate(seq):
                m = seg if q < 3 else test_minutes - 3 * seg
                sp = SCENARIOS[name]
                mu[pos:pos + m] = np.log(growth) / m
                sigma[pos:pos + m] = sp["sigma"]
                pos += m
            return
        growth = params["growth"]
        mu[warmup_minutes:] = np.log(growth) / test_minutes
        sigma[warmup_minutes:] = params["sigma"]
        event = params.get("event")
        if event:
            kind, frac, size = event
            ev_start = warmup_minutes + int(frac * test_minutes)
            if kind == "crash":
                span = max(int(0.02 * test_minutes), 30)
                mu[ev_start:ev_start + span] = np.log(1 - size) / span
            elif kind == "shock":
                span = max(int(0.01 * test_minutes), 15)
                mu[ev_start:ev_start + span] = np.log(1 - size) / span
                sigma[ev_start:ev_start + span * 3] = 0.012
            elif kind == "pump_dump":
                span = max(int(0.02 * test_minutes), 30)
                mu[ev_start:ev_start + span] = np.log(1 + size) / span
                dump = ev_start + span
                mu[dump:dump + span] = (np.log(1 - size) - np.log(1 + size)) / span

    apply_test(mu, sigma)
    return rng.normal(mu, sigma)


def _minute_frame(price: np.ndarray, n_minutes: int, rng: np.random.Generator,
                  params: dict, end: Optional[str] = None) -> pd.DataFrame:
    bar_range = params.get("bar_range", 0.004)
    vol_scale = params.get("vol_scale", 1.0)
    end = (pd.Timestamp(end).floor("5min") if end
           else pd.Timestamp.now("UTC").floor("5min"))
    start = end - pd.Timedelta(minutes=n_minutes)
    dt_idx = pd.date_range(start=start, periods=n_minutes, freq="min", tz="UTC")
    volume = rng.lognormal(mean=12, sigma=1.0, size=n_minutes) * vol_scale * (price / price[0])
    open_ = np.roll(price, 1)
    open_[0] = price[0]
    close = price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, bar_range, n_minutes))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, bar_range, n_minutes))
    time_ms = (dt_idx - _EPOCH).total_seconds().astype("int64") * 1000
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume, "time": time_ms})


def generate_market(settings: Settings, scenario: str, test_days: int,
                    coins: list, seed: int = 7, warmup_days: int = 30,
                    end: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Build a full multi-timeframe market for one scenario.

    ``end`` pins the market's last bar to a fixed UTC instant (default: now) so
    validation runs are reproducible regardless of wall-clock time.

    Returns ``{coin: {tf: DataFrame}}`` with ascending UTC ``time`` columns,
    matching the shape the backtester consumes.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; use one of {sorted(SCENARIOS)}")
    params = dict(SCENARIOS[scenario])
    warmup_minutes = warmup_days * _MINUTES_PER_DAY
    test_minutes = test_days * _MINUTES_PER_DAY
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for coin in coins:
        rng = np.random.default_rng(_coin_seed(seed, coin))
        rets = _build_returns(test_minutes, warmup_minutes, params, rng)
        base_price = SyntheticFeed._BASE_PRICE.get(coin, 1.0)
        price = base_price * np.exp(np.cumsum(rets))
        base = _minute_frame(price, test_minutes + warmup_minutes, rng, params, end=end)
        tfs = {tf: SyntheticFeed._resample(base, tf) for tf in settings.timeframes}
        out[coin] = tfs
    return out

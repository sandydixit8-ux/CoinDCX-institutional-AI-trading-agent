"""Data feeds: live CoinDCX and synthetic (for paper/demo/offline use)."""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .coindcx_client import CoinDCXClient
from .indicators import market_depth, order_flow_imbalance
from .settings import Settings
from .timestamps import datetime_to_ms, series_to_ms

log = logging.getLogger("ta_agent.feed")


class CoinDCXFeed:
    def __init__(self, client: CoinDCXClient, settings: Settings):
        self.c = client
        self.s = settings

    def available_coins(self) -> list:
        avail = self.c.supported_coins(self.s.quote)
        return [c for c in self.s.watchlist if c in avail] or list(self.s.watchlist)

    def get_frames(self, coins: Optional[list] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        coins = coins or self.available_coins()
        out: Dict[str, Dict[str, pd.DataFrame]] = {}
        for coin in coins:
            pair = self.c.futures_pair_for(coin, self.s.quote) if self.s.market_type == "futures" \
                else self.c.spot_market_for(coin, self.s.quote)
            tfs: Dict[str, pd.DataFrame] = {}
            try:
                for tf in self.s.timeframes:
                    bars = int(self.s.max_data_bars.get(tf, 1000))
                    tfs[tf] = self.c.get_candles_since(pair, interval=tf, bars=bars)
                out[coin] = tfs
            except Exception as exc:  # pragma: no cover
                log.warning("fetch failed for %s: %s", coin, exc)
        return out

    def get_micro(self, coin: str) -> Dict[str, float]:
        pair = self.c.futures_pair_for(coin, self.s.quote)
        out: Dict[str, float] = {"ob_imbalance": 0.0, "depth_ratio": 1.0,
                                 "long_short_ratio": 1.0, "funding_proxy": 0.0}
        try:
            bids, asks = self.c.get_orderbook_levels(pair, depth=25)
            out["ob_imbalance"] = order_flow_imbalance(bids, asks, 10)
            depth = market_depth(bids, asks)
            out["depth_ratio"] = depth["ratio"]
            stats = self.c.get_futures_stats(pair)
            vp = stats.get("position", {}).get("value_percent", {})
            long_p = float(vp.get("long", 50.0))
            short_p = float(vp.get("short", 50.0))
            out["long_short_ratio"] = (long_p / short_p) if short_p > 0 else 1.0
        except Exception as exc:  # pragma: no cover
            log.debug("micro failed for %s: %s", coin, exc)
        return out

    def get_price(self, coin: str) -> Optional[float]:
        pair = self.c.futures_pair_for(coin, self.s.quote) if self.s.market_type == "futures" \
            else self.c.spot_market_for(coin, self.s.quote)
        try:
            df = self.c.get_candles(pair, interval="1m", limit=1)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as exc:  # pragma: no cover
            log.debug("price failed for %s: %s", coin, exc)
        return None


class SyntheticFeed:
    """Geometric-Brownian-Motion candles for offline paper trading / demos.

    Each ``get_frames`` call advances the simulated clock by one trade-timeframe
    step, so position management sees fresh bars every cycle.
    """

    _BASE_PRICE = {"BTC": 60_000.0, "ETH": 1_800.0, "SOL": 70.0, "BNB": 300.0,
                   "XRP": 0.5, "DOGE": 0.08, "LINK": 10.0, "ADA": 0.35,
                   "AVAX": 20.0, "SUI": 0.8, "ARB": 0.5, "OP": 1.0,
                   "PEPE": 0.00001, "SHIB": 0.000008}

    def __init__(self, settings: Settings, seed: int = 7):
        self.s = settings
        self.rng = np.random.default_rng(seed)
        self._base: Dict[str, pd.DataFrame] = {}
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    def _make_base(self, coin: str) -> pd.DataFrame:
        bars = max(self.s.max_data_bars.values()) if self.s.max_data_bars else 1000
        tf_mins = self._tf_minutes(self.s.trade_timeframe)
        n = max(60 * 24 * 60, int(bars) * tf_mins * 2)
        dt_idx = pd.date_range(end=pd.Timestamp.now("UTC").floor("min"),
                               periods=n, freq="min", tz="UTC")
        base_price = self._BASE_PRICE.get(coin, 1.0)
        mu, sigma = 0.0004, 0.0018
        rets = self.rng.normal(mu, sigma, n)
        # add mild cycles so trends appear
        trend = 0.0008 * np.sin(np.linspace(0, 12 * np.pi, n))
        rets = rets + trend
        price = base_price * np.exp(np.cumsum(rets))
        volume = self.rng.lognormal(mean=12, sigma=1.0, size=n) * (price / base_price)
        open_ = np.roll(price, 1)
        open_[0] = price[0]
        close = price
        high = np.maximum(open_, close) * (1 + self.rng.uniform(0, 0.004, n))
        low = np.minimum(open_, close) * (1 - self.rng.uniform(0, 0.004, n))
        return pd.DataFrame({"open": open_, "high": high, "low": low,
                             "close": close, "volume": volume,
                             "time": (dt_idx - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds().astype("int64") * 1000})

    def _extend_base(self, base: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Append ``minutes`` of fresh 1m bars, dropping the oldest to keep the
        history window bounded."""
        last_ts = int(base["time"].iloc[-1])
        n = len(base)
        prev_close = float(base["close"].iloc[-1])
        dt_idx = pd.date_range(start=pd.to_datetime(last_ts, unit="ms", utc=True) + pd.Timedelta(minutes=1),
                               periods=minutes, freq="min", tz="UTC")
        rets = self.rng.normal(0.0004, 0.0018, minutes)
        price = prev_close * np.exp(np.cumsum(rets))
        volume = self.rng.lognormal(mean=12, sigma=1.0, size=minutes) * (price / prev_close)
        open_ = np.roll(price, 1)
        open_[0] = prev_close
        close = price
        high = np.maximum(open_, close) * (1 + self.rng.uniform(0, 0.004, minutes))
        low = np.minimum(open_, close) * (1 - self.rng.uniform(0, 0.004, minutes))
        new = pd.DataFrame({"open": open_, "high": high, "low": low,
                            "close": close, "volume": volume,
                            "time": (dt_idx - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds().astype("int64") * 1000})
        out = pd.concat([base, new], ignore_index=True)
        return out.iloc[-n:].reset_index(drop=True)

    def _build_tfs(self, base: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {tf: self._resample(base, tf) for tf in self.s.timeframes}

    def _make(self, coin: str) -> Dict[str, pd.DataFrame]:
        base = self._make_base(coin)
        self._base[coin] = base
        return self._build_tfs(base)

    @staticmethod
    def _tf_minutes(tf: str) -> int:
        return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
                "1d": 1440, "1w": 10080, "1M": 43200}.get(tf, 60)

    @staticmethod
    def _resample(base: pd.DataFrame, tf: str) -> pd.DataFrame:
        mins = SyntheticFeed._tf_minutes(tf)
        ts = pd.to_datetime(base["time"], unit="ms", utc=True)
        s = base.set_index(ts)
        g = s.resample(f"{mins}min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        g = g.dropna()
        out = g.reset_index()
        out["time"] = datetime_to_ms(out[out.columns[0]])
        out = out[["open", "high", "low", "close", "volume", "time"]].reset_index(drop=True)
        return out

    def get_frames(self, coins: Optional[list] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        coins = coins or self.s.watchlist
        step = self._tf_minutes(self.s.trade_timeframe)
        for coin in coins:
            if coin not in self._cache:
                self._cache[coin] = self._make(coin)
            else:
                base = self._base.get(coin)
                if base is not None:
                    self._base[coin] = self._extend_base(base, step)
                    self._cache[coin] = self._build_tfs(self._base[coin])
        return {c: self._cache[c] for c in coins}

    def available_coins(self) -> list:
        return list(self.s.watchlist)

    def get_micro(self, coin: str) -> Dict[str, float]:
        return {"ob_imbalance": 0.05 * self.rng.normal(), "depth_ratio": 1.0,
                "long_short_ratio": 1.0, "funding_proxy": 0.0}

    def get_price(self, coin: str) -> Optional[float]:
        frames = self._cache.get(coin)
        if not frames:
            frames = self._make(coin)
            self._cache[coin] = frames
        df = frames["1m"]
        return float(df["close"].iloc[-1]) if not df.empty else None

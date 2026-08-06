"""CoinDCX REST API client (public + private, spot + futures).

Public endpoints are unauthenticated; private endpoints sign the JSON body
with HMAC-SHA256 and pass ``X-AUTH-APIKEY`` / ``X-AUTH-SIGNATURE`` headers.

Reference: https://docs.coindcx.com/
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

log = logging.getLogger("ta_agent.coindcx")

# The /market_data/candles endpoint natively serves only these intervals.
NATIVE_CANDLE_INTERVALS = {"1m", "15m", "1h", "1d"}
# Unsupported intervals are derived client-side by resampling a native interval
# that divides them. Mapping: requested interval -> (source interval, multiplier).
CANDLE_RESAMPLE_SOURCES = {
    "5m": ("1m", 5), "30m": ("15m", 2), "2h": ("1h", 2), "4h": ("1h", 4),
    "6h": ("1h", 6), "8h": ("1h", 8), "3d": ("1d", 3), "1w": ("1d", 7),
    "1M": ("1d", 30),
}
SUPPORTED_INTERVALS = set(NATIVE_CANDLE_INTERVALS) | set(CANDLE_RESAMPLE_SOURCES)

_INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
    "6h": 360, "8h": 480, "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}
MAX_CANDLE_LIMIT = 1000


class CoinDCXError(RuntimeError):
    pass


class CoinDCXClient:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://api.coindcx.com",
        timeout: float = 20.0,
        rate_sleep: float = 0.25,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_sleep = rate_sleep
        self.session = requests.Session()
        self._markets_details: Optional[List[dict]] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise CoinDCXError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _signed_body(self, body: dict) -> dict:
        if not self.api_key or not self.api_secret:
            raise CoinDCXError("API key/secret missing for authenticated request")
        body = dict(body)
        if "timestamp" not in body:
            body["timestamp"] = int(round(time.time() * 1000))
        return body

    def _sign(self, body: dict) -> Tuple[str, str]:
        payload = json.dumps(body, separators=(",", ":"))
        signature = hmac.new(
            self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return payload, signature

    def _post_signed(self, path: str, body: dict) -> Any:
        body = self._signed_body(body)
        payload, signature = self._sign(body)
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
        }
        r = self.session.post(url, data=payload, headers=headers, timeout=self.timeout)
        if r.status_code >= 400:
            raise CoinDCXError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get_signed(self, path: str) -> Any:
        body = self._signed_body({})
        payload, signature = self._sign(body)
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
        }
        # CoinDCX signed GETs (e.g. futures wallet details) require the signed
        # JSON body to be sent along with the request; the signature is
        # validated against the received body, so omitting it yields 401.
        r = self.session.get(url, data=payload, headers=headers, timeout=self.timeout)
        if r.status_code >= 400:
            raise CoinDCXError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    # ------------------------------------------------------------------
    # public market data
    # ------------------------------------------------------------------
    def get_markets_details(self, force: bool = False) -> List[dict]:
        if self._markets_details is None or force:
            self._markets_details = self._get("/exchange/v1/markets_details")
        return self._markets_details

    def get_ticker(self) -> List[dict]:
        return self._get("/exchange/ticker")

    def get_active_futures_instruments(self, margin_currency: str = "USDT") -> List[str]:
        params = {f"margin_currency_short_name[]": margin_currency}
        return self._get("/exchange/v1/derivatives/futures/data/active_instruments", params=params)

    def get_futures_instrument(self, pair: str, margin_currency: str = "USDT") -> dict:
        params = {"pair": pair, "margin_currency_short_name": margin_currency}
        return self._get("/exchange/v1/derivatives/futures/data/instrument", params=params)

    def get_futures_stats(self, pair: str) -> dict:
        """Price change, high/low and trader long/short positioning."""
        return self._post_signed(f"/api/v1/derivatives/futures/data/stats?pair={pair}", {"timestamp": int(time.time() * 1000)})

    def get_orderbook(self, pair: str) -> Dict[str, Dict[str, str]]:
        return self._get("/market_data/orderbook", params={"pair": pair})

    def get_orderbook_levels(self, pair: str, depth: int = 25) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return (bids, asks) as (price, qty) tuples sorted best-first."""
        book = self.get_orderbook(pair)
        bids = sorted(((float(p), float(q)) for p, q in book.get("bids", {}).items()), reverse=True)[:depth]
        asks = sorted(((float(p), float(q)) for p, q in book.get("asks", {}).items()))[:depth]
        return bids, asks

    def get_trade_history(self, pair: str, limit: int = 100) -> List[dict]:
        return self._get("/market_data/trade_history", params={"pair": pair, "limit": limit})

    def get_candles(
        self,
        pair: str,
        interval: str = "1m",
        limit: int = 500,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """OHLCV candles as an ascending DataFrame indexed by UTC timestamp.

        CoinDCX returns candles sorted descending; we re-sort ascending.
        """
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}; use one of {sorted(SUPPORTED_INTERVALS)}")
        params = {"pair": pair, "interval": interval, "limit": min(limit, MAX_CANDLE_LIMIT)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = self._get("/market_data/candles", params=params)
        time.sleep(self.rate_sleep)
        if not data:
            return self._empty_candles()
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _empty_candles(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "time"])

    def get_candles_since(
        self, pair: str, interval: str = "1m", bars: int = 1000,
        end_ms: Optional[int] = None, max_requests: int = 30,
    ) -> pd.DataFrame:
        """Fetch ``bars`` candles ending before ``end_ms``.

        Unsupported intervals (e.g. 5m, 4h) are resampled from a native interval.
        """
        if interval not in CANDLE_RESAMPLE_SOURCES:
            return self._fetch_paginated(pair, interval, bars, end_ms, max_requests)
        source, ratio = CANDLE_RESAMPLE_SOURCES[interval]
        raw = self._fetch_paginated(pair, source, bars * ratio + 1, end_ms, max_requests)
        if raw.empty:
            return self._empty_candles()
        out = self._resample_candles(raw, source, interval)
        return out.iloc[-bars:].reset_index(drop=True)

    def _fetch_paginated(
        self, pair: str, interval: str, count: int,
        end_ms: Optional[int], max_requests: int,
    ) -> pd.DataFrame:
        """Backward pagination using BOTH startTime and endTime.

        CoinDCX ignores ``endTime`` unless ``startTime`` is also supplied, so a
        plain ``endTime`` cursor loop would return the same newest 1000 bars
        every iteration.
        """
        step_ms = _INTERVAL_MINUTES.get(interval, 60) * 60_000
        cursor = end_ms if end_ms is not None else int(time.time() * 1000)
        chunks: List[pd.DataFrame] = []
        remaining = count
        for _ in range(max_requests):
            if remaining <= 0:
                break
            take = min(remaining, MAX_CANDLE_LIMIT)
            df = self.get_candles(pair, interval=interval, limit=take,
                                  start_ms=cursor - take * step_ms, end_ms=cursor)
            if df.empty:
                break
            chunks.append(df)
            cursor = int(df["time"].min().timestamp() * 1000) - step_ms
            remaining -= len(df)
            if len(df) < take:
                break
        if not chunks:
            return self._empty_candles()
        out = pd.concat(chunks, ignore_index=True)
        out = out.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        return out.iloc[-count:].reset_index(drop=True)

    @staticmethod
    def _resample_candles(df: pd.DataFrame, source_interval: str, target_interval: str) -> pd.DataFrame:
        """Aggregate ``source_interval`` bars into ``target_interval`` bars.

        Drops the trailing bucket when it is not fully formed (matching the
        native endpoint, which only returns completed candles).
        """
        minutes = _INTERVAL_MINUTES[target_interval]
        g = df.set_index("time").resample(
            f"{minutes}min", label="left", closed="left"
        ).agg({"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}).dropna()
        if len(g):
            last_src = df["time"].iloc[-1]
            bucket_end = g.index[-1] + pd.Timedelta(minutes=minutes)
            src_end = last_src + pd.Timedelta(minutes=_INTERVAL_MINUTES[source_interval])
            if src_end < bucket_end:
                g = g.iloc[:-1]
        out = g.reset_index()
        out = out[["open", "high", "low", "close", "volume", "time"]].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # private: spot
    # ------------------------------------------------------------------
    def get_spot_balances(self) -> List[dict]:
        return self._post_signed("/exchange/v1/users/balances", {})

    def create_spot_order(
        self,
        market: str,
        side: str,
        quantity: float,
        order_type: str = "market_order",
        price: Optional[float] = None,
    ) -> dict:
        body = {
            "side": side,
            "order_type": order_type,
            "market": market,
            "total_quantity": quantity,
            "timestamp": int(round(time.time() * 1000)),
        }
        if price is not None:
            body["price_per_unit"] = float(price)
        return self._post_signed("/exchange/v1/orders/create", body)

    def get_spot_order_status(self, id_: str) -> dict:
        return self._post_signed("/exchange/v1/orders/status", {"id": id_})

    # ------------------------------------------------------------------
    # private: futures
    # ------------------------------------------------------------------
    def get_futures_wallets(self) -> List[dict]:
        return self._get_signed("/exchange/v1/derivatives/futures/wallets")

    def create_futures_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "market_order",
        price: Optional[float] = None,
        leverage: float = 3.0,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        margin_currency: str = "USDT",
    ) -> dict:
        if order_type not in ("market_order", "limit_order"):
            raise ValueError("Futures create supports market_order/limit_order")
        order = {
            "side": side,
            "pair": pair,
            "order_type": order_type,
            "total_quantity": float(quantity),
            "leverage": int(leverage),
            "notification": "no_notification",
            "hidden": False,
            "post_only": False,
            "margin_currency_short_name": [margin_currency],
        }
        if order_type == "limit_order":
            if price is None:
                raise ValueError("limit_order requires price")
            order["price"] = str(float(price))
            order["time_in_force"] = "good_till_cancel"
        if take_profit_price is not None:
            order["take_profit_price"] = float(take_profit_price)
        if stop_loss_price is not None:
            order["stop_loss_price"] = float(stop_loss_price)
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "order": order,
        }
        res = self._post_signed("/exchange/v1/derivatives/futures/orders/create", body)
        if isinstance(res, list) and res:
            return res[0]
        return res

    def get_futures_positions(self, margin_currency: str = "USDT", page: int = 1, size: int = 100) -> List[dict]:
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "page": str(page),
            "size": str(size),
            "margin_currency_short_name": [margin_currency],
        }
        return self._post_signed("/exchange/v1/derivatives/futures/positions", body)

    def get_futures_orders_list(
        self, side: str = "both", status: str = "open", margin_currency: str = "USDT",
        page: int = 1, size: int = 50,
    ) -> List[dict]:
        """Open/closed futures orders.

        NOTE: the orders-list API rejects ``side: "both"`` (422 "Side Filter
        values are incorrect") — query buy and sell separately.
        """
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "status": status,
            "side": side,
            "page": str(page),
            "size": str(size),
            "margin_currency_short_name": [margin_currency],
        }
        return self._post_signed("/exchange/v1/derivatives/futures/orders", body)

    def exit_futures_position(
        self, position_id: str, pair: str, quantity: float, margin_currency: str = "USDT",
    ) -> dict:
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "position_id": position_id,
            "pair": pair,
            "total_quantity": float(quantity),
            "margin_currency_short_name": margin_currency,
        }
        return self._post_signed("/exchange/v1/derivatives/futures/positions/exit", body)

    def cancel_futures_order(self, order_id: str, margin_currency: str = "USDT") -> dict:
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "order_id": order_id,
            "margin_currency_short_name": margin_currency,
        }
        return self._post_signed("/exchange/v1/derivatives/futures/orders/cancel", body)

    def create_futures_tpsl(
        self,
        pair: str,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        quantity: float,
        margin_currency: str = "USDT",
        position_id: Optional[str] = None,
    ) -> dict:
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "pair": pair,
            "side": side,
            "take_profit_price": float(take_profit_price),
            "stop_loss_price": float(stop_loss_price),
            "total_quantity": float(quantity),
            "margin_currency_short_name": margin_currency,
        }
        if position_id:
            body["position_id"] = position_id
        return self._post_signed("/exchange/v1/derivatives/futures/positions/create_tpsl", body)

    def transfer_wallet(self, source: str, destination: str, currency: str, amount: float) -> dict:
        body = {
            "timestamp": int(round(time.time() * 1000)),
            "source_wallet_type": source,
            "destination_wallet_type": destination,
            "currency_short_name": currency,
            "amount": float(amount),
        }
        return self._post_signed("/exchange/v1/wallets/transfer", body)

    # ------------------------------------------------------------------
    # market discovery
    # ------------------------------------------------------------------
    def futures_pair_for(self, coin: str, quote: str = "USDT") -> str:
        return f"B-{coin}_{quote}"

    def spot_market_for(self, coin: str, quote: str = "USDT") -> str:
        return f"{coin}{quote}"

    def available_futures_coins(self, quote: str = "USDT") -> set:
        try:
            insts = self.get_active_futures_instruments(quote)
        except CoinDCXError as exc:  # pragma: no cover - network dependent
            log.warning("Could not fetch active futures instruments: %s", exc)
            return set()
        coins = set()
        for i in insts:
            if i.startswith("B-") and i.endswith(f"_{quote}"):
                coins.add(i[2 : -len(f"_{quote}")])
        return coins

    def supported_coins(self, quote: str = "USDT") -> set:
        avail = self.available_futures_coins(quote)
        return avail

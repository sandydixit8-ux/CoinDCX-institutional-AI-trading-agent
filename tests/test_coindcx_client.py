import time

import pandas as pd
import pytest

from ta_agent.coindcx_client import CoinDCXClient


class _Capture:
    def __init__(self):
        self.bodies = []

    def __call__(self, path, body):
        self.bodies.append((path, body))
        return {"id": "test-id", "status": "open", "avg_price": None}


def test_futures_order_body_matches_documented_schema():
    c = CoinDCXClient(api_key="k", api_secret="s")
    cap = _Capture()
    c._post_signed = cap
    res = c.create_futures_order(
        pair="B-BTC_USDT", side="buy", quantity=0.01, order_type="market_order",
        leverage=3, take_profit_price=70000.0, stop_loss_price=55000.0,
        margin_currency="USDT",
    )
    assert res["id"] == "test-id"
    path, body = cap.bodies[0]
    assert path == "/exchange/v1/derivatives/futures/orders/create"
    # timestamp must be epoch MILLISECONDS (13 digits in 2026)
    assert body["timestamp"] > 1_700_000_000_000
    assert body["timestamp"] <= int(time.time() * 1000) + 5000
    order = body["order"]
    assert order["leverage"] == 3 and isinstance(order["leverage"], int)
    assert order["margin_currency_short_name"] == ["USDT"]
    assert order["order_type"] == "market_order"
    assert order["take_profit_price"] == 70000.0
    assert order["stop_loss_price"] == 55000.0
    assert "margin_currency_short_name" not in body


def test_futures_order_limit_requires_price():
    c = CoinDCXClient(api_key="k", api_secret="s")
    cap = _Capture()
    c._post_signed = cap
    with pytest.raises(ValueError):
        c.create_futures_order(pair="B-BTC_USDT", side="buy", quantity=0.01,
                               order_type="limit_order")
    c.create_futures_order(pair="B-BTC_USDT", side="buy", quantity=0.01,
                           order_type="limit_order", price=50000.0)
    order = cap.bodies[0][1]["order"]
    assert order["time_in_force"] == "good_till_cancel"
    assert order["price"] == "50000.0"


def test_get_futures_orders_list_body_shape():
    c = CoinDCXClient(api_key="k", api_secret="s")
    cap = _Capture()
    c._post_signed = cap
    c.get_futures_orders_list(side="buy", status="open")
    path, body = cap.bodies[0]
    assert path == "/exchange/v1/derivatives/futures/orders"
    assert body["side"] == "buy"
    assert body["status"] == "open"
    assert body["margin_currency_short_name"] == ["USDT"]


class _FakeCandles:
    """Stand-in for GET /market_data/candles.

    Serves minute-aligned 1m bars over a fixed window and records every request
    so tests can assert how the client paginates.
    """

    def __init__(self, minutes: int = 3000):
        # start on a 5-minute boundary so resampled buckets align to :00/:05
        now_ms = int(time.time() * 1000)
        aligned = now_ms - (now_ms % (5 * 60 * 1000))
        self.start = aligned - minutes * 60 * 1000
        self.minutes = minutes
        self.calls = []

    def _rows(self):
        idx = pd.date_range(start=pd.Timestamp(self.start, unit="ms", tz="UTC"),
                            periods=self.minutes, freq="min")
        rows = []
        for i, ts in enumerate(idx):
            o = 100.0 + i * 0.01
            rows.append({"time": int(ts.timestamp() * 1000), "open": o, "high": o + 0.5,
                         "low": o - 0.5, "close": o + 0.1, "volume": 10.0})
        return rows

    def __call__(self, path, params=None):
        p = dict(params or {})
        self.calls.append((path, p))
        assert ("endTime" in p) == ("startTime" in p), "endTime must be paired with startTime"
        s = int(p.get("startTime", 0))
        e = int(p.get("endTime", 1 << 62))
        lim = int(p.get("limit", 1000))
        rows = [r for r in self._rows() if s <= r["time"] <= e]
        return rows[-lim:]


def test_pagination_sends_start_and_end_time():
    c = CoinDCXClient(rate_sleep=0)
    fake = _FakeCandles(minutes=3000)
    c._get = fake
    end_ms = fake.start + (fake.minutes - 1) * 60_000
    df = c.get_candles_since("B-BTC_USDT", "1m", bars=2000, end_ms=end_ms)
    assert len(df) == 2000
    assert df["time"].is_monotonic_increasing
    assert len(fake.calls) >= 2
    for _, params in fake.calls:
        assert params["interval"] == "1m"


def test_get_candles_since_resamples_unsupported_interval():
    c = CoinDCXClient(rate_sleep=0)
    fake = _FakeCandles(minutes=3000)
    c._get = fake
    end_ms = fake.start + (fake.minutes - 1) * 60_000
    df = c.get_candles_since("B-BTC_USDT", "5m", bars=60, end_ms=end_ms)
    assert len(df) == 60
    assert all(t.minute % 5 == 0 for t in df["time"].iloc[:3])
    # aggregate check on a middle bucket vs the raw 1m bars
    ts = int(df["time"].iloc[30].timestamp() * 1000)
    src = pd.DataFrame([r for r in fake._rows() if ts <= r["time"] < ts + 5 * 60_000])
    row = df.iloc[30]
    assert row["open"] == src["open"].iloc[0]
    assert row["close"] == src["close"].iloc[-1]
    assert row["high"] == src["high"].max()
    assert row["low"] == src["low"].min()
    assert row["volume"] == src["volume"].sum()

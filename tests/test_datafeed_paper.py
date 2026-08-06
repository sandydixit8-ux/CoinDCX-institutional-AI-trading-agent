import pytest

from ta_agent.brokers import PaperBroker
from ta_agent.datafeed import SyntheticFeed
from ta_agent.settings import Settings


@pytest.fixture()
def loose_settings():
    s = Settings.load("config.json")
    s.mode = "paper"
    s.risk["min_confidence"] = 0.0
    s.risk["min_rr"] = 0.5
    return s


def test_paper_broker_market_order_needs_mark_prices(loose_settings):
    broker = PaperBroker(loose_settings)
    order = broker.place_order("B-BTC_USDT", "buy", 0.01, order_type="market_order")
    assert order.status == "rejected"
    broker.mark_prices = {"B-BTC_USDT": 100.0}
    order = broker.place_order("B-BTC_USDT", "buy", 0.01, order_type="market_order")
    assert order.status == "filled"
    assert order.avg_price is not None and order.avg_price > 0


def test_synthetic_feed_advances_on_each_call(loose_settings):
    feed = SyntheticFeed(loose_settings, seed=3)
    first = feed.get_frames(["BTC"])
    t1 = first["BTC"]["1h"]["time"].iloc[-1]
    second = feed.get_frames(["BTC"])
    t2 = second["BTC"]["1h"]["time"].iloc[-1]
    assert t2 > t1
    third = feed.get_frames(["BTC"])
    t3 = third["BTC"]["1h"]["time"].iloc[-1]
    assert t3 > t2

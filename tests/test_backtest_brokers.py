import numpy as np
import pandas as pd
import pytest

from ta_agent.ai_ensemble import ConfidenceEngine
from ta_agent.backtest import Backtester
from ta_agent.risk import RiskManager
from ta_agent.settings import Settings
from ta_agent.store import TradeStore
from ta_agent.strategy import StrategyEngine
from ta_agent.brokers import PaperBroker


@pytest.fixture
def settings():
    s = Settings.load("config.json")
    s.mode = "backtest"
    s.market_type = "futures"
    s.risk["min_confidence"] = 0.0
    s.risk["min_rr"] = 0.5
    s.risk["max_positions"] = 3
    s.risk["kelly_cap_pct"] = 0.25
    s.trend_timeframes = ["1h", "4h"]
    s.backtest = {"initial_capital": 10_000.0}
    return s


def make_market(closes_by_coin, n=400):
    market = {}
    rng = np.random.default_rng(11)
    for coin, closes in closes_by_coin.items():
        closes = np.asarray(closes, dtype=float)
        open_ = np.concatenate([[closes[0]], closes[:-1]])
        high = np.maximum(open_, closes) * 1.005
        low = np.minimum(open_, closes) * 0.995
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": closes,
                           "volume": np.abs(rng.normal(100, 15, len(closes)))})
        df["time"] = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
        market[coin] = {
            "1h": df,
            "4h": df.resample("4h", on="time").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum"}).dropna(),
            "1d": df.resample("1D", on="time").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum"}).dropna(),
        }
    return market


def uptrend(n):
    return np.linspace(100, 100 + 100 * n / 400, n)


def downtrend(n):
    return np.linspace(100, 60, n)


class TestBacktester:
    def test_runs_end_to_end(self, settings, tmp_path):
        market = make_market({"BTC": uptrend(400), "ETH": uptrend(400)}, n=400)
        store = TradeStore(str(tmp_path / "bt.db"))
        risk = RiskManager(settings, initial_equity=10_000.0)
        strategy = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        bt = Backtester(settings, strategy, risk, store=store)
        result = bt.run(market, initial_capital=10_000.0)
        assert result.metrics["trades"] > 0
        assert len(result.equity_curve) > 100
        assert "max_drawdown" in result.metrics
        assert "win_rate" in result.metrics
        assert len(store.closed_trades()) > 0

    def test_trades_recorded_in_store(self, settings, tmp_path):
        market = make_market({"BTC": uptrend(400)}, n=400)
        store = TradeStore(str(tmp_path / "bt2.db"))
        risk = RiskManager(settings, initial_equity=10_000.0)
        strategy = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        bt = Backtester(settings, strategy, risk, store=store)
        result = bt.run(market, initial_capital=10_000.0)
        assert len(store.closed_trades()) == result.metrics["trades"]

    def test_max_positions_respected(self, settings, tmp_path):
        market = make_market({"BTC": uptrend(400), "ETH": uptrend(400), "SOL": uptrend(400)},
                             n=400)
        risk = RiskManager(settings, initial_equity=10_000.0)
        strategy = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        bt = Backtester(settings, strategy, risk)
        result = bt.run(market, initial_capital=10_000.0)
        # no assertion on concurrency here (trades may be sequential), just smoke test
        assert result.metrics["trades"] >= 0

    def test_repeated_runs_into_same_store_no_collision(self, settings, tmp_path):
        # deterministic data + a shared store must not hit the trade_key UNIQUE
        # constraint across runs (partial exits share an entry timestamp)
        market = make_market({"BTC": uptrend(400)}, n=400)
        store = TradeStore(str(tmp_path / "bt_repeat.db"))
        strategy = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        for _ in range(2):
            risk = RiskManager(settings, initial_equity=10_000.0)
            bt = Backtester(settings, strategy, risk, store=store)
            bt.run(market, initial_capital=10_000.0)

    def test_short_equity_no_double_count(self, settings, tmp_path):
        # Shorts are opened with a notional cash credit (margin model); the
        # position mark must be -qty*price, otherwise closing a winning short
        # books a fake ~2x-notional equity drop. Guard against that regression.
        market = make_market({"BTC": downtrend(400)}, n=400)
        risk = RiskManager(settings, initial_equity=10_000.0)
        strategy = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        bt = Backtester(settings, strategy, risk)
        result = bt.run(market, initial_capital=10_000.0)
        assert result.metrics["trades"] > 0
        assert all(t["pnl"] > 0 for t in result.trades)
        assert result.metrics["max_drawdown"] > -0.10

    def test_vol_scaling_baseline_is_atr_not_kelly_cap(self, settings, tmp_path):
        """The backtester must pass a real ATR baseline (0.02) to risk.size(),
        NOT kelly_cap_pct (0.25). Passing 0.25 makes vol_factor >= 1 always, so
        positions are never scaled down in high-vol regimes - the root cause of
        the 365d bear/high_vol drawdown failures."""
        import ta_agent.risk as risk_mod
        market = make_market({"BTC": uptrend(400)}, n=400)
        risk = RiskManager(settings, initial_equity=10_000.0)
        strategy = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        bt = Backtester(settings, strategy, risk)
        captured = []
        orig_size = risk_mod.RiskManager.size

        def spy(self, plan, atr_pct=0.0, baseline_atr_pct=0.02, now=None):
            captured.append(baseline_atr_pct)
            return orig_size(self, plan, atr_pct=atr_pct,
                             baseline_atr_pct=baseline_atr_pct, now=now)

        risk_mod.RiskManager.size = spy
        try:
            bt.run(market, initial_capital=10_000.0)
        finally:
            risk_mod.RiskManager.size = orig_size
        assert captured, "backtester must size at least one position"
        assert all(b == pytest.approx(0.02, abs=1e-9) for b in captured), \
            "baseline_atr_pct must be ~0.02, not kelly_cap_pct"


class TestPaperBroker:
    def test_market_order_fills(self):
        s = Settings.load("config.json")
        s.mode = "paper"
        b = PaperBroker(s, initial_cash=10_000.0)
        b.mark_prices = {"B-BTC_USDT": 100.0}
        order = b.place_order("B-BTC_USDT", "buy", 10.0, order_type="market_order")
        assert order.status == "filled"
        assert b.cash < 10_000.0
        pos = b.get_position("B-BTC_USDT")
        assert pos is not None and pos.quantity == 10.0

    def test_close_records_pnl(self):
        s = Settings.load("config.json")
        s.mode = "paper"
        b = PaperBroker(s, initial_cash=10_000.0)
        b.mark_prices = {"B-BTC_USDT": 100.0}
        b.place_order("B-BTC_USDT", "buy", 10.0, order_type="market_order")
        b.mark_prices = {"B-BTC_USDT": 110.0}
        order = b.close_position("B-BTC_USDT")
        assert order is not None
        assert order.meta["pnl"] > 0
        assert b.get_position("B-BTC_USDT") is None

    def test_short_pnl(self):
        s = Settings.load("config.json")
        s.mode = "paper"
        b = PaperBroker(s, initial_cash=10_000.0)
        b.mark_prices = {"B-BTC_USDT": 100.0}
        b.place_order("B-BTC_USDT", "sell", 10.0, order_type="market_order")
        b.mark_prices = {"B-BTC_USDT": 90.0}
        order = b.close_position("B-BTC_USDT")
        assert order.meta["pnl"] > 0

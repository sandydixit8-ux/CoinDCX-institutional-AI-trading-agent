from types import SimpleNamespace

import pytest

from ta_agent.store import TradeStore
from ta_agent.self_learning import LearningJournal
from ta_agent.reporting import compute_stats


def plan(coin="BTC", side="long"):
    return SimpleNamespace(coin=coin, pair=f"B-{coin}_USDT", side=side, entry=100.0,
                           stop_loss=95.0, take_profit=115.0, confidence=0.95,
                           probability=0.62, rr=3.0, trigger="breakout",
                           timeframe="1h", reason="test", ai_signals={},
                           technical_signals={}, as_report=lambda: {
                               "coin": coin, "entry": 100.0, "stop_loss": 95.0,
                               "take_profit": 115.0, "risk_%": 1.0, "probability_%": 62.0,
                               "confidence_%": 95.0, "expected_return": 0.02,
                               "position_size": 0.1, "notional": 10.0, "reason": "test",
                               "technical_signals": {}, "ai_signals": {}, "news_impact": "none",
                               "risk_assessment": "", "expected_holding_time": ""})


class TestStore:
    def test_trade_lifecycle(self, tmp_path):
        st = TradeStore(str(tmp_path / "s.db"))
        st.record_trade_entry(plan(), "k1", 1.0, 100.0, 1000, features=[0.1] * 5)
        st.record_trade_exit("k1", 110.0, "target", 10.0, 0.5)
        closed = st.closed_trades()
        assert len(closed) == 1
        assert closed[0]["outcome"] == "win"
        assert closed[0]["pnl"] == pytest.approx(10.0)

    def test_params(self, tmp_path):
        st = TradeStore(str(tmp_path / "p.db"))
        st.set_param("kelly", {"fraction": 0.3})
        assert st.get_param("kelly") == {"fraction": 0.3}
        assert st.get_param("missing", 42) == 42

    def test_equity_curve(self, tmp_path):
        st = TradeStore(str(tmp_path / "e.db"))
        st.append_equity(1, 1000, 1000, 0.0)
        st.append_equity(2, 1050, 1050, 0.0)
        curve = st.equity_curve()
        assert len(curve) == 2 and curve[-1]["equity"] == 1050.0


class TestLearning:
    def test_edge_estimates(self, tmp_path):
        st = TradeStore(str(tmp_path / "l.db"))
        j = LearningJournal(st, buffer_size=50)
        for _ in range(10):
            j.observe("win", 0.95, 0.6, coin="BTC", side="long", trigger="breakout")
        for _ in range(10):
            j.observe("loss", 0.95, 0.6, coin="BTC", side="long", trigger="breakout")
        est = j.edge_estimates["BTC:long:breakout"]
        assert est["win_rate"] == pytest.approx(0.5)

    def test_calibration_offset(self):
        j = LearningJournal(None, buffer_size=50)
        for _ in range(30):
            j.observe("win", 0.9, 0.6)
        assert j.calibration_offset() == pytest.approx(0.5)


class TestReporting:
    def test_compute_stats(self):
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}]
        stats = compute_stats(trades)
        assert stats["trades"] == 3
        assert stats["win_rate"] == pytest.approx(2 / 3)
        assert stats["profit_factor"] == pytest.approx(30.0 / 5.0)

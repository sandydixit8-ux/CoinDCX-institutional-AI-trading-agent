"""Phase 2: per-trade reports, daily/weekly/monthly reports, failure monitor."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from ta_agent.monitor import FailureMonitor
from ta_agent.reporting import (build_trade_report, lessons_for_trade,
                                periodic_report, trade_reports)
from ta_agent.settings import Settings
from ta_agent.store import TradeStore
from ta_agent.strategy import TradePlan


def plan(coin="BTC", side="long"):
    return SimpleNamespace(coin=coin, pair=f"B-{coin}_USDT", side=side, entry=100.0,
                           stop_loss=95.0, take_profit=115.0, confidence=0.95,
                           probability=0.62, rr=3.0, trigger="breakout",
                           timeframe="1h", reason="test", ai_signals={},
                           technical_signals={}, as_report=lambda: {})


def seed_trades(st, n=4):
    for i in range(n):
        key = f"k{i}"
        st.record_trade_entry(plan(), key, 1.0, 100.0, 1_700_000_000_000 + i,
                              context={"funding": 0.01, "oi": 1e8, "whale": 50.0,
                                       "order_flow": 1.2, "macro_event": "FOMC",
                                       "event_risk": 0.3},
                              regime="up/normal")
        st.record_trade_exit(key, 105.0 + i, "take_profit", 5.0, 0.0)
    st.set_param("_seed", n)


def settings():
    s = Settings()
    s.risk = {"daily": 0.03, "weekly": 0.06, "monthly": 0.10, "per_trade": 0.01,
              "min_confidence": 0.90, "min_rr": 3.0, "max_positions": 5,
              "circuit_breaker": {"consecutive": 3}}
    s.backtest = {"initial_capital": 10_000.0}
    return s


class TestPerTradeReports:
    def test_build_trade_report_has_spec_fields(self, tmp_path):
        st = TradeStore(str(tmp_path / "r.db"))
        st.record_trade_entry(plan(), "r1", 1.0, 100.0, 1_700_000_000_000,
                              context={"funding": 0.01, "oi": 1e8, "whale": 50.0,
                                       "order_flow": 1.2, "macro_event": "FOMC",
                                       "event_risk": 0.3}, regime="up/normal")
        st.record_trade_exit("r1", 110.0, "take_profit", 10.0, 0.0)
        rep = build_trade_report(dict(st.closed_trades()[-1]))
        for field in ("trade_id", "entry_date", "regime", "funding", "oi", "whale",
                      "order_flow", "macro_event", "lessons_learned"):
            assert field in rep, f"missing {field}"
        assert rep["regime"] == "up/normal"
        assert rep["macro_event"] == "FOMC"
        assert rep["outcome"] == "win"

    def test_lessons_are_stored_and_merged(self, tmp_path):
        st = TradeStore(str(tmp_path / "l.db"))
        st.record_trade_entry(plan(), "l1", 1.0, 100.0, 1_700_000_000_000)
        st.record_trade_exit("l1", 90.0, "stop_loss", -6.0, 0.0)
        st.record_trade_lessons("l1", {"notes": ["manual"]})
        rep = build_trade_report(dict(st.closed_trades()[-1]))
        assert rep["lessons_learned"]["notes"] == ["manual"]

    def test_trade_reports_json_serializable(self, tmp_path):
        st = TradeStore(str(tmp_path / "t.db"))
        seed_trades(st)
        reports = trade_reports(st)
        assert len(reports) == 4
        import json
        json.dumps(reports)  # must not raise


class TestPeriodicReports:
    def test_daily_weekly_monthly_grouping(self, tmp_path):
        st = TradeStore(str(tmp_path / "p.db"))
        # two trades on the same day
        st.record_trade_entry(plan(), "a", 1.0, 100.0, 1_700_000_000_000)
        st.record_trade_exit("a", 110.0, "take_profit", 10.0, 0.0)
        st.record_trade_entry(plan(), "b", 1.0, 100.0, 1_700_000_000_000 + 3600_000)
        st.record_trade_exit("b", 110.0, "take_profit", 10.0, 0.0)
        rep = periodic_report(st, settings())
        daily = rep["periods"]["daily"]
        weekly = rep["periods"]["weekly"]
        monthly = rep["periods"]["monthly"]
        assert len(daily) == 1 and daily[0]["trades"] == 2
        assert len(weekly) == 1 and weekly[0]["trades"] == 2
        assert len(monthly) == 1 and monthly[0]["trades"] == 2
        assert daily[0]["compliant"] is True
        assert daily[0]["win_rate"] == pytest.approx(1.0)

    def test_daily_loss_limit_flags_noncompliant(self, tmp_path):
        st = TradeStore(str(tmp_path / "pl.db"))
        # 4% loss in one day against a 3% daily limit
        st.record_trade_entry(plan(), "x", 1.0, 10_000.0, 1_700_000_000_000)
        st.record_trade_exit("x", 96.0, "stop_loss", -400.0, 0.0)
        rep = periodic_report(st, settings())
        daily = rep["periods"]["daily"][0]
        assert daily["compliant"] is False

    def test_writes_markdown_and_json(self, tmp_path):
        st = TradeStore(str(tmp_path / "w.db"))
        seed_trades(st)
        out = tmp_path / "reports"
        periodic_report(st, settings(), out_dir=out)
        assert (out / "periodic_report.json").exists()
        assert (out / "periodic_report.md").exists()


class TestFailureMonitor:
    def test_raises_daily_loss_alert(self, tmp_path):
        st = TradeStore(str(tmp_path / "m.db"))
        mon = FailureMonitor(settings(), store=st)
        state = SimpleNamespace(equity=9_500.0, peak_equity=10_000.0,
                                realized_today=-500.0, realized_week=-500.0,
                                realized_month=-500.0, consecutive_losses=2,
                                open_positions=0)
        alerts = mon.check_risk_limits(state, __import__("datetime").datetime.now())
        assert any(a["rule"] == "daily_loss_limit" for a in alerts)
        assert st.recent_alerts()  # persisted to store

    def test_max_positions_alert(self, tmp_path):
        mon = FailureMonitor(settings())
        alerts = mon.check_open_positions(7)
        assert alerts and alerts[0]["rule"] == "max_positions"

    def test_drawdown_alert(self, tmp_path):
        mon = FailureMonitor(settings())
        state = SimpleNamespace(equity=8_000.0, peak_equity=10_000.0)
        alerts = mon.check_drawdown(state)
        assert alerts and alerts[0]["rule"] == "drawdown_limit"

    def test_plan_gates(self, tmp_path):
        mon = FailureMonitor(settings())
        bad = SimpleNamespace(coin="BTC", side="long", risk_pct=0.03, rr=2.0,
                              stop_loss=None, confidence=0.5)
        alerts = mon.check_plans([bad])
        rules = {a["rule"] for a in alerts}
        assert {"risk_per_trade", "rr_below", "stop_loss_missing", "confidence_below"} <= rules

    def test_summary_counts(self, tmp_path):
        st = TradeStore(str(tmp_path / "ms.db"))
        mon = FailureMonitor(settings(), store=st)
        mon.alert("critical", "drawdown_limit", "dd")
        mon.alert("warning", "loss_streak", "streak")
        s = mon.summary()
        assert s["critical"] == 1 and s["warning"] == 1

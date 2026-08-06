import datetime as dt

import pytest

from ta_agent.risk import RiskManager
from ta_agent.settings import Settings


def make_plan(entry=100.0, sl=95.0, tp=115.0, rr=3.0, prob=0.62, confidence=0.95):
    from types import SimpleNamespace
    return SimpleNamespace(entry=entry, stop_loss=sl, take_profit=tp, rr=rr,
                           probability=prob, confidence=confidence, notional=0.0)


@pytest.fixture
def settings():
    s = Settings.load("config.json")
    s.mode = "backtest"
    return s


class TestRiskManager:
    def test_approve_when_healthy(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        ok, reason = rm.approve(make_plan(), open_positions=0)
        assert ok and reason == ""

    def test_daily_loss_limit_blocks(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        rm.on_trade_close(-0.04 * rm.state.equity)  # -4% in one trade
        ok, reason = rm.approve(make_plan())
        assert not ok
        assert "daily" in reason

    def test_weekly_limit_blocks(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        base = dt.datetime(2024, 1, 8, tzinfo=dt.timezone.utc)  # Monday, week 02/2024
        rm.state.day_key = rm._bucket_key("day", base)
        rm.state.week_key = rm._bucket_key("week", base)
        rm.state.month_key = rm._bucket_key("month", base)
        for i in range(4):  # one 1.5% loss per day, same week (daily never breached)
            rm.on_trade_close(-0.015 * rm.state.equity)
            rm.roll_buckets(base + dt.timedelta(days=i + 1))
        ok, reason = rm.approve(make_plan(), now=base + dt.timedelta(days=4))
        assert not ok
        assert "weekly" in reason

    def test_consecutive_losses_block(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        limit = settings.risk.get("circuit_breaker", {}).get("consecutive", 3)
        for _ in range(limit):
            rm.on_trade_close(-10.0)
        ok, reason = rm.approve(make_plan())
        assert not ok
        assert "consecutive" in reason

    def test_roll_buckets_resets_day(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        rm.on_trade_close(-200.0)
        tomorrow = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
        rm.roll_buckets(tomorrow)
        assert rm.state.realized_today == 0.0

    def test_size_risk_based(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        plan = make_plan(entry=100.0, sl=95.0)  # risk 5 per unit
        qty = rm.size(plan, atr_pct=0.02, baseline_atr_pct=0.02)
        risk_amount = settings.per_trade_risk * 10_000.0  # 100
        assert qty * 5.0 <= risk_amount * 1.001
        assert qty > 0

    def test_size_kelly_caps_notional(self, settings):
        settings.risk["kelly_cap_pct"] = 0.25
        rm = RiskManager(settings, initial_equity=10_000.0)
        plan = make_plan(entry=100.0, sl=50.0, prob=0.95, rr=5.0)  # risk 50/unit
        qty = rm.size(plan, atr_pct=0.02, baseline_atr_pct=0.02)
        assert qty * plan.entry <= 0.25 * 10_000.0 * 1.001

    def test_size_vol_scaling(self, settings):
        rm = RiskManager(settings, initial_equity=10_000.0)
        plan = make_plan()
        q_low_vol = rm.size(plan, atr_pct=0.01, baseline_atr_pct=0.02)
        q_high_vol = rm.size(plan, atr_pct=0.05, baseline_atr_pct=0.02)
        assert q_high_vol < q_low_vol

    def test_equity_never_negative(self, settings):
        rm = RiskManager(settings, initial_equity=1_000.0)
        rm.on_trade_close(-10_000.0)
        assert rm.state.equity == 0.0

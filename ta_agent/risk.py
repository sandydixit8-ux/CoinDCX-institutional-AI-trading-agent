"""Risk management: limits, position sizing (Kelly + volatility), circuit breaker.

Hard limits (from spec / config)
--------------------------------
* Max risk per trade  = 1%
* Max daily loss     = 3%
* Max weekly loss    = 6%
* Max monthly loss   = 10%
* Automatic position sizing (Kelly fraction, volatility adjusted, ATR stop)
* Circuit breaker halts new entries after loss-limit breaches / loss streaks
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .settings import Settings

log = logging.getLogger("ta_agent.risk")


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    realized_today: float
    realized_week: float
    realized_month: float
    day_key: str
    week_key: str
    month_key: str
    consecutive_losses: int
    open_positions: int = 0
    paused_reason: Optional[str] = None
    paused_until_ms: Optional[int] = None


class CircuitBreaker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.consecutive_loss_limit = int(settings.risk.get("circuit_breaker", {}).get("consecutive", 3) or 3)

    def check(self, state: RiskState, now: datetime) -> Optional[str]:
        """Return a reason string if trading must halt, else None."""
        if state.paused_reason and state.paused_until_ms:
            if now.timestamp() * 1000 < state.paused_until_ms:
                return state.paused_reason
            state.paused_reason = None
            state.paused_until_ms = None
        if state.realized_today <= -self.s.daily_loss_limit * state.equity:
            return f"daily loss limit ({self.s.daily_loss_limit:.0%}) breached"
        if state.realized_week <= -self.s.weekly_loss_limit * state.equity:
            return f"weekly loss limit ({self.s.weekly_loss_limit:.0%}) breached"
        if state.realized_month <= -self.s.monthly_loss_limit * state.equity:
            return f"monthly loss limit ({self.s.monthly_loss_limit:.0%}) breached"
        if state.consecutive_losses >= self.consecutive_loss_limit:
            return f"{state.consecutive_losses} consecutive losses"
        if state.equity <= state.peak_equity * 0.90:
            return "10% peak drawdown breach"
        return None

    def cooldown_ms(self) -> int:
        return int(self.s.risk.get("circuit_breaker", {}).get("cooldown_hours", 6) * 3600 * 1000)


class RiskManager:
    def __init__(self, settings: Settings, initial_equity: float = 10_000.0):
        self.s = settings
        self.breaker = CircuitBreaker(settings)
        self.state = RiskState(
            equity=initial_equity,
            peak_equity=initial_equity,
            realized_today=0.0,
            realized_week=0.0,
            realized_month=0.0,
            day_key=self._bucket_key("day"),
            week_key=self._bucket_key("week"),
            month_key=self._bucket_key("month"),
            consecutive_losses=0,
        )

    # ------------------------------------------------------------------
    def _bucket_key(self, kind: str, now: Optional[datetime] = None) -> str:
        now = now or datetime.now(timezone.utc)
        if kind == "day":
            return now.strftime("%Y-%m-%d")
        if kind == "week":
            return now.strftime("%Y-%W")
        return now.strftime("%Y-%m")

    def roll_buckets(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        if self.state.day_key != self._bucket_key("day", now):
            self.state.realized_today = 0.0
            self.state.day_key = self._bucket_key("day", now)
            if self.state.consecutive_losses > 0 and self.state.paused_reason and "consecutive" in self.state.paused_reason:
                self.state.consecutive_losses = 0
                self.state.paused_reason = None
                self.state.paused_until_ms = None
        if self.state.week_key != self._bucket_key("week", now):
            self.state.realized_week = 0.0
            self.state.week_key = self._bucket_key("week", now)
        if self.state.month_key != self._bucket_key("month", now):
            self.state.realized_month = 0.0
            self.state.month_key = self._bucket_key("month", now)
            self.state.consecutive_losses = 0

    def on_trade_close(self, pnl: float) -> None:
        self.state.realized_today += pnl
        self.state.realized_week += pnl
        self.state.realized_month += pnl
        self.state.equity = max(0.0, self.state.equity + pnl)
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def can_open(self, now: Optional[datetime] = None) -> Optional[str]:
        """Return None if trading is allowed, else the reason it is blocked."""
        self.roll_buckets(now)
        now = now or datetime.now(timezone.utc)
        reason = self.breaker.check(self.state, now)
        if reason:
            self.state.paused_reason = reason
            self.state.paused_until_ms = int(now.timestamp() * 1000) + self.breaker.cooldown_ms()
            return reason
        return None

    def approve(self, plan, open_notional: float = 0.0, open_positions: int = 0,
                now: Optional[datetime] = None) -> tuple:
        """Check a TradePlan against limits. Returns (approved: bool, reason: str)."""
        reason = self.can_open(now)
        if reason:
            return False, f"circuit breaker: {reason}"
        if open_positions >= self.s.max_positions:
            return False, "max concurrent positions reached"
        if plan.notional > 0 and (plan.notional + open_notional) > self.s.max_positions * 0.5 * self.state.equity:
            return False, "aggregate exposure cap reached"
        return True, ""

    # ------------------------------------------------------------------
    def size(self, plan, atr_pct: float = 0.0, baseline_atr_pct: float = 0.02,
             now: Optional[datetime] = None) -> float:
        """Compute position size (base qty) using risk, Kelly and vol adjustment."""
        self.roll_buckets(now)
        equity = self.state.equity
        risk_amt = equity * self.s.per_trade_risk
        risk_dist = abs(plan.entry - plan.stop_loss)
        if risk_dist <= 0:
            return 0.0
        qty_risk = risk_amt / risk_dist

        # Kelly sizing: f* = p - (1-p)/b ; apply fraction, cap.
        b = max(plan.rr, 1e-6)
        p = plan.probability
        kelly = max(0.0, p - (1 - p) / b)
        kelly_frac = float(self.s.risk.get("kelly_fraction", 0.25))
        cap_pct = float(self.s.risk.get("kelly_cap_pct", 0.25))
        kelly_notional_cap = equity * cap_pct
        kelly_use = min(kelly * kelly_frac, cap_pct)
        kelly_qty = (kelly_use * equity) / plan.entry if plan.entry else 0.0

        # Volatility adjustment: scale down when vol is above baseline.
        vol_factor = 1.0
        if atr_pct > 0 and baseline_atr_pct > 0:
            vol_factor = max(0.25, min(1.0, baseline_atr_pct / max(atr_pct, 1e-6)))

        qty = min(qty_risk, kelly_qty) * vol_factor

        # Coin weight cap
        coin_weight = float(self.s.risk.get("max_coin_weight", 0.25))
        max_notional = equity * coin_weight
        if qty * plan.entry > max_notional:
            qty = max_notional / plan.entry

        plan.position_size = qty
        plan.notional = qty * plan.entry
        return qty

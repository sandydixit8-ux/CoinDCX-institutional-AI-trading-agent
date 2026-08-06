"""Failure-condition monitor.

Periodically re-checks the spec's failure conditions against live/paper state
and raises structured alerts:

* per-trade risk > 1% of equity
* daily / weekly / monthly loss-limit breaches
* equity drawdown beyond the monthly limit
* open position count beyond ``max_positions``
* plan confidence below ``min_confidence`` or R:R below ``min_rr``
* missing stop-loss on a plan
* entries inside a news blackout window
* broker / feed errors (API failures, stale data, no price)

Alerts are persisted to the trade store (``monitor_alerts`` table) when one is
available and written to ``data/monitor_alerts.json`` as a rolling window.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("ta_agent.monitor")


class FailureMonitor:
    SEVERITIES = ("info", "warning", "critical")

    def __init__(self, settings, store=None,
                 alert_file: Optional[Path] = None,
                 max_alerts: int = 500):
        self.s = settings
        self.store = store
        self.alert_file = alert_file
        self.max_alerts = max_alerts
        self._memory: List[dict] = []

    # ------------------------------------------------------------------
    def alert(self, severity: str, rule: str, detail: str,
              meta: Optional[dict] = None) -> dict:
        if severity not in self.SEVERITIES:
            severity = "warning"
        a = {"ts": int(datetime.now(timezone.utc).timestamp() * 1000),
             "severity": severity, "rule": rule, "detail": detail,
             "meta": meta or {}}
        self._memory.append(a)
        if len(self._memory) > self.max_alerts:
            self._memory = self._memory[-self.max_alerts:]
        if self.store is not None:
            self.store.append_alert(severity, rule, detail, meta)
        self._persist(a)
        log.warning("MONITOR %s | %s | %s", severity.upper(), rule, detail)
        return a

    def _persist(self, a: dict) -> None:
        if self.alert_file is None:
            return
        try:
            existing = []
            if self.alert_file.exists():
                existing = json.loads(self.alert_file.read_text(encoding="utf-8"))
            existing.append(a)
            existing = existing[-self.max_alerts:]
            self.alert_file.parent.mkdir(parents=True, exist_ok=True)
            self.alert_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            log.debug("alert persistence failed: %s", exc)

    # ------------------------------------------------------------------
    def check(self, state, open_positions: dict, now: Optional[datetime] = None,
              plans: Optional[list] = None, error_rate: float = 0.0,
              last_event_risk=None) -> List[dict]:
        """Run all checks; returns the alerts raised this pass."""
        now = now or datetime.now(timezone.utc)
        raised: List[dict] = []
        raised.extend(self.check_risk_limits(state, now))
        raised.extend(self.check_drawdown(state))
        raised.extend(self.check_open_positions(len(open_positions)))
        raised.extend(self.check_plans(plans or [], last_event_risk=last_event_risk))
        raised.extend(self.check_health(error_rate))
        return raised

    def check_risk_limits(self, state, now: datetime) -> List[dict]:
        out: List[dict] = []
        if state is None:
            return out
        daily = float(self.s.risk.get("daily", 0.03))
        weekly = float(self.s.risk.get("weekly", 0.06))
        monthly = float(self.s.risk.get("monthly", 0.10))
        base = state.peak_equity if state.peak_equity else (state.equity or 1.0)
        day_ratio = -state.realized_today / base
        week_ratio = -state.realized_week / base
        month_ratio = -state.realized_month / base
        if day_ratio > daily:
            out.append(self.alert("critical", "daily_loss_limit",
                                  f"daily realized loss {day_ratio:.2%} exceeds {daily:.2%}",
                                  {"ratio": day_ratio, "limit": daily}))
        if week_ratio > weekly:
            out.append(self.alert("critical", "weekly_loss_limit",
                                  f"weekly realized loss {week_ratio:.2%} exceeds {weekly:.2%}",
                                  {"ratio": week_ratio, "limit": weekly}))
        if month_ratio > monthly:
            out.append(self.alert("critical", "monthly_loss_limit",
                                  f"monthly realized loss {month_ratio:.2%} exceeds {monthly:.2%}",
                                  {"ratio": month_ratio, "limit": monthly}))
        if state.consecutive_losses >= int(self.s.risk.get("circuit_breaker", {}).get("consecutive", 3) or 3):
            out.append(self.alert("warning", "loss_streak",
                                  f"{state.consecutive_losses} consecutive losses",
                                  {"streak": state.consecutive_losses}))
        return out

    def check_drawdown(self, state) -> List[dict]:
        if state is None:
            return []
        if state.equity <= 0:
            # Nothing at risk (e.g. unfunded live wallet) - drawdown is undefined.
            return []
        limit = float(self.s.risk.get("monthly", 0.10))
        peak = state.peak_equity if state.peak_equity else state.equity
        dd = (state.equity - peak) / peak if peak > 0 else 0.0
        if dd < -limit:
            return [self.alert("critical", "drawdown_limit",
                               f"drawdown {dd:.2%} below monthly limit {limit:.2%}",
                               {"drawdown": dd, "limit": limit})]
        return []

    def check_open_positions(self, count: int) -> List[dict]:
        max_pos = int(self.s.max_positions)
        if count > max_pos:
            return [self.alert("critical", "max_positions",
                               f"open positions {count} exceed {max_pos}",
                               {"count": count, "limit": max_pos})]
        return []

    def check_plans(self, plans: list, last_event_risk=None) -> List[dict]:
        out: List[dict] = []
        per_trade = float(self.s.risk.get("per_trade", 0.01))
        min_conf = float(self.s.risk.get("min_confidence", 0.90))
        min_rr = float(self.s.risk.get("min_rr", 3.0))
        for plan in plans:
            if plan is None:
                continue
            if getattr(plan, "risk_pct", 0.0) > per_trade + 1e-9:
                out.append(self.alert("critical", "risk_per_trade",
                                      f"{plan.coin} {plan.side} risk {plan.risk_pct:.2%} > {per_trade:.2%}"))
            if getattr(plan, "rr", 0.0) < min_rr - 1e-9:
                out.append(self.alert("warning", "rr_below",
                                      f"{plan.coin} {plan.side} R:R {plan.rr:.2f} < {min_rr:.1f}"))
            if not getattr(plan, "stop_loss", None) or float(getattr(plan, "stop_loss", 0) or 0) <= 0:
                out.append(self.alert("critical", "stop_loss_missing",
                                      f"{plan.coin} {plan.side} plan has no stop"))
            if getattr(plan, "confidence", 0.0) < min_conf - 1e-9:
                out.append(self.alert("warning", "confidence_below",
                                      f"{plan.coin} confidence {plan.confidence:.2f} < {min_conf:.2f}"))
        if last_event_risk is not None:
            risk = getattr(last_event_risk, "risk", 0.0)
            if risk > 0.5:
                out.append(self.alert("warning", "news_blackout",
                                      f"high-impact event active: {getattr(last_event_risk, 'nearest_event', '?')} "
                                      f"(risk {risk:.2f})",
                                      {"event": getattr(last_event_risk, "nearest_event", "?")}))
        return out

    def check_health(self, error_rate: float = 0.0) -> List[dict]:
        out: List[dict] = []
        if error_rate >= 0.5:
            out.append(self.alert("critical", "feed_errors",
                                  f"broker/feed error rate {error_rate:.0%} in last window"))
        return out

    # ------------------------------------------------------------------
    def recent_alerts(self, limit: int = 50) -> List[dict]:
        if self.store is not None:
            return [dict(r) for r in self.store.recent_alerts(limit)]
        return self._memory[-limit:]

    def summary(self) -> dict:
        alerts = self.recent_alerts(200)
        counts: Dict[str, int] = {}
        rules: Dict[str, int] = {}
        for a in alerts:
            counts[a["severity"]] = counts.get(a["severity"], 0) + 1
            rules[a["rule"]] = rules.get(a["rule"], 0) + 1
        return {"critical": counts.get("critical", 0),
                "warning": counts.get("warning", 0),
                "info": counts.get("info", 0),
                "by_rule": rules}

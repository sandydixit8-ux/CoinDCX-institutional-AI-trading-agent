"""Validation harness: extended performance metrics and rule-compliance checks.

Used by ``run_validation.py`` to grade the system against the spec's failure
conditions across regime scenarios and test durations.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .news_engine import EconomicCalendar

log = logging.getLogger("ta_agent.validation")

# Spec failure conditions (config keys -> labels)
CRITICAL_RULES = {
    "risk_per_trade": "risk per trade exceeds 1%",
    "confidence_below": "trade executed with confidence below 90%",
    "rr_below": "risk-reward below 1:3",
    "stop_loss_missing": "stop-loss missing",
    "daily_loss": "daily loss exceeds 3%",
    "drawdown": "monthly drawdown exceeds 10%",
    "news_blackout": "trade executed during restricted news events",
}


def extended_metrics(metrics: dict, trades: List[dict],
                     equity: pd.DataFrame, days: int) -> dict:
    """Spec performance metrics beyond the base backtest metrics."""
    out = dict(metrics)
    out["test_days"] = days
    if equity is not None and len(equity):
        eq = equity["equity"].to_numpy(dtype=float)
        rets = pd.Series(eq).pct_change().dropna()
        bar_secs = float(np.median(np.diff(equity["ts"].to_numpy()) / 1000.0)) if len(equity) > 1 else 3600.0
        ann = 365.25 * 24 * 3600 / max(bar_secs, 1.0)
        std = rets.std()
        out["annualized_return"] = float((eq[-1] / eq[0]) ** (365.25 / max(days, 1)) - 1) if eq[0] > 0 else 0.0
        out["capital_volatility"] = float(std * np.sqrt(ann)) if std > 0 else 0.0
        downside = rets[rets < 0]
        if len(downside) and downside.std() > 0:
            out["sortino"] = float(rets.mean() / downside.std() * np.sqrt(ann))
        else:
            out["sortino"] = 0.0
        mdd = float(equity["drawdown"].min())
        out["max_drawdown"] = mdd
        if mdd < 0:
            out["calmar"] = float(out["annualized_return"] / abs(mdd))
            out["recovery_factor"] = float(out["total_return"] / abs(mdd)) if abs(mdd) > 1e-12 else 0.0
        else:
            out["calmar"] = 0.0
            out["recovery_factor"] = 0.0
    else:
        out.setdefault("max_drawdown", 0.0)
        out.setdefault("annualized_return", 0.0)
        out.setdefault("sortino", 0.0)
        out.setdefault("calmar", 0.0)
        out.setdefault("recovery_factor", 0.0)
        out.setdefault("capital_volatility", 0.0)

    if trades:
        pnls = np.asarray([float(t["pnl"]) for t in trades], dtype=float)
        out["expectancy"] = float(pnls.mean())
        holds = [(float(t["exit_time"]) - float(t["entry_time"])) / 3.6e6 for t in trades]
        out["avg_holding_hours"] = float(np.mean(holds))
        rrs = [float(t.get("rr") or 0.0) for t in trades]
        out["avg_rr"] = float(np.mean(rrs))
        out["max_risk_pct"] = float(max(float(t.get("risk_pct") or 0.0) for t in trades))
        out["min_confidence"] = float(min(float(t.get("confidence") or 0.0) for t in trades))
    else:
        out["expectancy"] = 0.0
        out["avg_holding_hours"] = 0.0
        out["avg_rr"] = 0.0
        out["max_risk_pct"] = 0.0
        out["min_confidence"] = 0.0
    return out


def compliance(metrics: dict, trades: List[dict], settings,
               days: int, calendar: Optional[EconomicCalendar] = None,
               skip_rules: Optional[set] = None) -> dict:
    """Check every spec failure condition. Returns violations + PASS/FAIL.

    ``skip_rules`` excludes documented profile relaxations (e.g. the
    confidence gate in edge-validation mode) from being graded as violations.
    """
    skip_rules = skip_rules or set()
    cal = calendar or EconomicCalendar()
    risk_cfg = settings.risk
    per_trade = float(risk_cfg.get("per_trade", 0.01))
    daily = float(risk_cfg.get("daily", 0.03))
    drawdown_lim = float(risk_cfg.get("monthly", 0.10))
    min_conf = float(risk_cfg.get("min_confidence", 0.90))
    min_rr = float(risk_cfg.get("min_rr", 3.0))
    blackout = float(settings.news.get("blackout_hours", 2.0))
    initial = float(settings.backtest.get("initial_capital", 10_000.0))

    violations: List[Dict[str, str]] = []

    if not trades:
        return {"violations": violations, "passed": True,
                "checks": [{"rule": rule, "ok": True} for rule in CRITICAL_RULES],
                "trades": 0}

    if metrics.get("max_risk_pct", 0.0) > per_trade + 1e-9:
        violations.append({"rule": "risk_per_trade", "detail": f"max risk {metrics['max_risk_pct']:.4%} > {per_trade:.2%}"})
    if metrics.get("min_confidence", 1.0) < min_conf - 1e-9 and "confidence_below" not in skip_rules:
        violations.append({"rule": "confidence_below", "detail": f"min confidence {metrics['min_confidence']:.4f} < {min_conf:.2f}"})
    for t in trades:
        if float(t.get("rr") or 0.0) < min_rr - 1e-9:
            violations.append({"rule": "rr_below", "detail": f"{t['coin']} rr {t['rr']:.2f} < {min_rr:.1f}"})
            break
    for t in trades:
        if not t.get("stop_loss") or float(t.get("stop_loss") or 0.0) <= 0:
            violations.append({"rule": "stop_loss_missing", "detail": f"{t['coin']} no stop"})
            break

    # daily loss limit (in % of initial capital per UTC exit day)
    if trades:
        s = pd.Series([float(t["pnl"]) for t in trades],
                      index=pd.to_datetime([t["exit_time"] for t in trades], unit="ms", utc=True))
        per_day = s.groupby(s.index.date).sum() / initial
        worst_day = per_day.min()
        if worst_day < -daily:
            violations.append({"rule": "daily_loss", "detail": f"worst day {worst_day:.2%} < -{daily:.2%}"})

    mdd = float(metrics.get("max_drawdown", 0.0))
    if mdd < -drawdown_lim:
        violations.append({"rule": "drawdown", "detail": f"drawdown {mdd:.2%} < -{drawdown_lim:.2%}"})

    # trades executed inside a news blackout window
    for t in trades:
        ts_dt = pd.Timestamp(t["entry_time"], unit="ms", tz="UTC").to_pydatetime()
        er = cal.event_risk(ts_dt, blackout_hours=blackout)
        if er.risk > 0.5:
            violations.append({"rule": "news_blackout", "detail": f"{t['coin']} entry near {er.name}"})
            break

    checks = [{"rule": rule, "ok": rule not in {v["rule"] for v in violations}}
              for rule in CRITICAL_RULES]
    failed = len(violations) > 3
    return {"violations": violations, "passed": not violations,
            "hard_failed": failed, "checks": checks, "trades": len(trades)}

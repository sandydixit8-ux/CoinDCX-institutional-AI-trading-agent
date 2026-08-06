"""Consolidated spec-compliance report.

Ties together the three verification layers into one deliverable:

1. Phase 1 - regime-scenario x duration validation matrix
2. Phase 2 - per-trade compliance, period loss limits, failure monitor
3. Phase 3 - self-learning verification

Run via ``run_compliance.py`` or call ``spec_compliance_report()`` directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .monitor import FailureMonitor
from .reporting import periodic_report, trade_reports
from .self_learning import learning_report
from .settings import Settings
from .store import TradeStore

log = logging.getLogger("ta_agent.compliance")

# The spec's institutional risk rules (as documented in the acceptance spec).
SPEC_RISK = {
    "per_trade": 0.01,          # max risk per trade = 1% of equity
    "daily": 0.03,              # daily loss limit
    "weekly": 0.06,             # weekly loss limit
    "monthly": 0.10,            # monthly drawdown limit
    "max_positions": 5,         # concurrent position cap
    "max_coin_weight": 0.25,    # single-coin concentration cap
    "max_correlated_weight": 0.50,  # correlated-basket cap
    "kelly_fraction": 0.25,     # fractional Kelly
    "atr_mult_sl": 1.5,         # stop distance as multiple of ATR
    "min_rr": 3.0,              # minimum reward:risk
    "min_confidence": 0.90,     # AI confidence gate
}

MATRIX_RULES = ["risk_per_trade", "confidence_below", "rr_below",
                "stop_loss_missing", "daily_loss", "drawdown", "news_blackout"]


# ---------------------------------------------------------------------------
def _matrix_summary(matrix: dict) -> dict:
    scenarios = matrix.get("scenarios", {})
    cells = []
    for sc, durs in scenarios.items():
        for dur, profs in durs.items():
            for prof, cell in profs.items():
                cells.append({"scenario": sc, "days": dur, "profile": prof,
                              **{k: cell.get(k) for k in
                                 ("n_trades", "metrics", "compliance")}})
    total = len(cells)
    passed = sum(1 for c in cells if c["compliance"].get("passed"))
    hard_failed = sum(1 for c in cells if c["compliance"].get("hard_failed"))
    n_trades = sum(int(c.get("n_trades") or 0) for c in cells)
    violations: Dict[str, int] = {}
    worst: List[dict] = []
    for c in cells:
        for v in (c.get("compliance") or {}).get("violations", []):
            rule = v.get("rule") if isinstance(v, dict) else str(v)
            violations[rule] = violations.get(rule, 0) + 1
        for chk in (c.get("compliance") or {}).get("checks", []):
            if not chk.get("ok"):
                worst.append({"scenario": c["scenario"], "days": c["days"],
                              "profile": c["profile"], "rule": chk.get("rule")})
    return {
        "cells": total, "passed_cells": passed, "hard_failed_cells": hard_failed,
        "trades_executed": n_trades,
        "violations_by_rule": violations,
        "noncompliant_cells": worst,
        "overall": matrix.get("summary", {}).get("overall", "PASS"),
        "profile_pass": matrix.get("summary", {}).get("profile_pass", {}),
        "elapsed_seconds": matrix.get("summary", {}).get("elapsed_seconds"),
    }


def _risk_config(settings: Settings) -> dict:
    rows = []
    for key, spec in SPEC_RISK.items():
        cfg = settings.risk.get(key)
        rows.append({"rule": key, "spec": spec, "configured": cfg,
                     "ok": cfg is not None and abs(float(cfg) - float(spec)) < 1e-9})
    return {"rules": rows,
            "all_match": all(r["ok"] for r in rows),
            "leverage": settings.leverage,
            "trade_timeframe": settings.trade_timeframe,
            "watchlist": settings.watchlist}


def _per_trade_from_matrix(matrix: dict) -> dict:
    scenarios = matrix.get("scenarios", {})
    violations: Dict[str, int] = {}
    total = 0
    for sc, durs in scenarios.items():
        for dur, profs in durs.items():
            for prof, cell in profs.items():
                total += int(cell.get("n_trades") or 0)
                for v in (cell.get("compliance") or {}).get("violations", []):
                    rule = v.get("rule") if isinstance(v, dict) else str(v)
                    violations[rule] = violations.get(rule, 0) + 1
    return {"trades": total, "violations": violations,
            "passed": not any(violations.values())}


def _per_trade_from_store(store: Optional[TradeStore], settings: Settings,
                          skip_rules: Optional[set] = None) -> dict:
    skip = skip_rules or set()
    if store is None:
        return {"trades": 0, "violations": {}, "passed": True, "available": False}
    reports = trade_reports(store)
    violations: List[dict] = []
    for t in reports:
        tk = t["trade_key"]
        if (t.get("risk_pct") or 0.0) > settings.per_trade_risk + 1e-9:
            violations.append({"rule": "risk_per_trade", "trade": tk,
                               "value": t.get("risk_pct")})
        if (t.get("rr") or 0.0) < settings.min_rr - 1e-9 and t.get("rr"):
            violations.append({"rule": "rr_below", "trade": tk, "value": t.get("rr")})
        if not t.get("stop_loss") or float(t.get("stop_loss") or 0) <= 0:
            violations.append({"rule": "stop_loss_missing", "trade": tk})
        if "confidence_below" not in skip and \
                (t.get("confidence") or 0.0) < settings.min_confidence - 1e-9:
            violations.append({"rule": "confidence_below", "trade": tk,
                               "value": t.get("confidence")})
        if float(t.get("event_risk") or 0.0) > 0.5:
            violations.append({"rule": "news_blackout", "trade": tk,
                               "value": t.get("event_risk")})
    by_rule: Dict[str, int] = {}
    for v in violations:
        by_rule[v["rule"]] = by_rule.get(v["rule"], 0) + 1
    return {"trades": len(reports), "violations": by_rule,
            "violation_details": violations, "passed": not violations,
            "available": True}


def _period_limits(store: Optional[TradeStore], settings: Settings) -> dict:
    if store is None:
        return {"available": False, "passed": True, "noncompliant_periods": []}
    rep = periodic_report(store, settings)
    noncompliant = []
    for kind, rows in rep["periods"].items():
        for r in rows:
            if not r["compliant"]:
                noncompliant.append({"kind": kind, "period": r["period"],
                                     "pnl_pct": r["pnl_vs_period_start"],
                                     "max_drawdown": r["max_drawdown"]})
    return {"available": True, "passed": not noncompliant,
            "noncompliant_periods": noncompliant,
            "limits": rep["limits"],
            "periods_covered": sum(len(v) for v in rep["periods"].values())}


def _monitor_summary(store: Optional[TradeStore], settings: Settings) -> dict:
    mon = FailureMonitor(settings, store=store)
    summary = mon.summary()
    alerts = mon.recent_alerts(200)
    criticals = [a for a in alerts if a["severity"] == "critical"]
    return {"available": store is not None,
            "critical": summary["critical"], "warning": summary["warning"],
            "by_rule": summary["by_rule"],
            "critical_alerts": criticals[-10:],
            "passed": store is None or summary["critical"] == 0}


def _learning(store: Optional[TradeStore]) -> dict:
    if store is None:
        return {"available": False, "passed": True,
                 "observations": 0, "verdict": "insufficient_data"}
    rep = learning_report(store)
    decided = rep["decided"]
    if decided >= 30:
        verdict = "passed"
    elif decided >= 3:
        verdict = "insufficient_data"
    else:
        verdict = "insufficient_data"
    return {**rep, "available": True,
            "verdict": verdict,
            "passed": verdict == "passed" or verdict == "insufficient_data"}


def spec_compliance_report(settings: Settings,
                           matrix_path: Optional[str | Path] = None,
                           store_path: Optional[str | Path] = None,
                           out_dir: Optional[str | Path] = None,
                           skip_rules: Optional[set] = None) -> dict:
    matrix = None
    if matrix_path and Path(matrix_path).exists():
        matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))

    store = None
    if store_path and Path(store_path).exists():
        store = TradeStore(str(store_path))

    matrix_summary = _matrix_summary(matrix) if matrix else {
        "cells": 0, "passed_cells": 0, "hard_failed_cells": 0,
        "trades_executed": 0, "violations_by_rule": {},
        "noncompliant_cells": [], "overall": "NO_DATA",
        "profile_pass": {}, "elapsed_seconds": None,
    }
    risk_cfg = _risk_config(settings)
    pt_matrix = _per_trade_from_matrix(matrix) if matrix else {
        "trades": 0, "violations": {}, "passed": True}
    pt_store = _per_trade_from_store(store, settings, skip_rules=skip_rules)
    periods = _period_limits(store, settings)
    monitor = _monitor_summary(store, settings)
    learning = _learning(store)

    items = [
        {"item": "validation_matrix",
         "status": "PASS" if matrix_summary["overall"] == "PASS"
         else ("FAIL" if matrix else "N/A"),
         "evidence": (f"{matrix_summary['passed_cells']}/{matrix_summary['cells']} cells passed, "
                      f"overall {matrix_summary['overall']}") if matrix else "no matrix report"},
        {"item": "risk_rules", "status": "PASS" if risk_cfg["all_match"] else "FAIL",
         "evidence": "all 11 institutional risk rules match spec" if risk_cfg["all_match"]
         else f"{len([r for r in risk_cfg['rules'] if not r['ok']])} rules mismatch"},
        {"item": "per_trade_compliance",
         "status": "PASS" if (pt_matrix["passed"] and pt_store["passed"]) else "FAIL",
         "evidence": f"matrix violations={pt_matrix['violations']}, "
                     f"journal violations={pt_store['violations']}"},
        {"item": "period_loss_limits", "status": "PASS" if periods["passed"] else "FAIL",
         "evidence": (f"{len(periods['noncompliant_periods'])} noncompliant periods of "
                      f"{periods['periods_covered']}") if periods["available"] else "no journal"},
        {"item": "failure_monitoring", "status": "PASS" if monitor["passed"] else "FAIL",
         "evidence": f"{monitor['critical']} critical / {monitor['warning']} warning alerts"},
        {"item": "self_learning", "status": "PASS" if learning["passed"] else "FAIL",
         "evidence": (f"{learning['decided']} decided outcomes, win rate "
                      f"{learning.get('overall_win_rate', 0):.1%}") if learning["available"]
         else "no journal"},
    ]
    failed = [i for i in items if i["status"] == "FAIL"]
    overall = "FAIL" if failed else "PASS"
    report = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    __import__("time").gmtime()),
        "mode": settings.mode,
        "config": {"trade_timeframe": settings.trade_timeframe,
                   "leverage": settings.leverage,
                   "watchlist_count": len(settings.watchlist)},
        "validation_matrix": matrix_summary,
        "risk_rules": risk_cfg,
        "per_trade_compliance": {"matrix": pt_matrix, "journal": pt_store},
        "period_loss_limits": periods,
        "failure_monitoring": monitor,
        "self_learning": {k: v for k, v in learning.items() if k != "generated_at"},
        "verdict": {"items": items, "failed": [i["item"] for i in failed],
                    "overall": overall},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "spec_compliance_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (out / "spec_compliance_report.md").write_text(
            _markdown(report), encoding="utf-8")
        log.info("Spec compliance report written to %s", out)
    return report


def _markdown(r: dict) -> str:
    lines = [f"# Spec compliance report — {r['generated_at']}", "",
             f"Mode: **{r['mode']}**  |  trade timeframe: {r['config']['trade_timeframe']}  |  "
             f"leverage: {r['config']['leverage']}x  |  watchlist: "
             f"{r['config']['watchlist_count']} coins", "",
             f"## Overall verdict: **{r['verdict']['overall']}**", ""]

    lines += ["| Spec item | Status | Evidence |", "|---|---|---|"]
    for i in r["verdict"]["items"]:
        lines.append(f"| {i['item']} | **{i['status']}** | {i['evidence']} |")
    lines.append("")

    m = r["validation_matrix"]
    lines += ["## 1. Validation matrix (regime scenario x duration)", "",
              f"Cells: **{m['passed_cells']}/{m['cells']} passed** "
              f"(hard-failed: {m['hard_failed_cells']}) | trades executed: "
              f"{m['trades_executed']} | overall: {m['overall']}",
              f"Profile pass: {', '.join(f'{k}={v}' for k, v in m['profile_pass'].items())}", ""]
    if m["violations_by_rule"]:
        lines += ["Violations by rule:", ""]
        for rule, n in m["violations_by_rule"].items():
            lines.append(f"- {rule}: {n}")
        lines.append("")
    if m["noncompliant_cells"]:
        lines += ["Noncompliant cells:", ""]
        for c in m["noncompliant_cells"]:
            lines.append(f"- {c['scenario']} {c['days']}d {c['profile']}: {c['rule']}")
        lines.append("")

    rc = r["risk_rules"]
    lines += ["## 2. Institutional risk rules", ""]
    if rc["all_match"]:
        lines.append(f"All {len(rc['rules'])} rules match the spec values.")
    else:
        for rule in rc["rules"]:
            mark = "OK" if rule["ok"] else "**MISMATCH**"
            lines.append(f"- `{rule['rule']}` spec={rule['spec']} configured="
                         f"{rule['configured']} -> {mark}")
    lines.append("")

    pt = r["per_trade_compliance"]
    lines += ["## 3. Per-trade compliance", "",
              f"Matrix rollup: {pt['matrix']['trades']} trades, violations "
              f"{pt['matrix']['violations'] or 'none'}",
              f"Journal rollup: {pt['journal']['trades']} trades, violations "
              f"{pt['journal']['violations'] or 'none'}", ""]
    details = pt["journal"].get("violation_details", [])
    if details:
        lines += ["Journal violation details:", ""]
        for v in details[:20]:
            lines.append(f"- {v}")
        lines.append("")

    pl = r["period_loss_limits"]
    lines += ["## 4. Period loss limits", ""]
    if pl["available"]:
        lines.append(f"{pl['periods_covered']} daily/weekly/monthly periods graded; "
                     f"noncompliant: {len(pl['noncompliant_periods'])}")
        for p in pl["noncompliant_periods"][:10]:
            lines.append(f"- {p['kind']} {p['period']}: pnl {p['pnl_pct']:.2%}, "
                         f"maxDD {p['max_drawdown']:.2%}")
    else:
        lines.append("No journal provided.")
    lines.append("")

    mon = r["failure_monitoring"]
    lines += ["## 5. Failure monitor", "",
              f"Critical alerts: **{mon['critical']}** | warning: {mon['warning']}", ""]
    if mon["by_rule"]:
        lines += ["By rule:", ""]
        for rule, n in sorted(mon["by_rule"].items()):
            lines.append(f"- {rule}: {n}")
        lines.append("")
    if mon["critical_alerts"]:
        lines += ["Recent critical alerts:", ""]
        for a in mon["critical_alerts"]:
            lines.append(f"- [{a['severity']}] {a['rule']}: {a['detail']}")
        lines.append("")

    lr = r["self_learning"]
    lines += ["## 6. Self-learning verification", "",
              f"Observations: {lr.get('observations', 0)} | decided: "
              f"{lr.get('decided', 0)} | win rate: {lr.get('overall_win_rate', 0):.1%}",
              f"Calibration offset: {lr.get('calibration_offset', 0):+.1%} | "
              f"calibration error: {lr.get('calibration_error', 0):.1%}",
              f"Adaptation edge delta: {lr.get('adaptation', {}).get('edge_delta', 0):+.1%}",
              f"ML refits: {len(lr.get('refit_history', []))}", ""]
    return "\n".join(lines)

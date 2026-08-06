"""Reporting: per-trade reports, performance statistics, export to JSON."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .store import TradeStore

log = logging.getLogger("ta_agent.reporting")

PERIOD_LIMITS = {"daily": "daily", "weekly": "weekly", "monthly": "monthly"}


def format_trade_report(plan) -> str:
    """Render the spec-required trade report as text."""
    r = plan.as_report()
    lines = [
        "=" * 62,
        f"TRADE REPORT - {r['coin']} ({r['pair']})",
        "=" * 62,
        f"  Side                    : {r['side'].upper()}",
        f"  Entry                   : {r['entry']}",
        f"  Stop Loss               : {r['stop_loss']}",
        f"  Take Profit             : {r['take_profit']}",
        f"  Risk %                  : {r['risk_%']}",
        f"  Probability %           : {r['probability_%']}",
        f"  Confidence %            : {r['confidence_%']}",
        f"  Expected Return         : {r['expected_return']}",
        f"  Position Size           : {r['position_size']}",
        f"  Notional (USDT)         : {r['notional']}",
        f"  R:R                     : {r['rr']}",
        f"  Reason for Entry        : {r['reason']}",
        f"  Trigger                 : {r['trigger']}",
        f"  Timeframe               : {r['timeframe']}",
        f"  Technical Signals       : {json.dumps(r['technical_signals'])}",
        f"  AI Signals              : {json.dumps(r['ai_signals'])}",
        f"  News Impact             : {r['news_impact']}",
        f"  Risk Assessment         : {r['risk_assessment']}",
        f"  Expected Holding Time   : {r['expected_holding_time']}",
        "=" * 62,
    ]
    return "\n".join(lines)


def compute_stats(trades: List[dict], equity_curve: Optional[pd.DataFrame] = None) -> dict:
    if not trades:
        return {"trades": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    prof_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    stats = {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "profit_factor": float(prof_factor) if np.isfinite(prof_factor) else float("inf"),
        "total_pnl": float(sum(pnls)),
        "max_consecutive_losses": _max_streak(pnls),
    }
    if equity_curve is not None and len(equity_curve):
        eq = equity_curve["equity"]
        rets = eq.pct_change().dropna()
        if len(rets) and rets.std() > 0:
            stats["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(365 * 24))
        stats["max_drawdown"] = float(equity_curve["drawdown"].min())
        stats["final_equity"] = float(eq.iloc[-1])
        stats["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1)
    return stats


def _max_streak(pnls: List[float]) -> int:
    worst = cur = 0
    for p in pnls:
        cur = cur + 1 if p < 0 else 0
        worst = max(worst, cur)
    return worst


# ---------------------------------------------------------------------------
# Per-trade reports (spec: trade_id, entry date, regime, funding, OI, whale,
# order-flow, macro event, lessons learned)
# ---------------------------------------------------------------------------
def _to_dict(row) -> dict:
    return dict(row) if not isinstance(row, dict) else row


def _iso(ms) -> str:
    if not ms:
        return ""
    try:
        return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()
    except Exception:
        return ""


def lessons_for_trade(r: dict) -> dict:
    """Auto-generated post-trade review from a closed-trade row."""
    outcome = r.get("outcome") or "breakeven"
    conf = float(r.get("confidence") or 0.0)
    rr_plan = float(r.get("rr") or 0.0)
    exit_reason = str(r.get("exit_reason") or "")
    pnl = float(r.get("pnl") or 0.0)
    notional = float(r.get("notional") or 0.0)
    risk_pct = float(r.get("risk_pct") or 0.01)
    trigger = r.get("trigger") or "?"
    realized_rr = (pnl / notional) / risk_pct if notional > 0 else 0.0
    entry_ts = r.get("entry_time")
    exit_ts = r.get("exit_time")
    hold_h = (float(exit_ts) - float(entry_ts)) / 3.6e6 if entry_ts and exit_ts else 0.0
    notes: List[str] = []
    low = exit_reason.lower()
    if outcome == "loss":
        notes.append(f"Loser on {trigger} setup at {conf:.0%} confidence - flag trigger for selectivity review")
        if "stop" in low:
            notes.append("Stopped at planned level: risk correctly capped at 1%")
        if "timeout" in low:
            notes.append("Exited on timeout before thesis resolved - expected-holding-time estimate needs tightening")
        if "news" in low:
            notes.append("Forced out by event risk - check news-blackout sizing")
        if "reversal" in low:
            notes.append("Trend reversal exit: higher-TF alignment was too weak to hold")
    else:
        notes.append(f"Winner from {trigger} at {conf:.0%} confidence")
        if realized_rr >= 2.5:
            notes.append(f"Full {realized_rr:.1f}R captured - let winners run")
    return {
        "outcome": outcome,
        "realized_rr": round(realized_rr, 2),
        "holding_hours": round(hold_h, 1),
        "exit_reason": exit_reason,
        "notes": notes,
    }


def build_trade_report(r: dict) -> dict:
    """Full spec per-trade report from a store row."""
    d = _to_dict(r)
    meta = d.get("meta")
    try:
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    except Exception:
        meta = {}
    try:
        context = json.loads(d.get("context")) if d.get("context") else {}
    except Exception:
        context = {}
    try:
        lessons = json.loads(d.get("lessons")) if d.get("lessons") else {}
    except Exception:
        lessons = {}
    report = {
        "trade_id": d.get("id"),
        "trade_key": d.get("trade_key"),
        "coin": d.get("coin"),
        "pair": d.get("pair"),
        "side": d.get("side"),
        "entry_date": _iso(d.get("entry_time")),
        "exit_date": _iso(d.get("exit_time")),
        "entry": d.get("entry"),
        "exit": d.get("exit_price"),
        "quantity": d.get("quantity"),
        "notional": d.get("notional"),
        "stop_loss": d.get("stop_loss"),
        "take_profit": d.get("take_profit"),
        "risk_pct": d.get("risk_pct"),
        "confidence": d.get("confidence"),
        "probability": d.get("probability"),
        "rr": d.get("rr"),
        "trigger": d.get("trigger"),
        "timeframe": d.get("timeframe"),
        "reason": d.get("reason"),
        "regime": d.get("regime"),
        "funding": context.get("funding"),
        "oi": context.get("oi"),
        "whale": context.get("whale"),
        "order_flow": context.get("order_flow"),
        "macro_event": context.get("macro_event"),
        "event_risk": context.get("event_risk"),
        "news_impact": (meta.get("report") or {}).get("news_impact"),
        "pnl": d.get("pnl"),
        "pnl_pct": d.get("pnl_pct"),
        "fees": d.get("fees"),
        "outcome": d.get("outcome"),
        "exit_reason": d.get("exit_reason"),
        "lessons_learned": lessons or lessons_for_trade(d),
    }
    return {k: v for k, v in report.items()}


def trade_reports(store: TradeStore, limit: Optional[int] = None) -> List[dict]:
    rows = store.closed_trades()
    if limit:
        rows = rows[-limit:]
    return [build_trade_report(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Daily / weekly / monthly reports
# ---------------------------------------------------------------------------
def _period_key(ts_ms, kind: str) -> str:
    t = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
    if kind == "daily":
        return t.strftime("%Y-%m-%d")
    if kind == "weekly":
        iso = t.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return t.strftime("%Y-%m")


def _period_start(ts_ms, kind: str) -> str:
    t = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
    if kind == "daily":
        return t.strftime("%Y-%m-%d")
    if kind == "weekly":
        monday = (t - pd.Timedelta(days=t.weekday())).strftime("%Y-%m-%d")
        return monday
    return t.strftime("%Y-%m-01")


def periodic_report(store: TradeStore, settings, kinds=("daily", "weekly", "monthly"),
                    out_dir: Optional[Path] = None) -> dict:
    """Stats + compliance per period. Writes JSON and markdown to ``out_dir``."""
    risk_cfg = settings.risk
    initial = float(settings.backtest.get("initial_capital", 10_000.0)) if settings.backtest else 10_000.0
    closed = [dict(r) for r in store.closed_trades()]
    equity = pd.DataFrame(store.equity_curve())
    if not equity.empty:
        ts_num = pd.to_numeric(equity["ts"], errors="coerce")
        equity = equity.assign(ts=ts_num).dropna(subset=["ts"])
        equity = equity.set_index(pd.to_datetime(equity["ts"], unit="ms", utc=True))

    out: Dict[str, list] = {}
    for kind in kinds:
        groups: Dict[str, List[dict]] = {}
        for t in closed:
            key = _period_key(t["entry_time"], kind)
            groups.setdefault(key, []).append(t)
        rows = []
        for key, trades in sorted(groups.items()):
            pnls = [float(t.get("pnl") or 0.0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            total = float(sum(pnls))
            start_ts = trades[0]["entry_time"]
            start_label = _period_start(start_ts, kind)
            start_eq = _equity_at(equity, start_ts)
            mdd = _period_mdd(equity, start_ts, kind)
            row = {
                "period": key,
                "start": start_label,
                "trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0.0,
                "total_pnl": round(total, 2),
                "pnl_vs_period_start": round(total / start_eq, 4) if start_eq else 0.0,
                "largest_win": round(max(wins), 2) if wins else 0.0,
                "largest_loss": round(min(losses), 2) if losses else 0.0,
                "avg_holding_hours": round(float(np.mean([(float(t.get("exit_time") or 0) - float(t.get("entry_time") or 0)) / 3.6e6 for t in trades])), 1),
                "max_drawdown": round(mdd, 4),
                "max_consecutive_losses": _max_streak(pnls),
            }
            row["compliant"] = _period_compliant(row, total, initial, mdd, risk_cfg, kind)
            rows.append(row)
        out[kind] = sorted(rows, key=lambda r: r["period"], reverse=True)

    report = {"as_of": pd.Timestamp.now("UTC").isoformat(),
              "limits": {k: float(risk_cfg.get(PERIOD_LIMITS[k], 0.0)) for k in kinds},
              "periods": out}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "periodic_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (out_dir / "periodic_report.md").write_text(
            _periodic_markdown(report), encoding="utf-8")
        log.info("Periodic reports written to %s", out_dir)
    return report


def _equity_at(equity: pd.DataFrame, ts_ms: int) -> float:
    if equity is None or equity.empty:
        return 0.0
    pos = equity.index.searchsorted(pd.Timestamp(int(ts_ms), unit="ms", tz="UTC"), side="right") - 1
    if pos < 0:
        return float(equity["equity"].iloc[0])
    return float(equity["equity"].iloc[pos])


def _period_mdd(equity: pd.DataFrame, ts_ms: int, kind: str) -> float:
    if equity is None or equity.empty:
        return 0.0
    start = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
    if kind == "daily":
        end = start + pd.Timedelta(days=1)
    elif kind == "weekly":
        end = start + pd.Timedelta(days=7)
    else:
        end = start + pd.Timedelta(days=31)
    mask = (equity.index >= start) & (equity.index < end)
    seg = equity.loc[mask]
    if seg.empty:
        return 0.0
    dd = seg["drawdown"].min()
    return float(dd) if dd == dd else 0.0


def _period_compliant(row: dict, total: float, initial: float, mdd: float,
                      risk_cfg: dict, kind: str) -> bool:
    limit = float(risk_cfg.get(PERIOD_LIMITS[kind], 0.0) or 0.0)
    loss_pct = -total / initial if initial else 0.0
    if kind == "daily":
        return loss_pct <= limit + 1e-9
    return loss_pct <= limit + 1e-9 and mdd >= -limit - 1e-9


def _periodic_markdown(report: dict) -> str:
    lines = [f"# Periodic report — {report['as_of']}", "",
             "| Period | Kind | Trades | Win% | PnL | PnL% | MaxDD | Compliant |"]
    lines.append("|---|---|---|---|---|---|---|---|")
    for kind, rows in report["periods"].items():
        for r in rows:
            lines.append(
                f"| {r['period']} | {kind} | {r['trades']} | {r['win_rate']:.0%} | "
                f"{r['total_pnl']:.2f} | {r['pnl_vs_period_start']:.2%} | "
                f"{r['max_drawdown']:.2%} | {'yes' if r['compliant'] else '**NO**'} |")
    if not any(report["periods"].values()):
        lines.append("No closed trades in journal yet.")
    return "\n".join(lines)


def summarize_store(store: TradeStore, out_path: Optional[Path] = None) -> dict:
    closed = [dict(r) for r in store.closed_trades()]
    stats = compute_stats(closed)
    curve = pd.DataFrame(store.equity_curve())
    if not curve.empty:
        stats.update({
            "sharpe": stats.get("sharpe", 0.0),
            "max_drawdown": stats.get("max_drawdown", float(curve["drawdown"].min())),
            "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else 0.0,
        })
    summary = {
        "as_of": pd.Timestamp.now("UTC").isoformat(),
        "stats": stats,
        "recent_trades": [dict(r) for r in store.recent_trades(20)],
    }
    if out_path:
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("Report written to %s", out_path)
    return summary

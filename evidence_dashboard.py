#!/usr/bin/env python
"""Evidence-progress dashboard for the paper-trading tracks.

Tracks the "30 decided trades" evidence bar across the three live paper
profiles (1h55, 1h75, 15m75) plus the main data journal, by reading the
journals read-only.  Safe to run while the daemons hold the DBs open.

Usage:
    python evidence_dashboard.py [--watch 60] [--bar 30]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent

TRACKS = [
    ("1h55  (1h / conf 55%)", ROOT / "data" / "practice_1h55" / "journal.db"),
    ("1h75  (1h / conf 75%)", ROOT / "data" / "practice_1h75" / "journal.db"),
    ("15m75 (15m / conf 75%)", ROOT / "data" / "practice_15m75" / "journal.db"),
    ("main  (data/journal.db)", ROOT / "data" / "journal.db"),
]

_QUERY_CLOSED = "SELECT outcome, pnl, exit_time FROM trades WHERE exit_time IS NOT NULL"
_QUERY_OPEN = "SELECT side, entry_time FROM trades WHERE exit_time IS NULL"
_QUERY_EQUITY = "SELECT ts, equity, peak FROM equity ORDER BY ts"
_QUERY_ALERTS = "SELECT severity, COUNT(*) FROM monitor_alerts GROUP BY severity"
_QUERY_SNAPSHOTS = "SELECT COUNT(*) FROM equity"


def _ro(path: Path):
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _read_track(name: str, path: Path) -> dict:
    out = {"name": name, "path": path, "exists": False}
    if not path.exists():
        return out
    out["exists"] = True
    out["age_sec"] = int(time.time() - path.stat().st_mtime)
    try:
        con = _ro(path)
    except sqlite3.Error as exc:
        out["error"] = str(exc)
        return out
    try:
        cur = con.cursor()
        try:
            out["snapshots"] = cur.execute(_QUERY_SNAPSHOTS).fetchone()[0]
        except sqlite3.Error:
            out["snapshots"] = 0
        closed = cur.execute(_QUERY_CLOSED).fetchall()
        open_rows = cur.execute(_QUERY_OPEN).fetchall()
        eq = cur.execute(_QUERY_EQUITY).fetchall()
        alerts = {r[0]: r[1] for r in cur.execute(_QUERY_ALERTS).fetchall()}
    except sqlite3.Error as exc:
        out["error"] = str(exc)
        return out
    finally:
        con.close()

    out["open"] = len(open_rows)
    out["closed"] = len(closed)
    out["decided"] = out["open"] + out["closed"]
    wins = sum(1 for r in closed if r[0] == "win")
    out["win_rate"] = (wins / len(closed)) if closed else None
    out["net_pnl"] = sum(float(r[1] or 0.0) for r in closed)
    if eq:
        out["last_equity"] = float(eq[-1][1])
        out["peak_equity"] = float(eq[-1][2])
    else:
        out["last_equity"] = None
        out["peak_equity"] = None
    out["alerts"] = alerts
    if out["closed"]:
        out["first_exit_ts"] = min(int(r[2]) for r in closed if r[2])
        out["last_exit_ts"] = max(int(r[2]) for r in closed if r[2])
    return out


def _fmt_age(sec: int) -> str:
    if sec < 600:
        return f"{sec}s (live)"
    h = sec // 3600
    return f"{h}h{sec % 3600 // 60}m (STALE)"


def _fmt_dt(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

def _fmt_ts(ts: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def render(bar: int) -> str:
    rows = [_read_track(n, p) for n, p in TRACKS]
    now = time.time()
    decided_total = sum(r.get("decided", 0) for r in rows if r.get("exists"))
    lines = []
    lines.append(f"Evidence dashboard  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  "
                 f"(bar = {bar} decided trades)")
    lines.append("-" * 104)
    lines.append(f"{'track':<22}{'status':<22}{'snaps':>6} {'open':>5} {'closed':>6} "
                 f"{'decided':>8} {'win%':>6} {'netPnL':>10} {'last eq':>10}")
    lines.append("-" * 104)
    for r in rows:
        if not r.get("exists"):
            lines.append(f"{r['name']:<22}no journal yet")
            continue
        if r.get("error"):
            lines.append(f"{r['name']:<22}READ ERROR: {r['error']}")
            continue
        wr = f"{r['win_rate']:.0%}" if r["win_rate"] is not None else "-"
        eq = f"{r['last_equity']:,.0f}" if r["last_equity"] is not None else "-"
        pnl = f"{r['net_pnl']:+.2f}" if r["net_pnl"] else "0.00"
        lines.append(f"{r['name']:<22}{_fmt_age(r['age_sec']):<22}{r['snapshots']:>6} "
                     f"{r['open']:>5} {r['closed']:>6} {r['decided']:>8} "
                     f"{wr:>6} {pnl:>10} {eq:>10}")
    lines.append("-" * 104)
    total = sum(r.get("decided", 0) for r in rows if r.get("exists"))
    lines.append(f"TOTAL decided trades: {total} / {bar}   "
                 f"({max(0.0, total / bar):.0%} of evidence bar)")
    lines.append(f"Snapshot age threshold: <10min = live, >=10min = STALE")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper-track evidence dashboard")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="refresh every N seconds (0 = single shot)")
    ap.add_argument("--bar", type=int, default=30, help="evidence trade bar")
    args = ap.parse_args()

    while True:
        print(render(args.bar))
        if args.watch <= 0:
            break
        time.sleep(args.watch)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

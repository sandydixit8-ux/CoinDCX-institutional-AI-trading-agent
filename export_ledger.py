#!/usr/bin/env python
"""Export transaction/P&L records for tax purposes, plus ledger integrity.

Reads a journal.db read-only and writes CSV files:

* ``trades_YYYYMMDD.csv`` - one row per closed trade (realized P&L, fees)
* ``open_YYYYMMDD.csv``  - currently open positions
* prints realized P&L summary + audit-chain verification result

Usage:
    python export_ledger.py [--journal data/journal.db] [--out data/ledger]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent.store import TradeStore  # noqa: E402  (reused for ledger verify)

COLS = ["coin", "pair", "side", "entry", "exit", "qty", "entry_time", "exit_time",
        "reason", "exit_reason", "pnl", "fees", "confidence", "outcome"]


def _ro(path: Path):
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _rows(path: Path, where: str):
    con = _ro(path)
    try:
        q = ("SELECT coin, pair, side, entry, exit_price AS exit, quantity AS qty, "
             "entry_time, exit_time, reason, exit_reason AS exit_reason, "
             "pnl, fees, confidence, outcome FROM trades " + where)
        return [dict(r) for r in con.execute(q).fetchall()]
    finally:
        con.close()


def _csv_row(r: dict) -> list:
    def t(ms):
        if not ms:
            return ""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [r["coin"], r["pair"], r["side"], r["entry"], r["exit"], r["qty"],
            t(r["entry_time"]), t(r["exit_time"]), r["reason"], r["exit_reason"],
            r["pnl"], r["fees"], r["confidence"], r["outcome"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Transaction/P&L export for tax records")
    ap.add_argument("--journal", default=str(Path("data") / "journal.db"))
    ap.add_argument("--out", default=str(Path("data") / "ledger"))
    args = ap.parse_args()

    db = Path(args.journal)
    if not db.exists():
        print(f"journal not found: {db}")
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    closed = _rows(db, "WHERE exit_time IS NOT NULL ORDER BY exit_time")
    open_ = _rows(db, "WHERE exit_time IS NULL ORDER BY entry_time")

    for name, rows in (("trades", closed), ("open", open_)):
        path = out_dir / f"{name}_{stamp}.csv"
        lines = [",".join(COLS)]
        for r in rows:
            lines.append(",".join("" if v is None else str(v) for v in _csv_row(r)))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name:>6}: {len(rows):>4} rows -> {path}")

    total_pnl = sum(float(r["pnl"] or 0.0) for r in closed)
    total_fees = sum(float(r["fees"] or 0.0) for r in closed)
    wins = sum(1 for r in closed if (r["pnl"] or 0.0) > 0)
    print(f"\nRealized P&L: {total_pnl:+.2f} USDT | fees: {total_fees:.2f} "
          f"| closed: {len(closed)} (win {wins}/{len(closed)})")

    # Audit-chain integrity for this journal
    try:
        store = TradeStore(str(db))
        print("Ledger hash-chain:", store.verify_ledger())
        store.close()
    except sqlite3.Error as exc:  # pragma: no cover
        print(f"ledger verify unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

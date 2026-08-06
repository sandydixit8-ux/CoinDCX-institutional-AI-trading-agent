#!/usr/bin/env python
"""Run the bot in PAPER trading mode.

Paper mode needs no API keys. Use --synthetic for offline GBM data.

Usage:
    python run_paper.py [--cycles 20] [--interval 60] [--synthetic] [--dry 2]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent.bot import TradingBot
from ta_agent.reporting import periodic_report, trade_reports
from ta_agent.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="CoinDCX paper-trading bot")
    parser.add_argument("--cycles", type=int, default=20, help="number of cycles (None=forever)")
    parser.add_argument("--interval", type=float, default=120.0, help="seconds between cycles")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic GBM data (offline)")
    parser.add_argument("--dry", type=int, default=2, help="warmup cycles without trading")
    parser.add_argument("--fresh", action="store_true",
                        help="wipe the trade journal before the run (clean per-run summary)")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = Settings.load(args.config)
    s.mode = "paper"
    if args.fresh:
        db = Path(s.data_dir) / "journal.db"
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(db) + suffix).unlink(missing_ok=True)
            except PermissionError:
                pass
    if s.api_key and s.api_secret and not args.synthetic:
        print("NOTE: API keys detected - using live market data but PAPER execution.")
    bot = TradingBot(s, synthetic=args.synthetic)
    bot.run(cycles=args.cycles, interval_seconds=args.interval, dry_cycles=args.dry)
    summary = bot.summary(Path(s.data_dir) / "paper_report.json")
    print("\n===== PAPER SUMMARY =====")
    print(json_print(summary["stats"]))
    write_reports(bot.store, s)
    return 0


def write_reports(store, settings) -> None:
    reports_dir = Path(settings.data_dir) / "reports" / "paper"
    reports_dir.mkdir(parents=True, exist_ok=True)
    periodic_report(store, settings, out_dir=reports_dir)
    (reports_dir / "trades.json").write_text(
        json.dumps(trade_reports(store), indent=2, default=str), encoding="utf-8")
    print(f"Reports written to {reports_dir}")


def json_print(d) -> str:
    import json
    return json.dumps(d, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())

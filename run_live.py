#!/usr/bin/env python
"""Run the bot in LIVE mode against CoinDCX futures.

REQUIRES COINDCX_API_KEY and COINDCX_API_SECRET in .env / environment.

WARNING: This places real orders. Start with paper mode and small risk
settings; never point live money at this without reviewing every parameter.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent.bot import TradingBot
from ta_agent.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="CoinDCX LIVE trading bot")
    parser.add_argument("--cycles", type=int, default=None, help="number of cycles (None=forever)")
    parser.add_argument("--interval", type=float, default=300.0, help="seconds between cycles")
    parser.add_argument("--dry", type=int, default=3, help="warmup cycles without trading")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = Settings.load(args.config)
    s.mode = "live"
    if not s.api_key or not s.api_secret:
        print("ERROR: COINDCX_API_KEY / COINDCX_API_SECRET must be set for live mode.")
        return 2

    bot = TradingBot(s)
    print("=" * 60)
    print("LIVE MODE - REAL ORDERS WILL BE PLACED ON COINDCX")
    print("=" * 60)
    bot.run(cycles=args.cycles, interval_seconds=args.interval, dry_cycles=args.dry)
    summary = bot.summary(Path(s.data_dir) / "live_report.json")
    print("\n===== LIVE SUMMARY =====")
    import json
    print(json.dumps(summary["stats"], indent=2, default=str))
    from ta_agent.reporting import periodic_report, trade_reports
    reports_dir = Path(s.data_dir) / "reports" / "live"
    reports_dir.mkdir(parents=True, exist_ok=True)
    periodic_report(bot.store, s, out_dir=reports_dir)
    (reports_dir / "trades.json").write_text(
        json.dumps(trade_reports(bot.store), indent=2, default=str), encoding="utf-8")
    print(f"Reports written to {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

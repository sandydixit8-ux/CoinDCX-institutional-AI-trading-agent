#!/usr/bin/env python
"""Run a backtest on CoinDCX historical data (or synthetic data).

Usage:
    python run_backtest.py [--synthetic] [--coins BTC,ETH] [--bars 1500]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent.ai_ensemble import ConfidenceEngine
from ta_agent.backtest import Backtester
from ta_agent.bot import TradingBot
from ta_agent.reporting import periodic_report, trade_reports
from ta_agent.settings import Settings
from ta_agent.store import TradeStore


def main() -> int:
    parser = argparse.ArgumentParser(description="CoinDCX quant backtester")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic GBM data")
    parser.add_argument("--coins", default=None, help="comma-separated coin subset")
    parser.add_argument("--bars", type=int, default=None, help="override trade-TF bars")
    parser.add_argument("--loose", action="store_true",
                        help="relax confidence/RR gates for smoke testing")
    parser.add_argument("--config", default="config.json", help="config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = Settings.load(args.config)
    s.mode = "backtest"
    if args.coins:
        s.watchlist = [c.strip() for c in args.coins.split(",")]
    if args.bars:
        for tf in s.max_data_bars:
            s.max_data_bars[tf] = args.bars
    if args.loose:
        s.risk["min_confidence"] = 0.0
        s.risk["min_rr"] = 0.5

    # fresh store per run: synthetic runs are deterministic, so repeated runs
    # would otherwise collide on the trades.trade_key UNIQUE constraint
    db = Path(s.data_dir) / "backtest.db"
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(db) + suffix).unlink(missing_ok=True)
        except PermissionError:
            pass

    bot = TradingBot(s, store_path=db, synthetic=args.synthetic)
    frames = bot.feed.get_frames(s.watchlist)

    if not frames:
        print("No data available - check network or use --synthetic.")
        return 1

    store = TradeStore(str(db))
    confidence = ConfidenceEngine(ml_scorer=bot.ml_scorer, threshold=s.min_confidence)
    bot.strategy.conf = confidence
    bt = Backtester(s, bot.strategy, bot.risk, store=store, portfolio=bot.portfolio)
    result = bt.run(frames, initial_capital=s.backtest.get("initial_capital", 10_000.0),
                    start=s.backtest.get("start"), end=s.backtest.get("end"))

    print("\n================ BACKTEST RESULT ================")
    for k, v in result.metrics.items():
        if k in ("final_equity",):
            print(f"  {k:24s}: {v:,.2f}")
        elif isinstance(v, float):
            print(f"  {k:24s}: {v:.4f}")
        else:
            print(f"  {k:24s}: {v}")

    out = Path(s.data_dir) / "backtest_report.json"
    out.write_text(json.dumps({"metrics": result.metrics, "trades": result.trades[:50]},
                              indent=2, default=str), encoding="utf-8")
    eq = Path(s.data_dir) / "backtest_equity.csv"
    result.equity_curve.to_csv(eq, index=False)

    reports_dir = Path(s.data_dir) / "reports" / "backtest"
    reports_dir.mkdir(parents=True, exist_ok=True)
    periodic_report(store, s, out_dir=reports_dir)
    (reports_dir / "trades.json").write_text(
        json.dumps(trade_reports(store), indent=2, default=str), encoding="utf-8")

    print(f"\nReport:  {out}")
    print(f"Equity:  {eq}")
    print(f"Periodic reports: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

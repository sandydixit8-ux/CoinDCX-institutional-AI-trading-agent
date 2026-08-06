#!/usr/bin/env python
"""Run the spec's regime-scenario x duration validation matrix.

Generates synthetic markets for every market regime the spec asks to test
(bull, bear, sideways, high volatility, low/high liquidity, flash crash,
pump & dump, black swan, mixed regimes), backtests the strategy over the
requested durations (1/7/30/90/365 days), and grades it against the spec's
failure conditions.

Usage:
    python run_validation.py [--scenarios bull,bear] [--durations 1,7,30,90,365]
                             [--coins BTC,ETH,SOL] [--warmup-days 60]
                             [--profiles strict,edge] [--edge-confidence 0.70]
                             [--end 2026-08-01T00:00:00Z]
                             [--seed 7] [--config config.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent import timestamps

from ta_agent.ai_ensemble import ConfidenceEngine
from ta_agent.backtest import Backtester
from ta_agent.bot import TradingBot
from ta_agent.news_engine import EconomicCalendar
from ta_agent.portfolio import PortfolioManager
from ta_agent.risk import RiskManager
from ta_agent.scenarios import SCENARIOS, generate_market
from ta_agent.settings import Settings
from ta_agent.validation import compliance, extended_metrics

DEFAULT_DURATIONS = [1, 7, 30, 90, 365]
DEFAULT_PROFILES = ["strict", "edge"]

# Profiles:
#   strict -> spec-faithful gates (0.90 conf / 3.0 R:R): documents rejection
#             discipline and capital preservation (expect ~0 trades).
#   edge   -> relaxes ONLY the confidence gate (the heuristic ensemble cannot
#             honestly reach 0.90 before the ML model is trained from >=30
#             real trades); every other risk/exit rule stays strict. Used to
#             exercise entries/exits/regime adaptation and grade the risk
#             engine. The confidence rule is documented, not graded.


def run_one(s: Settings, scenario: str, days: int, coins: list, seed: int,
            warmup_days: int, profile: str, edge_confidence: float,
            end: Optional[str] = None, market: Optional[dict] = None,
            store=None) -> dict:
    market = market if market is not None else generate_market(
        s, scenario, days, coins, seed=seed, warmup_days=warmup_days, end=end)
    orig_conf = s.risk.get("min_confidence")
    try:
        if profile == "edge":
            s.risk["min_confidence"] = edge_confidence
        bot = TradingBot(s, synthetic=True)
        confidence = ConfidenceEngine(ml_scorer=bot.ml_scorer, threshold=s.min_confidence)
        bot.strategy.conf = confidence
        risk = RiskManager(s)
        bt = Backtester(s, bot.strategy, risk, store=store,
                        portfolio=PortfolioManager(s))
        tf_frame = market[coins[0]][s.trade_timeframe]
        ts_dt = timestamps.col_to_datetime(tf_frame["time"])
        end_dt = ts_dt.iloc[-1]
        start_dt = end_dt - pd.Timedelta(days=days)
        result = bt.run(market, initial_capital=s.backtest.get("initial_capital", 10_000.0),
                        start=str(start_dt), end=str(end_dt))
        metrics = extended_metrics(result.metrics, result.trades,
                                   result.equity_curve, days)
        skip = {"confidence_below"} if profile == "edge" else set()
        check = compliance(metrics, result.trades, s, days,
                           calendar=EconomicCalendar(), skip_rules=skip)
        return {"scenario": scenario, "days": days, "profile": profile,
                "metrics": metrics, "compliance": check,
                "n_trades": len(result.trades)}
    finally:
        if orig_conf is not None:
            s.risk["min_confidence"] = orig_conf


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime-scenario validation matrix")
    parser.add_argument("--scenarios", default=None, help="comma-separated subset (default: all)")
    parser.add_argument("--durations", default=None, help="comma-separated days (default: 1,7,30,90,365)")
    parser.add_argument("--coins", default="BTC,ETH,SOL", help="coin subset (default: BTC,ETH,SOL)")
    parser.add_argument("--warmup-days", type=int, default=60, help="indicator warmup history (default 60)")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z",
                        help="fixed market end instant for reproducibility (default 2026-08-01T00:00:00Z)")
    parser.add_argument("--profiles", default=None,
                        help="comma-separated validation profiles: strict,edge (default: strict,edge)")
    parser.add_argument("--edge-confidence", type=float, default=0.70,
                        help="confidence gate used by the edge profile (default 0.70)")
    parser.add_argument("--seed", type=int, default=7, help="deterministic seed")
    parser.add_argument("--persist-store", default=None,
                        help="journal DB to persist trades/equity/context into "
                             "(each cell namespaced by run id)")
    parser.add_argument("--out", default=None,
                        help="report JSON path (default data/validation_report.json)")
    parser.add_argument("--config", default="config.json", help="config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    scenarios = sorted(SCENARIOS) if not args.scenarios else \
        [x.strip() for x in args.scenarios.split(",")]
    durations = DEFAULT_DURATIONS if not args.durations else \
        [int(x) for x in args.durations.split(",")]
    coins = [x.strip() for x in args.coins.split(",")]
    profiles = DEFAULT_PROFILES if not args.profiles else \
        [x.strip() for x in args.profiles.split(",")]

    s = Settings.load(args.config)
    s.mode = "backtest"

    persist_store = None
    if args.persist_store:
        from ta_agent.store import TradeStore
        persist_store = TradeStore(args.persist_store)
        print(f"Persisting trades/equity to: {args.persist_store}")

    print(f"Validation matrix: {len(scenarios)} scenarios x {len(durations)} durations x {len(profiles)} profiles x {len(coins)} coins")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Durations: {', '.join(f'{d}d' for d in durations)}")
    print(f"Profiles : {', '.join(profiles)}"
          f"{'  (edge confidence gate = %.2f, not graded)' % args.edge_confidence if 'edge' in profiles else ''}\n")

    report: dict = {"scenarios": {}}
    start = time.time()
    for scenario in scenarios:
        report["scenarios"][scenario] = {}
        for days in durations:
            report["scenarios"][scenario][f"{days}d"] = {}
            market = None
            for profile in profiles:
                t0 = time.time()
                try:
                    if market is None:
                        market = generate_market(s, scenario, days, coins,
                                                 seed=args.seed,
                                                 warmup_days=args.warmup_days,
                                                 end=args.end)
                    row = run_one(s, scenario, days, coins, args.seed, args.warmup_days,
                                  profile, args.edge_confidence, end=args.end,
                                  market=market, store=persist_store)
                except Exception as exc:  # keep the matrix running
                    logging.getLogger("run_validation").exception("failed %s/%dd/%s", scenario, days, profile)
                    row = {"scenario": scenario, "days": days, "profile": profile,
                           "error": str(exc), "n_trades": 0, "metrics": {},
                           "compliance": {"passed": False}}
                report["scenarios"][scenario][f"{days}d"][profile] = row
                m = row.get("metrics", {})
                c = row.get("compliance", {})
                status = "FAIL" if not c.get("passed") else "PASS"
                print(f"  {scenario:<16s} {days:>4d}d  {profile:<6s} "
                      f"trades={row.get('n_trades', 0):>4d}  "
                      f"return={m.get('total_return', 0.0):>8.2%}  "
                      f"PF={m.get('profit_factor', 0.0):>6.2f}  "
                      f"win%={m.get('win_rate', 0.0):>6.1%}  "
                      f"MDD={m.get('max_drawdown', 0.0):>8.2%}  "
                      f"annRet={m.get('annualized_return', 0.0):>9.2%}  "
                      f"{status:<4s} ({time.time() - t0:5.1f}s)")

    report["summary"] = {
        "elapsed_seconds": round(time.time() - start, 1),
        "profile_pass": {p: all(report["scenarios"][sc][f"{d}d"][p]["compliance"].get("passed", False)
                                for sc in scenarios for d in durations) for p in profiles},
        "overall": "PASS" if all(report["scenarios"][sc][f"{d}d"][p]["compliance"].get("passed", False)
                                  for sc in scenarios for d in durations for p in profiles) else "FAIL",
    }

    out = Path(args.out) if args.out else Path(s.data_dir) / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nPer-profile: {report['summary']['profile_pass']}")
    print(f"Overall: {report['summary']['overall']}   ({time.time() - start:.1f}s)")
    print(f"Report:  {out}")
    return 0 if report["summary"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Generate the consolidated spec-compliance report.

Ties together Phase 1 (validation matrix), Phase 2 (per-trade/periodic reports,
failure monitor) and Phase 3 (self-learning) into one deliverable.

Usage:
    python run_compliance.py [--matrix data/validation_report.json]
                             [--store data/journal.db]
                             [--out data/reports/compliance]
                             [--config config.json]

With no --store the report grades the validation matrix + risk config; add
--store to also grade the journal's per-trade compliance, period limits,
monitor alerts and self-learning evidence.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent.compliance import spec_compliance_report
from ta_agent.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidated spec-compliance report")
    parser.add_argument("--matrix", default=str(Path("data") / "validation_report.json"),
                        help="Phase 1 validation matrix JSON")
    parser.add_argument("--store", default=None,
                        help="trade journal DB (journal.db or backtest.db)")
    parser.add_argument("--skip", action="append", default=[],
                        help="compliance rule to skip in the journal rollup "
                             "(repeatable; e.g. --skip confidence_below for an "
                             "edge-profile journal)")
    parser.add_argument("--out", default=str(Path("data") / "reports" / "compliance"),
                        help="output directory")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = Settings.load(args.config)
    report = spec_compliance_report(s, matrix_path=args.matrix,
                                    store_path=args.store, out_dir=args.out,
                                    skip_rules=set(args.skip))

    print(f"===== SPEC COMPLIANCE: {report['verdict']['overall']} =====")
    for item in report["verdict"]["items"]:
        print(f"  {item['item']:22s}: {item['status']:4s}  {item['evidence']}")
    print(f"\nReport: {Path(args.out).resolve()}")
    return 0 if report["verdict"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

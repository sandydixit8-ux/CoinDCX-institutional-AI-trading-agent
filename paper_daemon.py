#!/usr/bin/env python
"""Run the real-data PAPER trading bot continuously as a background daemon.

Live CoinDCX market data with simulated fills (no real orders). The bot loop
runs forever (``cycles=None``); if it crashes it is restarted with a short
backoff. Periodic reports are refreshed every ``--report-every`` cycles so
progress is visible while the daemon keeps running.

Usage:
    python paper_daemon.py [--interval 120] [--dry 2] [--report-every 60]
"""
from __future__ import annotations

import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ta_agent.bot import TradingBot
from ta_agent.reporting import periodic_report, trade_reports
from ta_agent.settings import Settings

_MUTEX_NAME = "Global\\TradingAgentPaperDaemonMutex"


def _acquire_single_instance():
    """Refuse to start if another daemon is already running, so two bots can
    never write the same journal concurrently. A named mutex is atomic and the
    OS releases it when the holding process exits (or dies), so no stale locks
    can block restarts."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    handle = kernel32.CreateMutexW(None, True, _MUTEX_NAME)
    if not handle:
        sys.exit("failed to create mutex - exiting")
    if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        sys.exit("paper_daemon already running - exiting")
    return handle


def _write_reports(store, settings) -> None:
    reports_dir = Path(settings.data_dir) / "reports" / "paper"
    reports_dir.mkdir(parents=True, exist_ok=True)
    periodic_report(store, settings, out_dir=reports_dir)
    (reports_dir / "trades.json").write_text(
        __import__("json").dumps(trade_reports(store), indent=2, default=str),
        encoding="utf-8")
    logging.getLogger("paper_daemon").info("reports refreshed -> %s", reports_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-data paper trading daemon")
    parser.add_argument("--interval", type=float, default=120.0,
                        help="seconds between bot cycles")
    parser.add_argument("--dry", type=int, default=2,
                        help="warmup cycles without trading")
    parser.add_argument("--report-every", type=int, default=60,
                        help="refresh reports every N cycles")
    parser.add_argument("--max-restarts", type=int, default=20,
                        help="give up after this many consecutive crashes")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    log_file = Path("data") / "paper_daemon.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=5_000_000,
                                  backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[handler])
    log = logging.getLogger("paper_daemon")

    lock = _acquire_single_instance()

    s = Settings.load(args.config)
    s.mode = "paper"

    restarts = 0
    cycles_so_far = 0
    while True:
        bot = TradingBot(s)
        try:
            bot.run(cycles=None, interval_seconds=args.interval,
                    dry_cycles=args.dry, on_cycle=_write_reports,
                    report_every=args.report_every)
            log.info("bot loop exited cleanly; restarting")
        except KeyboardInterrupt:
            log.info("daemon stopped by user")
            return 0
        except Exception as exc:  # pragma: no cover - restart safety net
            restarts += 1
            log.exception("bot crashed (%s/%s); restarting in 15s", restarts, args.max_restarts)
            if restarts >= args.max_restarts:
                log.critical("too many restarts; giving up")
                return 1
            time.sleep(15)
            continue
        restarts = 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())

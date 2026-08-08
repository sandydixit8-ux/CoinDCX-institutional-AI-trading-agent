"""End-to-end orchestrator for paper and live modes.

Per cycle: fetch multi-timeframe data -> compute features/regimes -> news &
microstructure risk -> strategy analysis -> risk approval & sizing -> execute
via broker -> manage open positions (partials, trailing, stops) -> journal &
self-learn.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .ai_ensemble import ConfidenceEngine, MLScorer
from .brokers import Broker, LiveBroker, PaperBroker
from .coindcx_client import CoinDCXClient
from .datafeed import CoinDCXFeed, SyntheticFeed
from .features import FEATURE_COLUMNS, build_features
from .monitor import FailureMonitor
from .news_engine import EconomicCalendar, NewsSentiment
from .portfolio import PortfolioManager
from .regime import detect_regime, trend_alignment
from .reporting import format_trade_report, lessons_for_trade, summarize_store
from .risk import RiskManager
from .self_learning import LearningJournal
from .settings import Settings
from .store import TradeStore
from .strategy import ExitEngine, PositionState, StrategyEngine

log = logging.getLogger("ta_agent.bot")


class TradingBot:
    def __init__(self, settings: Settings, store_path: Optional[Path] = None,
                 synthetic: bool = False):
        self.s = settings
        self.client = CoinDCXClient(settings.api_key, settings.api_secret,
                                    base_url=settings.base_url)
        self.store = TradeStore(str(store_path or Path(settings.data_dir) / "journal.db"))

        self.ml_scorer = MLScorer(FEATURE_COLUMNS)
        self.confidence = ConfidenceEngine(ml_scorer=self.ml_scorer)
        self.strategy = StrategyEngine(settings, self.confidence)
        self.risk = RiskManager(settings, initial_equity=settings.backtest.get("initial_capital", 10_000.0))
        self.portfolio = PortfolioManager(settings)
        self.exit_engine = ExitEngine(settings)
        self.cal = EconomicCalendar()
        self.news = NewsSentiment()
        self.learning = LearningJournal(self.store, buffer_size=settings.learning.get("buffer_size", 300))

        if synthetic:
            self.feed = SyntheticFeed(settings)
            self.broker: Broker = PaperBroker(settings)
        else:
            self.feed = CoinDCXFeed(self.client, settings)
            if settings.is_live():
                if not settings.api_key:
                    raise RuntimeError("Live mode requires COINDCX_API_KEY / COINDCX_API_SECRET")
                self.broker: Broker = LiveBroker(self.client, settings)
            else:
                self.broker = PaperBroker(settings)

        self.open_positions: Dict[str, PositionState] = {}
        self._close_counter = 0
        self._consecutive_errors = 0
        self._last_plans: list = []
        self._last_event_risk = None
        self._equity_synced = False
        self.monitor = FailureMonitor(
            settings, store=self.store,
            alert_file=Path(settings.data_dir) / "monitor_alerts.json")

    # ------------------------------------------------------------------
    def run(self, cycles: Optional[int] = None, interval_seconds: float = 300.0,
            dry_cycles: int = 0, on_cycle=None, report_every: int = 60) -> None:
        """Main loop. ``cycles=None`` runs forever; ``dry_cycles`` warms the
        data/indicators without trading (used at startup). ``on_cycle``, when
        given, is called with ``(store, settings)`` every ``report_every``
        cycles so external processes can refresh reports while running."""
        if self.s.is_backtest():
            coins = list(self.s.watchlist)
        else:
            try:
                coins = self.feed.available_coins()
            except Exception as exc:
                log.warning("available_coins() failed (%s); falling back to watchlist", exc)
                coins = list(self.s.watchlist)
        log.info("Bot starting | mode=%s | coins=%s | trade_tf=%s | min_confidence=%.0f%%",
                 self.s.mode, coins, self.s.trade_timeframe, self.s.min_confidence * 100)
        self._sync_equity()
        cycle = 0
        warm = 0
        while cycles is None or cycle < cycles:
            try:
                self._cycle(coins, warm < dry_cycles)
                self._consecutive_errors = 0
            except Exception:
                log.error("Cycle failed:\n%s", traceback.format_exc())
                self._consecutive_errors += 1
            self.monitor.check(self.risk.state, self.open_positions,
                               plans=self._last_plans,
                               error_rate=min(1.0, self._consecutive_errors),
                               last_event_risk=self._last_event_risk)
            cycle += 1
            warm += 1
            if on_cycle is not None and report_every and cycle % report_every == 0:
                try:
                    on_cycle(self.store, self.s)
                except Exception:
                    log.error("on_cycle hook failed:\n%s", traceback.format_exc())
            time.sleep(interval_seconds)

    # ------------------------------------------------------------------
    def _sync_equity(self) -> None:
        """Size from the REAL account balance in live mode, never the config
        default. Zero/empty balance means no trades get sized. The peak is
        initialised from the real balance on first sync so an underfunded
        wallet does not produce a spurious drawdown vs the config default."""
        if not self.s.is_live():
            return
        try:
            eq = self.broker.get_balances().get("equity")
        except Exception as exc:  # pragma: no cover
            log.error("Could not sync live equity: %s", exc)
            return
        if eq is None:
            return
        eq = float(eq)
        if not self._equity_synced:
            self.risk.state.equity = eq
            self.risk.state.peak_equity = eq
            self._equity_synced = True
            if eq <= 0:
                log.warning("Live USDT equity is %.2f - no trades will be sized "
                            "until the futures wallet is funded", eq)
            else:
                log.info("Live equity synced from broker: %.2f USDT", eq)
            return
        self.risk.state.equity = eq
        self.risk.state.peak_equity = max(self.risk.state.peak_equity, eq)

    def _verify_position_protection(self, coin: str, pair: str, qty: float,
                                    stop_loss: float, take_profit: float) -> None:
        """In live mode, confirm the exchange-side TP/SL exists after entry;
        attach it if missing so an unprotected position is never left behind."""
        if not self.s.is_live():
            return
        try:
            pos = self.broker.get_position(pair)
        except Exception as exc:  # pragma: no cover
            log.error("Could not verify TP/SL for %s: %s", coin, exc)
            return
        if pos and pos.stop_loss and pos.take_profit:
            return
        res = self.broker.set_tpsl(pair, stop_loss, take_profit, qty)
        if not res.get("ok"):
            log.error("ATTENTION %s: TP/SL could not be attached - position is "
                      "UNPROTECTED on the exchange: %s", coin, res.get("error"))

    def _cycle(self, coins: List[str], warmup: bool = False) -> None:
        frames = self.feed.get_frames(coins)
        if not frames:
            log.warning("No market data fetched; skipping cycle")
            return

        self.portfolio.update_correlation({c: tfs.get("1d") for c, tfs in frames.items()})

        mark = {}
        for coin, tfs in frames.items():
            df = tfs.get(self.s.trade_timeframe)
            if df is not None and not df.empty and "close" in df.columns:
                mark[self._pair(coin)] = float(df["close"].iloc[-1])
        if mark:
            self.broker.mark_prices = mark

        event_risk = self.cal.event_risk(blackout_hours=self.s.news.get("blackout_hours", 2.0))
        self._last_event_risk = event_risk
        self._last_plans = []

        # ---- manage exits ----------------------------------------------
        for coin in list(self.open_positions.keys()):
            if coin not in frames:
                continue
            self._manage_position(coin, frames[coin], event_risk)

        # ---- look for entries -------------------------------------------
        if not warmup:
            blocked = self.risk.can_open()
            if blocked:
                log.info("Entries blocked: %s", blocked)
            elif len(self.open_positions) >= self.s.max_positions:
                log.info("Max positions reached (%d)", self.s.max_positions)
            else:
                self._scan_entries(frames, event_risk)

        self._snapshot(frames)

    # ------------------------------------------------------------------
    def _scan_entries(self, frames, event_risk) -> None:
        for coin, tfs in frames.items():
            if coin in self.open_positions:
                continue
            if len(self.open_positions) >= self.s.max_positions:
                break
            try:
                clean = {tf: df.iloc[:-1] for tf, df in tfs.items() if df is not None and len(df) > 1}
                if self.s.trade_timeframe not in clean or clean[self.s.trade_timeframe].empty:
                    continue
                features = {tf: build_features(clean[tf]) for tf in clean}
                regimes = {tf: detect_regime(clean[tf]) for tf in clean if len(clean[tf]) > 30}
                align = trend_alignment(regimes)
                if align < 0.6:
                    continue
                micro = self.feed.get_micro(coin)
                news_impact = self.news.sentiment(coin).get("risk", 0.0)
                er = max(event_risk.risk, news_impact)
                plan = self.strategy.analyze(coin, self._pair(coin), clean, features, regimes,
                                             micro=micro, event_risk=er,
                                             timestamp=int(pd.Timestamp.now("UTC").timestamp() * 1000))
                if plan is None:
                    continue
                ok, reason = self.risk.approve(plan, open_notional=sum(
                    p.notional for p in self.open_positions.values()),
                    open_positions=len(self.open_positions))
                if not ok:
                    log.info("Risk rejected %s %s: %s", coin, plan.side, reason)
                    continue
                atr_pct = 0.0
                last = features.get(self.s.trade_timeframe)
                if last is not None and not last.empty and "atr_pct" in last.columns:
                    atr_pct = float(last["atr_pct"].iloc[-1])
                baseline = float(self.s.risk.get("kelly_cap_pct", 0.25))
                self.risk.size(plan, atr_pct=atr_pct, baseline_atr_pct=baseline)
                if plan.notional <= 0:
                    continue
                pcheck = self.portfolio.check_new_position(
                    coin, plan.notional, self.risk.state.equity,
                    {c: p.notional for c, p in self.open_positions.items()})
                if not pcheck.approved:
                    log.info("Portfolio rejected %s: %s", coin, pcheck.reason)
                    continue
                context = {}
                if micro:
                    context["funding"] = micro.get("funding_proxy")
                    context["oi"] = micro.get("oi") or micro.get("open_interest")
                    context["whale"] = micro.get("whale") or micro.get("whale_activity")
                    context["order_flow"] = micro.get("order_flow") or micro.get("ob_imbalance")
                context["macro_event"] = getattr(event_risk, "nearest_event", None)
                context["event_risk"] = round(er, 3)
                regime = regimes.get(self.s.trade_timeframe) or \
                    (next(iter(regimes.values())) if regimes else None)
                self._execute_entry(plan, features.get(self.s.trade_timeframe),
                                    context=context, regime=regime)
            except Exception as exc:  # pragma: no cover
                log.error("Entry scan failed for %s: %s", coin, exc)

    def _execute_entry(self, plan, feat_df, context: Optional[dict] = None,
                       regime: Optional[str] = None) -> None:
        news_impact = "elevated" if plan.ai_signals.get("penalty", 1.0) < 0.9 else "none"
        plan.news_impact = news_impact
        order = self.broker.place_order(
            plan.pair, "buy" if plan.side == "long" else "sell", plan.position_size,
            order_type="market_order", stop_loss=plan.stop_loss, take_profit=plan.take_profit,
        )
        if order.status == "rejected":
            log.warning("Order rejected for %s: %s", plan.coin, order.meta.get("error"))
            return
        entry_price = order.avg_price or plan.entry
        qty = order.filled_quantity or plan.position_size
        now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)
        trade_key = f"{plan.coin}-{now_ms}"
        feature_vec = None
        if feat_df is not None and len(feat_df):
            row = feat_df.iloc[-1]
            feature_vec = [float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0
                           for c in FEATURE_COLUMNS]
        self.store.record_trade_entry(plan, trade_key, qty, qty * entry_price,
                                      now_ms, features=feature_vec,
                                      context=context, regime=regime)
        self._last_plans.append(plan)
        self.open_positions[plan.coin] = PositionState(
            coin=plan.coin, pair=plan.pair, side=plan.side, entry=entry_price, qty=qty,
            notional=qty * entry_price, stop_loss=plan.stop_loss, take_profit=plan.take_profit,
            entry_time_ms=now_ms, peak_price=entry_price,
            reason=trade_key, confidence=plan.confidence,
        )
        self.risk.state.open_positions += 1
        log.info("OPEN %s %s qty=%.6f entry=%.8f sl=%.8f tp=%.8f conf=%.0f%% rr=%.2f",
                 plan.coin, plan.side.upper(), qty, entry_price, plan.stop_loss,
                 plan.take_profit, plan.confidence * 100, plan.rr)
        log.info(format_trade_report(plan))
        self._verify_position_protection(plan.coin, plan.pair, qty,
                                         plan.stop_loss, plan.take_profit)

    # ------------------------------------------------------------------
    def _manage_position(self, coin: str, tfs, event_risk) -> None:
        pos = self.open_positions[coin]
        df = tfs.get(self.s.trade_timeframe)
        if df is None or df.empty:
            return
        price = float(df["close"].iloc[-1])
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        atr_pct = 0.0
        if len(df) >= 14:
            from . import indicators as ta
            atr_pct = float(ta.atr(df, 14)[-1] / price)
        news = event_risk.risk > 0.5
        reversal = self._quick_reversal(df, pos.side)
        dec = self.exit_engine.evaluate(pos, price, high, low, atr_pct,
                                        news_event=news, signal_reversal=reversal,
                                        now_ms=int(pd.Timestamp.now("UTC").timestamp() * 1000))
        if dec.action == "hold":
            return
        self._close_position(coin, pos, dec.exit_price or price, dec.reason, dec.fraction)

    def _close_position(self, coin: str, pos: PositionState, price: float,
                        reason: str, fraction: float = 1.0) -> None:
        qty = pos.qty * fraction
        order = self.broker.close_position(pos.pair, qty)
        if order is None:
            # Exit failed at the broker (live) or no mark price (paper). The
            # position still exists: do NOT record a fill, do NOT forget it.
            log.error("CLOSE FAILED for %s %s - position kept, will retry next cycle",
                      coin, pos.side)
            return
        fill = order.avg_price if order.avg_price else price
        pnl = (fill - pos.entry) * qty if pos.side == "long" else (pos.entry - fill) * qty
        fees = abs(fill) * qty * self.s.taker_fee()
        net = pnl - fees
        self.risk.on_trade_close(net)
        self.learning.observe("win" if net > 0 else "loss",
                              pos.confidence, 0.5, coin=coin, side=pos.side)
        self.store.record_trade_exit(pos.reason, fill, reason, net, fees)
        if self.s.learning.get("enabled", True):
            for r in self.store.closed_trades():
                if r["trade_key"] == pos.reason:
                    self.store.record_trade_lessons(pos.reason, lessons_for_trade(dict(r)))
                    break
        if fraction >= 1.0:
            self.open_positions.pop(coin, None)
            self.risk.state.open_positions = max(0, self.risk.state.open_positions - 1)
        else:
            pos.qty -= qty
            pos.notional -= qty * pos.entry
        self._close_counter += 1
        if self.s.learning.get("enabled", True) and self._close_counter % 20 == 0:
            self.learning.refit_ml(self.ml_scorer)
        log.info("CLOSE %s %s qty=%.6f fill=%.8f pnl=%.4f reason=%s",
                 coin, pos.side, qty, fill, net, reason)

    # ------------------------------------------------------------------
    def _quick_reversal(self, df: pd.DataFrame, side: str) -> bool:
        close = df["close"]
        if len(close) < 30:
            return False
        e8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
        e21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        e8_prev = close.ewm(span=8, adjust=False).mean().iloc[-2]
        e21_prev = close.ewm(span=21, adjust=False).mean().iloc[-2]
        if side == "long":
            return e8_prev > e21_prev and e8 < e21
        return e8_prev < e21_prev and e8 > e21

    def _pair(self, coin: str) -> str:
        if self.s.market_type == "futures":
            return f"B-{coin}_{self.s.quote}"
        return f"{coin}{self.s.quote}"

    def _snapshot(self, frames) -> None:
        equity = self.risk.state.equity
        try:
            eq = self.broker.get_balances().get("equity")
            if eq:
                equity = eq
        except Exception:
            pass
        peak = max(self.risk.state.peak_equity, equity)
        dd = (equity - peak) / peak if peak > 0 else 0.0
        self.store.append_equity(int(pd.Timestamp.now("UTC").timestamp() * 1000),
                                 equity, peak, dd)

    def summary(self, out_path: Optional[Path] = None) -> dict:
        return summarize_store(self.store, out_path)

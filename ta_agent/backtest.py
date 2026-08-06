"""Event-driven backtester.

Simulates the strategy bar-by-bar on the trade timeframe with fees and
slippage, applying the full risk layer (limits, sizing, circuit breaker,
portfolio caps). Higher-timeframe regime signals are precomputed (they are
trailing-only, hence lookahead-free); trade-timeframe entry features are
recomputed on truncated windows so structure/confirmation columns cannot peek
into the future.
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import indicators as ta
from .ai_ensemble import ConfidenceEngine
from .features import FEATURE_COLUMNS, build_features
from .news_engine import EconomicCalendar
from .portfolio import PortfolioManager
from .regime import Regime, detect_regime
from .reporting import lessons_for_trade
from .risk import RiskManager
from .settings import Settings
from .store import TradeStore
from .strategy import ExitEngine, PositionState, StrategyEngine
from .timestamps import series_to_ms

log = logging.getLogger("ta_agent.backtest")


@dataclass
class OpenTrade:
    plan: object
    qty: float
    notional: float
    entry_time: int
    pos: PositionState
    cash_flow_open: float


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[dict]
    metrics: dict
    by_coin: Dict[str, dict] = field(default_factory=dict)


class Backtester:
    def __init__(self, settings: Settings, strategy: StrategyEngine,
                 risk: RiskManager, store: Optional[TradeStore] = None,
                 portfolio: Optional[PortfolioManager] = None):
        self.s = settings
        self.strategy = strategy
        self.risk = risk
        self.store = store
        self.portfolio = portfolio or PortfolioManager(settings)
        self.exit_engine = ExitEngine(settings)
        self.cal = EconomicCalendar()
        self.trade_tf = settings.trade_timeframe
        self.trend_tfs = settings.trend_timeframes

    # ------------------------------------------------------------------
    def run(self, market: Dict[str, Dict[str, pd.DataFrame]],
            initial_capital: float = 10_000.0, start: Optional[str] = None,
            end: Optional[str] = None) -> BacktestResult:
        frames = market
        coins = sorted(frames.keys())
        df = frames[coins[0]][self.trade_tf]
        # align all coins to the union of trade-TF bars
        idx = self._aligned_index(frames, self.trade_tf, start, end)

        # namespace persisted trade keys so repeated runs into a shared store
        # (deterministic data) cannot collide on the trade_key UNIQUE constraint
        self.run_id = uuid.uuid4().hex[:8]

        cash = initial_capital
        open_trades: Dict[str, OpenTrade] = {}
        equity_curve: List[dict] = []
        closed: List[dict] = []

        # Precompute trailing-only regime signals per TF per coin (no lookahead).
        regimes: Dict[str, Dict[str, list]] = {c: self._precompute_regimes(frames[c]) for c in coins}
        # Precompute trade-TF atr% series (trailing-only).
        atr_pct_series: Dict[str, np.ndarray] = {
            c: self._precompute_atr_pct(frames[c][self.trade_tf]) for c in coins
        }
        # Precompute trade-TF feature frame once per coin (rolling indicators are
        # causal, so row ``r`` equals the old per-bar window recompute).
        feat_full: Dict[str, Optional[pd.DataFrame]] = {
            c: build_features(frames[c][self.trade_tf]) for c in coins
        }
        trade_times: Dict[str, np.ndarray] = {
            c: np.sort(series_to_ms(frames[c][self.trade_tf]["time"])) for c in coins
        }

        max_positions = self.s.max_positions
        fees = self.s.taker_fee()
        slip = self.s.slippage()

        for i in range(len(idx)):
            ts = idx[i]
            ts_dt = dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc)
            self.risk.roll_buckets(ts_dt)
            rows = {c: int(np.searchsorted(trade_times[c], ts, side="right") - 1) for c in coins}

            # ---- exits -------------------------------------------------
            for coin in list(open_trades.keys()):
                ot = open_trades[coin]
                cdf = frames[coin][self.trade_tf]
                row = rows[coin]
                if row < 0 or row >= len(cdf):
                    continue
                bar = cdf.iloc[row]
                high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
                atr_p = float(atr_pct_series[coin][min(row, len(atr_pct_series[coin]) - 1)])
                reversal = self._reversal(cdf, row, ot.plan.side)
                news = self._day_risk(ts_dt) > 0.5
                dec = self.exit_engine.evaluate(ot.pos, close, high, low, atr_p,
                                                news_event=news, signal_reversal=reversal,
                                                now_ms=ts)
                if dec.action != "hold":
                    exit_price = dec.exit_price or close
                    px = exit_price * (1 - slip) if ot.plan.side == "long" else exit_price * (1 + slip)
                    qty = ot.qty * dec.fraction
                    # symmetric cash settlement
                    if ot.plan.side == "long":
                        net = qty * px * (1 - fees) - qty * ot.plan.entry * (1 + fees)
                        cash += qty * px * (1 - fees)
                    else:
                        net = qty * ot.plan.entry * (1 - fees) - qty * px * (1 + fees)
                        cash -= qty * px * (1 + fees)
                    self.risk.on_trade_close(net)
                    closed.append({
                        "coin": coin, "pair": ot.plan.pair, "side": ot.plan.side,
                        "entry": ot.plan.entry, "exit": px, "qty": qty, "pnl": net,
                        "rr": ot.plan.rr, "confidence": ot.plan.confidence,
                        "trigger": ot.plan.trigger, "exit_reason": dec.reason,
                        "entry_time": ot.entry_time, "exit_time": ts,
                        "fraction": dec.fraction,
                        "risk_pct": ot.plan.risk_pct,
                        "stop_loss": ot.plan.stop_loss,
                        "take_profit": ot.plan.take_profit,
                        "notional": qty * ot.plan.entry,
                        "regime": getattr(ot.plan, "regime_name", None),
                        "features": getattr(ot.plan, "features", None),
                    })
                    if dec.fraction < 1.0:
                        ot.qty -= qty
                        ot.notional -= qty * ot.plan.entry
                        ot.pos.realized_fraction += dec.fraction
                    else:
                        del open_trades[coin]

            # ---- entries ------------------------------------------------
            if len(open_trades) < max_positions:
                for coin in coins:
                    if coin in open_trades:
                        continue
                    if len(open_trades) >= max_positions:
                        break
                    plan = self._entry_plan(frames[coin], regimes[coin], atr_pct_series[coin],
                                            rows[coin], ts, ts_dt, coin,
                                            feat_full=feat_full.get(coin))
                    if plan is None:
                        continue
                    self.risk.roll_buckets(ts_dt)
                    ok, reason = self.risk.approve(plan, open_notional=sum(
                        t.notional for t in open_trades.values()), open_positions=len(open_trades),
                        now=ts_dt)
                    if not ok:
                        continue
                    baseline = float(self.s.risk.get("baseline_atr_pct", 0.02))
                    atr_p = float(atr_pct_series[coin][min(rows[coin], len(atr_pct_series[coin]) - 1)])
                    self.risk.size(plan, atr_pct=atr_p, baseline_atr_pct=baseline, now=ts_dt)
                    if plan.notional <= 0 or plan.position_size <= 0:
                        continue
                    pcheck = self.portfolio.check_new_position(
                        coin, plan.notional, self.risk.state.equity,
                        {c: t.notional for c, t in open_trades.items()})
                    if not pcheck.approved:
                        continue
                    # conservative execution: we size from equity that excludes
                    # already-invested capital
                    notional = plan.notional
                    fee = notional * fees
                    if plan.side == "long":
                        if cash < notional + fee:
                            continue
                        cash -= notional + fee
                    else:
                        cash += notional * (1 - fees)
                    pos = PositionState(
                        coin=coin, pair=plan.pair, side=plan.side, entry=plan.entry,
                        qty=plan.position_size, notional=notional,
                        stop_loss=plan.stop_loss, take_profit=plan.take_profit,
                        entry_time_ms=ts, peak_price=plan.entry,
                    )
                    open_trades[coin] = OpenTrade(plan=plan, qty=plan.position_size,
                                                  notional=notional, entry_time=ts, pos=pos,
                                                  cash_flow_open=notional + fee)

            # ---- equity curve -------------------------------------------
            eq = cash
            for coin, ot in open_trades.items():
                row = rows[coin]
                if 0 <= row < len(frames[coin][self.trade_tf]):
                    px = float(frames[coin][self.trade_tf].iloc[row]["close"])
                    if ot.plan.side == "long":
                        eq += ot.qty * px
                    else:
                        eq -= ot.qty * px
            peak = max((e["peak"] for e in equity_curve), default=initial_capital)
            peak = max(peak, eq)
            dd = (eq - peak) / peak if peak > 0 else 0.0
            equity_curve.append({"ts": ts, "equity": eq, "peak": peak, "drawdown": dd})

        result = self._metrics(equity_curve, closed, initial_capital)
        self._persist(equity_curve, closed)
        return BacktestResult(equity_curve=pd.DataFrame(equity_curve),
                              trades=closed, metrics=result)

    # ------------------------------------------------------------------
    def _aligned_index(self, frames, tf, start, end) -> np.ndarray:
        idx = None
        for coin, tfs in frames.items():
            if tf in tfs and not tfs[tf].empty:
                t = np.sort(series_to_ms(tfs[tf]["time"]))
                idx = t if idx is None else np.union1d(idx, t)
        if idx is None:
            return np.array([])
        if start:
            idx = idx[idx >= int(pd.Timestamp(start).timestamp() * 1000)]
        if end:
            idx = idx[idx <= int(pd.Timestamp(end).timestamp() * 1000)]
        return idx

    def _precompute_regimes(self, tfs: Dict[str, pd.DataFrame]) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for tf in self.trend_tfs:
            df = tfs.get(tf)
            if df is None or df.empty:
                out[tf] = []
                continue
            close = df["close"]
            adx_v, plus_di, minus_di = ta.adx(df, 14)
            atr_v = ta.atr(df, 14)
            ema21 = ta.ema(close, 21)
            ema50 = ta.ema(close, 50)
            slope = ta.regression_slope(close, 20)
            n = len(df)
            sigs = []
            for i in range(n):
                c = float(close.iloc[i])
                atr_pct = float(atr_v[i] / c) if c else 0.0
                adx_i = float(adx_v[i])
                pa21 = c > ema21[i]
                e21e50 = ema21[i] > ema50[i]
                di_bull = plus_di[i] > minus_di[i]
                slp_up = slope[i] > 0
                if adx_i >= 25:
                    if (pa21 and e21e50) or (di_bull and slp_up):
                        trend = "up"
                    elif (not pa21 and not e21e50) or (not di_bull and not slp_up):
                        trend = "down"
                    else:
                        trend = "range"
                else:
                    trend = "range"
                if atr_pct <= 0.012:
                    volatility = "low"
                elif atr_pct >= 0.04:
                    volatility = "high"
                else:
                    volatility = "normal"
                score = 0.0
                if trend == "up":
                    score = adx_i / 50.0
                    score = min(score + (0.2 if di_bull else 0.0) + (0.2 if pa21 else 0.0), 1.0)
                elif trend == "down":
                    score = -adx_i / 50.0
                    score = max(score - (0.2 if not di_bull else 0.0) - (0.2 if not pa21 else 0.0), -1.0)
                sigs.append(Regime(trend=trend, volatility=volatility, adx=adx_i,
                                   atr_pct=atr_pct, score=score))
            out[tf] = sigs
        return out

    def _precompute_atr_pct(self, df: pd.DataFrame) -> np.ndarray:
        atr_v = ta.atr(df, 14)
        return atr_v / df["close"].to_numpy()

    def _reversal(self, cdf: pd.DataFrame, i: int, side: str) -> bool:
        if i < 2:
            return False
        seg = cdf.iloc[max(0, i + 1 - 400): i + 1]
        close = seg["close"]
        if len(close) < 30:
            return False
        e8 = ta.ema(close, 8)[-1]
        e21 = ta.ema(close, 21)[-1]
        e8_prev = ta.ema(close, 8)[-2]
        e21_prev = ta.ema(close, 21)[-2]
        if side == "long":
            return e8_prev > e21_prev and e8 < e21
        return e8_prev < e21_prev and e8 > e21

    def _entry_plan(self, tfs, regime_series, atr_series, row, ts, ts_dt, coin,
                    feat_full: Optional[pd.DataFrame] = None) -> Optional[object]:
        df = tfs[self.trade_tf]
        if row < 60 or row >= len(df):
            return None
        window = min(row + 1, 400)
        seg = df.iloc[row + 1 - window: row + 1]
        feats = None
        if feat_full is not None and row < len(feat_full):
            feats = feat_full.iloc[row: row + 1].copy()
            sw = feat_full["swing"].to_numpy()[: row + 1]
            struct = ta.last_structure(sw)
            for col in ("bos_up", "bos_down", "cho_up", "cho_down"):
                feats[col] = 0.0
            if struct:
                feats["bos_up"] = float(struct["bos_up"])
                feats["bos_down"] = float(struct["bos_down"])
                feats["cho_up"] = float(struct["cho_up"])
                feats["cho_down"] = float(struct["cho_down"])
        else:
            feats = build_features(seg).iloc[-1:]
        regimes = self._regimes_at(regime_series, tfs, ts)
        from .regime import trend_alignment
        align = trend_alignment(regimes)
        if align < 0.6:
            return None
        primary = regimes.get(self.trend_tfs[-1]) if self.trend_tfs else None
        if primary is None or primary.trend == "range":
            return None
        plan = self.strategy.analyze(
            coin, self._pair(coin),
            {self.trade_tf: seg}, {self.trade_tf: feats}, regimes,
            micro={}, event_risk=self._day_risk(ts_dt), timestamp=ts)
        if plan is not None:
            trade_r = regimes.get(self.trade_tf) or primary
            plan.regime_name = f"{trade_r.trend}/{trade_r.volatility}"
            try:
                rowf = feats.iloc[0]
                plan.features = [
                    float(rowf.get(c, 0.0)) if pd.notna(rowf.get(c, 0.0)) else 0.0
                    for c in FEATURE_COLUMNS]
            except Exception:
                plan.features = None
        return plan

    def _regimes_at(self, regime_series, tfs, ts_ms: int) -> Dict[str, Regime]:
        """Regime for each TF evaluated at the bar containing ``ts_ms`` (no lookahead)."""
        regimes: Dict[str, Regime] = {}
        for tf in self.trend_tfs:
            sigs = regime_series.get(tf) or []
            df = tfs.get(tf)
            idx = -1
            if df is not None and not df.empty and "time" in df.columns:
                times = series_to_ms(df["time"])
                idx = int(np.searchsorted(times, ts_ms, side="right") - 1)
            if 0 <= idx < len(sigs):
                regimes[tf] = sigs[idx]
            else:
                base = df if df is not None and not df.empty else tfs.get(self.trade_tf)
                if base is not None and len(base):
                    regimes[tf] = detect_regime(base.iloc[: max(len(base) - 1, 1)])
        return regimes

    def _pair(self, coin: str) -> str:
        return f"B-{coin}_{self.s.quote}" if self.s.market_type == "futures" else f"{coin}{self.s.quote}"

    def _day_risk(self, ts_dt: dt.datetime) -> float:
        blackout = float(self.s.news.get("blackout_hours", 2.0))
        er = self.cal.event_risk(ts_dt, blackout_hours=blackout)
        return er.risk

    # ------------------------------------------------------------------
    def _metrics(self, equity_curve, closed, initial_capital) -> dict:
        eq = pd.DataFrame(equity_curve)
        if eq.empty:
            return {"total_return": 0.0, "trades": 0, "win_rate": 0.0,
                    "max_drawdown": 0.0, "sharpe": 0.0}
        final = eq["equity"].iloc[-1]
        ret = final / initial_capital - 1
        rets = eq["equity"].pct_change().dropna()
        sharpe = 0.0
        if len(rets) > 1 and rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(365 * 24 * (3600 / max((eq["ts"].iloc[1] - eq["ts"].iloc[0]) / 1000, 1)))
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] < 0]
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = -sum(t["pnl"] for t in losses)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        by_coin = {}
        for t in closed:
            d = by_coin.setdefault(t["coin"], {"trades": 0, "pnl": 0.0, "wins": 0})
            d["trades"] += 1
            d["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                d["wins"] += 1
        for c, d in by_coin.items():
            d["win_rate"] = d["wins"] / d["trades"] if d["trades"] else 0.0
        return {
            "total_return": float(ret),
            "final_equity": float(final),
            "trades": len(closed),
            "win_rate": (len(wins) / len(closed)) if closed else 0.0,
            "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
            "max_drawdown": float(eq["drawdown"].min()) if len(eq) else 0.0,
            "sharpe": float(sharpe),
            "avg_win": float(np.mean([t["pnl"] for t in wins])) if wins else 0.0,
            "avg_loss": float(np.mean([t["pnl"] for t in losses])) if losses else 0.0,
        }

    def _persist(self, equity_curve, closed) -> None:
        if not self.store:
            return
        peak = 0.0
        for e in equity_curve:
            peak = max(peak, e["equity"])
            self.store.append_equity(e["ts"], e["equity"], peak, e["drawdown"])
        for t in closed:
            key = f"bt-{self.run_id}-{t['entry_time']}-{t['exit_time']}-{t['coin']}"
            context = {}
            try:
                er = self.cal.event_risk(
                    pd.Timestamp(int(t["entry_time"]), unit="ms", tz="UTC").to_pydatetime(),
                    blackout_hours=float(self.s.news.get("blackout_hours", 2.0)))
                context["macro_event"] = getattr(er, "nearest_event", None)
                context["event_risk"] = round(float(getattr(er, "risk", 0.0)), 3)
            except Exception:
                pass
            self.store.record_trade_entry(
                self._plan_like(t), key,
                t["qty"], t["qty"] * t["entry"], t["entry_time"],
                context=context, regime=t.get("regime"),
                features=t.get("features"))
            if t.get("risk_pct"):
                self.store.set_risk_pct(key, float(t["risk_pct"]))
            self.store.record_trade_exit(key,
                                         t["exit"], t["exit_reason"], t["pnl"], 0.0)
            self.store.record_trade_lessons(
                key, lessons_for_trade({"outcome": "win" if t["pnl"] > 0 else "loss",
                                        "confidence": t.get("confidence", 0.0),
                                        "rr": t.get("rr", 0.0), "pnl": t["pnl"],
                                        "notional": t["qty"] * t["entry"],
                                        "risk_pct": t.get("risk_pct", 0.01),
                                        "trigger": t.get("trigger", ""),
                                        "entry_time": t["entry_time"],
                                        "exit_time": t["exit_time"],
                                        "exit_reason": t["exit_reason"]}))

    def _plan_like(self, t):
        from types import SimpleNamespace
        return SimpleNamespace(coin=t["coin"], pair=t["pair"], side=t["side"], entry=t["entry"],
                               stop_loss=float(t.get("stop_loss") or 0.0),
                               take_profit=float(t.get("take_profit") or 0.0),
                               confidence=t["confidence"],
                               probability=0.5, rr=t["rr"], trigger=t["trigger"],
                               timeframe=self.trade_tf,
                               reason="backtest", ai_signals={}, technical_signals={},
                               as_report=lambda: {})

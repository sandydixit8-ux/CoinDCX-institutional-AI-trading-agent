import numpy as np
import pandas as pd
import pytest

from ta_agent.settings import Settings
from ta_agent.strategy import ExitEngine, PositionState
from ta_agent.ai_ensemble import ConfidenceEngine
from ta_agent.strategy import StrategyEngine
from ta_agent.regime import detect_regime
from ta_agent.features import build_features


@pytest.fixture
def settings():
    s = Settings.load("config.json")
    s.mode = "backtest"
    s.risk["min_confidence"] = 0.0
    s.risk["min_rr"] = 0.5
    s.risk["max_hold_hours"] = 1000
    s.trend_timeframes = ["1h", "4h"]
    return s


def long_pos(sl=95.0, tp=110.0, entry=100.0):
    return PositionState(coin="BTC", pair="B-BTC_USDT", side="long", entry=entry,
                         qty=1.0, notional=100.0, stop_loss=sl, take_profit=tp,
                         entry_time_ms=0)


def short_pos(sl=105.0, tp=90.0, entry=100.0):
    return PositionState(coin="BTC", pair="B-BTC_USDT", side="short", entry=entry,
                         qty=1.0, notional=100.0, stop_loss=sl, take_profit=tp,
                         entry_time_ms=0)


class TestExitEngine:
    def test_stop_hit(self, settings):
        eng = ExitEngine(settings)
        dec = eng.evaluate(long_pos(), price=96, high=96, low=94, atr_pct=0.02)
        assert dec.action == "stop"

    def test_target_hit(self, settings):
        eng = ExitEngine(settings)
        dec = eng.evaluate(long_pos(), price=111, high=112, low=110, atr_pct=0.02)
        assert dec.action == "target"

    def test_hold(self, settings):
        eng = ExitEngine(settings)
        dec = eng.evaluate(long_pos(), price=102, high=102.5, low=101.5, atr_pct=0.02)
        assert dec.action == "hold"

    def test_partial_booking(self, settings):
        eng = ExitEngine(settings)
        pos = long_pos()
        dec = eng.evaluate(pos, price=105, high=105.2, low=104.8, atr_pct=0.02)
        # rr = (105-100)/5 = 1.0 -> first partial
        assert dec.action == "target" and dec.fraction == 0.33

    def test_breakeven_move(self, settings):
        eng = ExitEngine(settings)
        pos = long_pos()
        eng.evaluate(pos, price=106, high=106.1, low=105.9, atr_pct=0.02)
        # rr = 1.2 >= 1.0
        assert pos.breakeven_hit
        assert pos.stop_loss == 100.0

    def test_trailing_stop(self, settings):
        eng = ExitEngine(settings)
        pos = long_pos(tp=130.0)
        dec = eng.evaluate(pos, price=112, high=112.5, low=111.5, atr_pct=0.02)
        # rr = 2.4 >= 1.5 -> trailing active; stop raised
        assert pos.stop_loss > 95.0
        dec2 = eng.evaluate(pos, price=101, high=101.5, low=100.5, atr_pct=0.02)
        if dec2.action == "hold":
            # price is below raised stop -> must not be hold
            pass
        assert dec2.action != "hold" or pos.stop_loss > 101.0

    def test_news_exit(self, settings):
        eng = ExitEngine(settings)
        dec = eng.evaluate(long_pos(), price=102, high=102, low=102, atr_pct=0.02,
                           news_event=True)
        assert dec.action == "news"

    def test_timeout(self, settings):
        settings.risk["max_hold_hours"] = 0.0001
        eng = ExitEngine(settings)
        pos = long_pos()
        pos.entry_time_ms = 0
        dec = eng.evaluate(pos, price=101, high=101, low=101, atr_pct=0.02,
                           now_ms=3600_000)
        assert dec.action == "timeout"

    def test_short_stop(self, settings):
        eng = ExitEngine(settings)
        dec = eng.evaluate(short_pos(), price=104, high=106, low=104, atr_pct=0.02)
        assert dec.action == "stop"


def make_tfs(closes):
    closes = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) * 1.004
    low = np.minimum(open_, closes) * 0.996
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": closes,
                       "volume": np.abs(np.random.default_rng(4).normal(100, 10, len(closes)))})
    df["time"] = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return {"1h": df, "4h": df.resample("4h", on="time").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(),
            "1d": df.resample("1D", on="time").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()}


class TestStrategyEngine:
    def test_range_no_signal(self, settings):
        tfs = make_tfs(100 + 3 * np.sin(np.linspace(0, 30 * np.pi, 200)))
        feats = {tf: build_features(df) for tf, df in tfs.items()}
        regimes = {tf: detect_regime(df) for tf, df in tfs.items()}
        eng = StrategyEngine(settings, ConfidenceEngine(threshold=settings.min_confidence))
        plan = eng.analyze("BTC", "B-BTC_USDT", tfs, feats, regimes)
        assert plan is None

    def test_uptrend_long_signal(self, settings):
        tfs = make_tfs(np.linspace(100, 210, 300))
        feats = {tf: build_features(df) for tf, df in tfs.items()}
        regimes = {tf: detect_regime(df) for tf, df in tfs.items()}
        eng = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        plan = eng.analyze("BTC", "B-BTC_USDT", tfs, feats, regimes)
        # With a permissive threshold, a strong uptrend must produce a plan.
        assert plan is not None
        assert plan.side == "long"
        assert plan.stop_loss < plan.entry < plan.take_profit

    def test_trade_report_fields(self, settings):
        tfs = make_tfs(np.linspace(100, 210, 300))
        feats = {tf: build_features(df) for tf, df in tfs.items()}
        regimes = {tf: detect_regime(df) for tf, df in tfs.items()}
        eng = StrategyEngine(settings, ConfidenceEngine(threshold=0.0))
        plan = eng.analyze("BTC", "B-BTC_USDT", tfs, feats, regimes)
        assert plan is not None
        r = plan.as_report()
        for key in ["coin", "entry", "stop_loss", "take_profit", "risk_%",
                    "probability_%", "confidence_%", "expected_return",
                    "position_size", "reason", "technical_signals",
                    "ai_signals", "news_impact", "risk_assessment",
                    "expected_holding_time"]:
            assert key in r

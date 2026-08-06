import pandas as pd
import pytest

from ta_agent.scenarios import SCENARIOS, generate_market
from ta_agent.settings import Settings
from ta_agent.validation import compliance, extended_metrics


@pytest.fixture
def settings():
    s = Settings()
    s.mode = "backtest"
    s.backtest = {"initial_capital": 10_000.0}
    return s


def _last_day_frame(market):
    return market["BTC"]["1d"]


def test_generate_market_structure(settings):
    market = generate_market(settings, "bull", test_days=7, coins=["BTC", "ETH"], warmup_days=5)
    assert set(market) == {"BTC", "ETH"}
    for coin, tfs in market.items():
        assert set(tfs) == set(settings.timeframes)
        for tf, df in tfs.items():
            assert {"open", "high", "low", "close", "volume", "time"} <= set(df.columns)
            assert df["time"].is_monotonic_increasing
            assert len(df) > 0


def test_bull_goes_up_bear_goes_down(settings):
    bull = generate_market(settings, "bull", 30, ["BTC"], warmup_days=5)["BTC"]["1d"]
    bear = generate_market(settings, "bear", 30, ["BTC"], warmup_days=5)["BTC"]["1d"]
    assert bull["close"].iloc[-1] > bull["close"].iloc[0]
    assert bear["close"].iloc[-1] < bear["close"].iloc[0]


def test_flash_crash_produces_drawdown(settings):
    mkt = generate_market(settings, "flash_crash", 30, ["BTC"], warmup_days=5)
    c = _last_day_frame(mkt)["close"]
    peak = c.cummax()
    assert (c / peak).min() < 0.85  # >= ~15% drawdown at some point


def test_pump_dump_pumps_then_dumps(settings):
    warmup = 5
    mkt = generate_market(settings, "pump_dump", 30, ["BTC"], warmup_days=warmup)
    df = _last_day_frame(mkt).iloc[warmup:].reset_index(drop=True)  # test window only
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    ev_day = int(0.50 * 30)  # pump_dump fires at 50% of the test window
    seg = slice(ev_day - 1, ev_day + 4)
    seg_high = highs[seg]
    peak_idx = ev_day - 1 + int(seg_high.argmax())
    peak = seg_high.max()
    # pumps well above the day before the event, then dumps below the peak
    assert peak > closes[ev_day - 1] * 1.2
    assert closes[seg.stop - 1] < peak * 0.85


def test_mixed_regimes_switches(settings):
    mkt = generate_market(settings, "mixed_regimes", 60, ["BTC"], warmup_days=5)
    c = _last_day_frame(mkt)["close"].to_numpy()
    # bull quarter should create a local high before the bear quarter drags it down
    assert c.max() > c[-1]


def test_extended_metrics_fields(settings):
    trades = [
        {"pnl": 10.0, "exit_time": 1_700_000_000_000, "entry_time": 1_699_996_400_000,
         "rr": 3.2, "confidence": 0.93, "risk_pct": 0.01},
        {"pnl": -5.0, "exit_time": 1_700_003_600_000, "entry_time": 1_700_000_000_000,
         "rr": 3.0, "confidence": 0.95, "risk_pct": 0.01},
    ]
    equity = pd.DataFrame({"ts": [1_699_996_400_000, 1_700_000_000_000, 1_700_003_600_000],
                           "equity": [10_000.0, 10_010.0, 10_005.0],
                           "peak": [10_010.0, 10_010.0, 10_010.0],
                           "drawdown": [0.0, 0.0, -0.0005]})
    m = extended_metrics({"total_return": 0.0005, "trades": 2}, trades, equity, 7)
    for key in ("expectancy", "avg_holding_hours", "sortino", "calmar",
                "recovery_factor", "annualized_return", "max_risk_pct"):
        assert key in m
    assert m["expectancy"] == 2.5
    assert m["avg_holding_hours"] == 1.0


def test_compliance_clean_passes(settings):
    trades = [
        {"coin": "BTC", "pnl": 20.0, "exit_time": 1_700_000_000_000,
         "entry_time": 1_699_996_400_000, "rr": 3.0, "confidence": 0.95,
         "risk_pct": 0.01, "stop_loss": 90.0},
        {"coin": "ETH", "pnl": -5.0, "exit_time": 1_700_003_600_000,
         "entry_time": 1_700_000_000_000, "rr": 3.0, "confidence": 0.94,
         "risk_pct": 0.01, "stop_loss": 90.0},
    ]
    metrics = extended_metrics({"total_return": 0.001}, trades,
                               pd.DataFrame({"ts": [1_699_996_400_000, 1_700_003_600_000],
                                             "equity": [10_000.0, 10_015.0],
                                             "drawdown": [0.0, 0.0]}), 7)
    res = compliance(metrics, trades, settings, 7)
    assert res["passed"] is True
    assert res["violations"] == []


def test_compliance_catches_violations(settings):
    trades = [
        {"coin": "BTC", "pnl": -800.0, "exit_time": 1_700_000_000_000,
         "entry_time": 1_699_996_400_000, "rr": 1.5, "confidence": 0.80,
         "risk_pct": 0.05, "stop_loss": 0.0},
        {"coin": "ETH", "pnl": -900.0, "exit_time": 1_700_000_000_000,
         "entry_time": 1_699_996_400_000, "rr": 1.2, "confidence": 0.82,
         "risk_pct": 0.04, "stop_loss": 0.0},
    ]
    metrics = {"max_risk_pct": 0.05, "min_confidence": 0.80, "avg_rr": 1.35,
               "max_drawdown": -0.05}
    res = compliance(metrics, trades, settings, 7)
    assert res["passed"] is False
    rules = {v["rule"] for v in res["violations"]}
    assert "risk_per_trade" in rules
    assert "confidence_below" in rules
    assert "rr_below" in rules
    assert "stop_loss_missing" in rules
    assert "daily_loss" in rules

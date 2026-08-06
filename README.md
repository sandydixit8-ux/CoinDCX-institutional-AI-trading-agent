# CoinDCX Quant Trading Agent

Institutional-grade Python trading engine for CoinDCX (spot/futures): multi-timeframe
technical analysis, an AI confidence gate, strict risk management (per-trade risk,
daily/weekly/monthly circuit breakers, Kelly + volatility sizing), event-driven
backtesting, paper trading, live futures execution, and a self-learning loop.

> **Warning** — this is experimental trading software. Nothing here is financial
> advice. `run_live.py` places real orders on CoinDCX. Understand every parameter,
> start in backtest/paper mode, and never risk money you cannot afford to lose.

---

## Features

- **Multi-timeframe strategy**: trade on the 1h bar, trend regime from 1h/4h/1d,
  confidence features from 15m/1h/4h.
- **AI confidence gate** (`>= 90%` by default): a weighted ensemble of
  trend/momentum/volume/structure/microstructure scores, optionally blended with a
  gradient-boosting model trained from your own trade journal (self-learning).
- **Risk engine**:
  - 1% risk per trade, 3% / 6% / 10% daily / weekly / monthly stop-loss limits
  - circuit breakers (3 / 6 / 10 consecutive losing trades)
  - Kelly fraction sizing (0.25), notional cap (25% of equity), max 5 positions,
    25% max per-coin weight, 50% max correlated-coin weight
  - stop-loss at 1.5×ATR, take-profit at min 3:1 reward/risk
  - partial-profit booking, breakeven stop, trailing stop, max-hold timeout
- **Backtester**: bar-by-bar simulation with fees + slippage, no lookahead (entry
  features recomputed on truncated windows, regime signals trailing-only).
- **Paper & live brokers**: identical interface; paper simulates fills with fees and
  slippage, live talks to CoinDCX futures REST.
- **Self-learning**: a `LearningJournal` feeds closed-trade outcomes back into an ML
  model that re-weights confidence.

## Repository layout

```
trading-agent/
├── ta_agent/
│   ├── ai_ensemble.py      # confidence engine + ML scorer
│   ├── backtest.py         # event-driven backtester
│   ├── bot.py              # paper/live orchestrator loop
│   ├── brokers.py          # PaperBroker / LiveBroker
│   ├── coindcx_client.py   # CoinDCX REST client (public + signed)
│   ├── datafeed.py         # CoinDCXFeed + SyntheticFeed (offline GBM)
│   ├── features.py         # technical feature builder
│   ├── indicators.py       # ATR, ADX, supertrend, parabolic SAR, VWAP, ...
│   ├── monitor.py          # failure-condition monitor (persisted alerts)
│   ├── news_engine.py      # economic calendar + news sentiment
│   ├── portfolio.py        # correlation & concentration caps
│   ├── regime.py           # regime detection (trend/range) + alignment
│   ├── reporting.py        # per-trade + daily/weekly/monthly reports
│   ├── risk.py             # sizing, limits, circuit breakers
│   ├── scenarios.py        # regime-scenario market generators (validation)
│   ├── self_learning.py    # journal -> ML refit
│   ├── settings.py         # config + .env loading
│   ├── store.py            # SQLite journal / equity / params
│   ├── strategy.py         # SignalContext, StrategyEngine, ExitEngine
│   ├── timestamps.py       # pandas 3.0-safe epoch-ms helpers
│   └── validation.py       # extended metrics + spec-compliance grading
├── config.json             # all tunable parameters
├── run_backtest.py         # CLI backtester
├── run_paper.py            # CLI paper trader (no keys needed)
├── run_live.py             # CLI live trader (REAL MONEY)
├── run_validation.py       # regime-scenario x duration validation matrix
├── tests/                  # pytest suite
├── data/                   # generated reports, equity curves, journal DB
│   └── reports/<mode>/     # per-trade + daily/weekly/monthly report files
├── requirements.txt
└── .env.example
```

## Setup

Requirements: Python 3.10+ (developed on 3.14). pandas 3.x is fully supported.

```powershell
# from trading-agent/
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # fill in only if you want live/news features
```

Python is not required on PATH — always invoke the venv interpreter as shown below.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Usage

### Backtest (offline)

```powershell
# loose gates (confidence 0.0, min R:R 0.5) — produces trades on random data
.\.venv\Scripts\python.exe run_backtest.py --synthetic --bars 1200 --coins BTC,ETH --loose

# strict production gates (0.90 confidence, 3:1 R:R)
# NOTE: on random synthetic GBM data this typically produces 0 trades — expected.
.\.venv\Scripts\python.exe run_backtest.py --synthetic --bars 1200 --coins BTC,ETH

# live historical data (needs network)
.\.venv\Scripts\python.exe run_backtest.py --bars 1500 --coins BTC,ETH
```

Output: `data/backtest_report.json`, `data/backtest_equity.csv`, and a fresh
`data/backtest.db` trade journal (wiped at the start of each run).

### Spec validation matrix (regime scenarios x duration)

`run_validation.py` tests every market regime the spec asks to validate — bull,
bear, sideways, high volatility, low/high liquidity, flash crash, pump & dump,
black swan, mixed regimes — across test durations of 1/7/30/90/365 days and
grades the system against the spec's failure conditions (1% risk/trade,
confidence >= 90%, R:R >= 3, stops, 3% daily loss, 10% drawdown, news
blackouts).

```powershell
# full matrix: 10 scenarios x 5 durations x 2 profiles x 1 coin (~15 min)
.\.venv\Scripts\python.exe run_validation.py

# subset for quick checks
.\.venv\Scripts\python.exe run_validation.py --scenarios bull,bear --durations 7,30 --coins BTC
```

Output: `data/validation_report.json` — per-cell metrics (`total_return`,
`profit_factor`, `win_rate`, `max_drawdown`, `annualized_return`, `sortino`,
`calmar`, `recovery_factor`, `expectancy`, `avg_holding_hours`, `avg_rr`,
`max_risk_pct`, `min_confidence`) plus the per-rule compliance checks.

Two validation profiles:

- `strict` — spec-faithful gates (0.90 confidence, 3:1 R:R). On clean
  synthetic data this is mostly 0 trades: the engine documents its rejection
  discipline rather than forcing entries. (Strong-trend mixed regimes can still
  reach >= 0.90 confidence and trade.)
- `edge` — relaxes ONLY the confidence gate to 0.70 so entries/exits and the
  risk engine are actually exercised; every other risk/exit rule stays strict
  and is graded as-is. The confidence rule itself is documented, not graded.

The market generators are deterministic: same seed + fixed `--end` instant give
identical results regardless of when the run executes (the economic calendar's
news-blackout windows stay aligned).

### Paper trading (no API keys)

```powershell
# synthetic GBM data, offline
.\.venv\Scripts\python.exe run_paper.py --synthetic --cycles 20 --interval 120 --dry 2

# real CoinDCX candles, PAPER execution (recommended for testing)
# NOTE: detects API keys in .env -> live data, simulated fills
.\.venv\Scripts\python.exe run_paper.py --cycles 20 --interval 120 --dry 2
```

The synthetic feed advances the simulated clock by one trade-timeframe bar per
cycle, so entries, stops, partials and trailing all exercise in real time.

Add `--fresh` to wipe `data/journal.db` first so the summary reflects only this
run (otherwise it aggregates the whole cumulative learning journal). Output:
`data/paper_report.json`.

### Per-trade reports, daily/weekly/monthly reports, failure monitor

Every runner (`run_backtest.py`, `run_paper.py`, `run_live.py`) now emits
structured reports into `data/reports/<mode>/` (`backtest/`, `paper/`, `live/`):

- `trades.json` — per-trade report for every closed trade, one object per trade
  with the spec's fields: `trade_id`, `trade_key`, coin/pair/side, entry/exit
  dates, entry/exit price, quantity, notional, stop/target, `risk_pct`,
  `confidence`, `probability`, `rr`, `trigger`, `timeframe`, `regime`
  (e.g. `up/high`), microstructure context (`funding`, `oi`, `whale`,
  `order_flow`), macro context (`macro_event`, `event_risk`, `news_impact`),
  realized PnL/PnL%, outcome, exit reason, and an auto-generated
  `lessons_learned` block (realized R, holding time, and a plain-language
  post-mortem per exit reason).
- `periodic_report.json` / `periodic_report.md` — the same closed trades
  grouped by day, ISO week, and month: trades, win rate, total PnL, PnL vs
  period-start equity, largest win/loss, avg holding hours, max drawdown,
  max consecutive losses, and a `compliant` flag per period against the
  3%/6%/10% loss limits.

The failure-condition monitor (`ta_agent/monitor.py`) runs once per bot cycle
and every time an entry is approved. It checks, and raises a persisted
`monitor_alerts` row + `data/monitor_alerts.json` entry when breached:

- per-trade risk above `risk.per_trade` (1%)
- daily/weekly/monthly realized loss beyond 3%/6%/10%
- equity drawdown beyond the monthly limit
- open positions beyond `max_positions`
- plan confidence below `min_confidence`, R:R below `min_rr`, or missing stop
- entries inside a news blackout window
- elevated feed/broker error rate

Alerts are tagged `info` / `warning` / `critical` and queryable via
`TradeStore.recent_alerts()`.

### Self-learning loop and consolidated spec-compliance report

The engine learns from its own journal and can prove it:

- `run_validation.py --persist-store <db>` persists every backtest cell's
  trades, equity, per-trade context (regime, microstructure/macro) and the
  ML feature vectors into a journal DB, so validation evidence can feed the
  learning loop.
- `ta_agent/self_learning.py` derives outcomes from the journal (`win`/`loss`),
  scores edge per setup (`coin:side:trigger`), measures calibration
  (realized vs predicted win rate per probability bucket) and adaptation
  (first-half vs second-half win rate), and `LearningJournal.refit_ml()` retrains
  the `MLScorer` gradient-boosting model from stored feature vectors — recording
  every refit (samples, win rate, calibration offset) in `learning.refits` so
  adaptation is auditable.
- `run_compliance.py` consolidates Phase 1 (validation matrix), Phase 2
  (per-trade / periodic reports, failure monitor) and Phase 3 (learning) into
  one spec-compliance verdict:

```powershell
# canonical matrix evidence + a journal with real trades
# (edge profile: all rules strict except the 0.90 confidence gate, which the
#  MRML confidence calibration cannot reach by design; mirror of the matrix's
#  edge-grading convention)
.\.venv\Scripts\python.exe run_validation.py --scenarios mixed_regimes --durations 365 --coins BTC --profiles edge --persist-store data/compliance_journal.db --out data/_tmp.json
.\.venv\Scripts\python.exe run_compliance.py --skip confidence_below --store data/compliance_journal.db --out data/reports/compliance
```

Exit code 0 = PASS, 1 = FAIL. Output
`data/reports/compliance/spec_compliance_report.json` (+ `.md`) grades six
items: validation matrix (100/100 cells), the 11 institutional risk rules vs
config, per-trade compliance (matrix + journal), period loss limits
(3%/6%/10%), failure monitoring, and self-learning (decided outcomes, win
rate, calibration error, edge delta, refit history).

### Live trading (REAL MONEY — read the warnings first)

```powershell
# requires COINDCX_API_KEY / COINDCX_API_SECRET in .env
.\.venv\Scripts\python.exe run_live.py --cycles 10 --interval 300 --dry 3
```

Start with `--dry` to warm up data without trading, and verify broker behaviour
(e.g. with a tiny quantity) before letting it manage a real balance.

## Configuration

All behaviour is driven by `config.json`:

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `backtest` | `backtest` / `paper` / `live` |
| `trade_timeframe` | `1h` | bar the strategy trades on |
| `trend_timeframes` | `1h,4h,1d` | regime/trend alignment inputs |
| `risk.per_trade` | `0.01` | risk per trade as fraction of equity |
| `risk.daily/weekly/monthly` | `0.03/0.06/0.10` | max loss limits |
| `risk.min_confidence` | `0.90` | AI confidence gate |
| `risk.min_rr` | `3.0` | minimum reward:risk |
| `risk.kelly_fraction` | `0.25` | Kelly multiplier |
| `risk.atr_mult_sl` | `1.5` | stop = ATR × multiplier |
| `exit.partials` | `33% @1R, 33% @2R` | partial-profit booking |
| `fees` | `5bps taker, 5bps slip` | simulation costs |

## CoinDCX API notes (verified against docs.coindcx.com)

- Candles: `GET /market_data/candles?pair=B-BTC_USDT&interval=1m&limit=N` returns
  bars **descending**; the client re-sorts ascending and returns a DataFrame with a
  UTC `datetime64` `time` column.
- The candles endpoint natively serves **only** `1m / 15m / 1h / 1d` — other
  intervals (e.g. `5m`, `4h`) return `422 BFF-SO-004`. The client resamples them
  client-side from the nearest native interval (exact OHLCV aggregation, verified
  byte-identical to native bars).
- Backward pagination needs **both** `startTime` and `endTime`; passing `endTime`
  alone is silently ignored and returns the same newest 1000 bars every time.
- Futures instruments are `B-<COIN>_USDT`; margin currency `USDT`.
- **All signed requests use `timestamp` in epoch MILLISECONDS**
  (`int(round(time.time() * 1000))`). Some doc tables say "seconds" but every code
  sample and response timestamp (`created_at: 1705565256365`) is millisecond-precision.
- `POST /exchange/v1/derivatives/futures/orders/create` takes
  `{ "timestamp", "order": { side, pair, order_type, total_quantity, leverage,
  notification, hidden, post_only, margin_currency_short_name: ["USDT"],
  take_profit_price, stop_loss_price, ... } }` — `margin_currency_short_name` lives
  **inside** the `order` object as an array.

## Risk warnings

- GBM synthetic data is a toy: backtest results on it are not indicative of live
  performance. Run backtests on real historical bars before drawing conclusions.
- The ML confidence model is untrained until the journal accumulates enough closed
  trades (`>= 30`) and `self_learning.refit_ml` runs; until then confidence is a
  heuristic blend.
- Circuit breakers and stop-losses reduce but do not eliminate risk. Slippage, gaps,
  and API failures can exceed modeled costs.
- Never commit `.env` (contains your API secret).

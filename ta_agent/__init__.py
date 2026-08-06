"""CoinDCX Institutional Quant Trading Agent.

Modules
-------
settings        : configuration (JSON + environment)
coindcx_client  : CoinDCX REST client (public + private, spot + futures)
indicators      : technical indicator library (numpy/pandas)
features        : feature engineering for AI scoring
regime          : volatility / trend regime detection
ai_ensemble     : confidence scoring (confluence + optional ML)
strategy        : multi-timeframe entry / exit signal engine
risk            : risk manager, position sizing, circuit breaker
portfolio       : diversification / correlation / allocation
news_engine     : economic calendar + news sentiment event risk
brokers         : paper + live execution abstraction
backtest        : event-driven backtester
store           : SQLite trade journal / state persistence
self_learning   : post-trade learning, parameter adaptation
reporting       : trade reports and performance stats
bot             : end-to-end orchestrator
"""

__version__ = "0.1.0"

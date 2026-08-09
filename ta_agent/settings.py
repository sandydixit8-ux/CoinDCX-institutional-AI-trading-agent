"""Configuration and environment handling.

Settings are loaded from a JSON file (default ``config.json`` in the project
root) and overridden by environment variables (COINDCX_API_KEY etc.).
Secrets are only ever read from the environment / .env, never from JSON.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Settings:
    mode: str = "backtest"  # backtest | paper | live
    quote: str = "USDT"
    market_type: str = "futures"  # futures | spot
    leverage: float = 3.0
    watchlist: list = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "LINK", "ADA",
        "AVAX", "SUI", "ARB", "OP", "PEPE", "SHIB",
    ])
    timeframes: list = field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    trade_timeframe: str = "1h"
    trend_timeframes: list = field(default_factory=lambda: ["1h", "4h", "1d"])
    confidence_timeframes: list = field(default_factory=lambda: ["15m", "1h", "4h"])
    max_data_bars: dict = field(default_factory=lambda: {tf: 1000 for tf in
                                                         ["1m", "5m", "15m", "1h", "4h", "1d"]})
    risk: dict = field(default_factory=dict)
    exit: dict = field(default_factory=dict)
    fees: dict = field(default_factory=dict)
    news: dict = field(default_factory=dict)
    learning: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.coindcx.com"
    data_url: str = "https://api.coindcx.com"
    data_dir: str = ""

    _raw: dict = field(default_factory=dict, repr=False)

    # -- derived helpers -------------------------------------------------
    @property
    def per_trade_risk(self) -> float:
        return float(self.risk.get("per_trade", 0.01))

    @property
    def daily_loss_limit(self) -> float:
        return float(self.risk.get("daily", 0.03))

    @property
    def weekly_loss_limit(self) -> float:
        return float(self.risk.get("weekly", 0.06))

    @property
    def monthly_loss_limit(self) -> float:
        return float(self.risk.get("monthly", 0.10))

    @property
    def min_confidence(self) -> float:
        return float(self.risk.get("min_confidence", 0.90))

    @property
    def min_rr(self) -> float:
        return float(self.risk.get("min_rr", 3.0))

    @property
    def max_positions(self) -> int:
        return int(self.risk.get("max_positions", 5))

    @property
    def auto_flatten(self) -> bool:
        """Flatten all open positions when a HALT marker is present."""
        return bool(self.safety.get("auto_flatten", True))

    @property
    def flatten_on_critical(self) -> bool:
        """Flatten + write a HALT marker when the monitor raises a CRITICAL alert."""
        return bool(self.safety.get("flatten_on_critical", True))

    @property
    def halt_file(self) -> Path:
        return Path(self.data_dir) / "HALT"

    @property
    def pause_file(self) -> Path:
        return Path(self.data_dir) / "PAUSE"

    @property
    def pause_on_errors(self) -> int:
        """Consecutive failed cycles before entries are paused (live)."""
        return int(self.safety.get("pause_on_errors", 5))

    @property
    def flatten_on_errors(self) -> int:
        """Consecutive failed cycles before flatten + HALT (live)."""
        return int(self.safety.get("flatten_on_errors", 12))

    @property
    def halt_on_stale_cycles(self) -> int:
        """Consecutive cycles with no market data before flatten + HALT (live)."""
        return int(self.safety.get("halt_on_stale_cycles", 6))

    @property
    def reconcile_positions(self) -> bool:
        """Reconcile exchange positions vs bot tracking each cycle (live)."""
        return bool(self.safety.get("reconcile_positions", True))

    @property
    def tpsl_verify(self) -> bool:
        """Re-verify TP/SL attachment on exchange each cycle (live)."""
        return bool(self.safety.get("tpsl_verify", True))

    def taker_fee(self) -> float:
        if self.market_type == "spot":
            return float(self.fees.get("spot_taker", 0.0010))
        return float(self.fees.get("futures_taker", 0.0005))

    def slippage(self) -> float:
        return float(self.fees.get("slippage_bps", 5)) / 10_000.0

    def is_live(self) -> bool:
        return self.mode == "live"

    def is_paper(self) -> bool:
        return self.mode == "paper"

    def is_backtest(self) -> bool:
        return self.mode == "backtest"

    # -- constructors ----------------------------------------------------
    @classmethod
    def load(cls, config_path: str | Path = "config.json", env_path: str | Path = ".env") -> "Settings":
        path = Path(config_path)
        base = {}
        if path.exists():
            base = json.loads(path.read_text(encoding="utf-8"))
        if env_path:
            _load_dotenv(Path(env_path))

        defaults = asdict(cls())
        cfg = _deep_merge(defaults, base)

        api_key = os.environ.get("COINDCX_API_KEY", "")
        api_secret = os.environ.get("COINDCX_API_SECRET", "")
        mode = os.environ.get("TA_MODE", cfg.get("mode", "backtest"))

        data_dir = Path(__file__).resolve().parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            mode=mode,
            quote=cfg.get("quote", "USDT"),
            market_type=cfg.get("market_type", "futures"),
            leverage=float(cfg.get("leverage", 3)),
            watchlist=list(cfg.get("watchlist", defaults["watchlist"])),
            timeframes=list(cfg.get("timeframes", defaults["timeframes"])),
            trade_timeframe=cfg.get("trade_timeframe", "1h"),
            trend_timeframes=list(cfg.get("trend_timeframes", defaults["trend_timeframes"])),
            confidence_timeframes=list(cfg.get("confidence_timeframes", defaults["confidence_timeframes"])),
            max_data_bars=dict(cfg.get("max_data_bars", {})),
            risk=dict(cfg.get("risk", {})),
            exit=dict(cfg.get("exit", {})),
            fees=dict(cfg.get("fees", {})),
            news=dict(cfg.get("news", {})),
            learning=dict(cfg.get("learning", {})),
            backtest=dict(cfg.get("backtest", {})),
            safety=dict(cfg.get("safety", {})),
            api_key=api_key,
            api_secret=api_secret,
            data_dir=str(data_dir),
            _raw=cfg,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_raw", None)
        d["api_key"] = "***" if self.api_key else ""
        d["api_secret"] = "***" if self.api_secret else ""
        return d

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Settings(mode={self.mode}, market_type={self.market_type}, quote={self.quote})"

"""Portfolio management: correlation, diversification and allocation caps.

Prevents over-concentration in a single coin or in highly-correlated
positions (e.g. BTC + ETH moving together).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import timestamps

log = logging.getLogger("ta_agent.portfolio")


@dataclass
class PortfolioCheck:
    approved: bool
    reason: str
    correlations: Optional[Dict[str, float]] = None
    group_exposure: float = 0.0
    total_exposure: float = 0.0


class PortfolioManager:
    def __init__(self, settings, max_correlated: float = 0.50):
        self.s = settings
        self.max_correlated = max_correlated
        self._corr: Optional[pd.DataFrame] = None

    def update_correlation(self, frames: Dict[str, pd.DataFrame]) -> None:
        """Build correlation matrix from daily closes of each coin."""
        closes: Dict[str, pd.Series] = {}
        for coin, df in frames.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            s = df["close"].copy()
            s.index = timestamps.col_to_datetime(df["time"]) if "time" in df.columns else s.index
            closes[coin] = s
        if len(closes) < 2:
            self._corr = None
            return
        panel = pd.DataFrame(closes)
        daily = panel.resample("1D").last().pct_change().dropna(how="all")
        if len(daily) < 5:
            self._corr = None
            return
        self._corr = daily.corr()

    def check_new_position(self, coin: str, notional: float, equity: float,
                           open_positions: Dict[str, float]) -> PortfolioCheck:
        """Validate adding a position against correlation and weight caps."""
        total_open = sum(open_positions.values())
        total_exposure = (total_open + notional) / max(equity, 1e-9)
        max_coin = float(self.s.risk.get("max_coin_weight", 0.25))
        if (notional + open_positions.get(coin, 0.0)) / max(equity, 1e-9) > max_coin:
            return PortfolioCheck(False, f"{coin} would exceed max coin weight {max_coin:.0%}",
                                  total_exposure=total_exposure)

        if not open_positions or self._corr is None or coin not in self._corr.columns:
            return PortfolioCheck(True, "", total_exposure=total_exposure)

        corr_map: Dict[str, float] = {}
        group_notional = notional
        for other, other_notional in open_positions.items():
            if other in self._corr.columns and other != coin:
                c = float(self._corr.loc[coin, other])
                corr_map[other] = c
                if c > 0.7:
                    group_notional += other_notional
        if corr_map:
            for other, c in corr_map.items():
                if c > 0.7:
                    log.info("Correlation %s~%s: %.2f", coin, other, c)
        group_exposure = group_notional / max(equity, 1e-9)
        if group_exposure > self.max_correlated:
            return PortfolioCheck(
                False,
                f"correlated group exposure {group_exposure:.0%} exceeds cap {self.max_correlated:.0%}",
                corr_map, group_exposure, total_exposure,
            )
        return PortfolioCheck(True, "", corr_map, group_exposure, total_exposure)

    def allocation_targets(self, coins: List[str]) -> Dict[str, float]:
        """Equal-risk allocation weights (used for capital planning)."""
        if not coins:
            return {}
        w = 1.0 / len(coins)
        return {c: w for c in coins}

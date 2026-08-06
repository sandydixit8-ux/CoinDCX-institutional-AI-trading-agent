"""News and macro event risk.

EconomicCalendar computes the approximate dates of high-impact events
(CPI, NFP, FOMC...) so the bot can reduce exposure / avoid entries inside a
blackout window. NewsSentiment optionally consumes a news API (GNews) to add
a sentiment layer; without an API key it returns a neutral signal so the
system still runs.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

log = logging.getLogger("ta_agent.news")

_POS_WORDS = {"surge", "soar", "gain", "rally", "bull", "adopt", "approve", "upgrade",
              "record", "milestone", "breakout", "launch", "partnership", "etf", "inflow",
              "buy", "long", "accumulate", "strong", "beat", "growth"}
_NEG_WORDS = {"crash", "plunge", "dump", "fear", "ban", "sue", "lawsuit", "hack", "exploit",
              "regulate", "scam", "fraud", "sell-off", "outflow", "warning", "downgrade",
              "liquidation", "bear", "collapse", "turbulence", "recession", "risk"}


@dataclass
class EventRisk:
    risk: float            # 0 none .. 1 extreme
    nearest_event: str
    within_hours: float


def _first_weekday(y: int, m: int, dow: int, week: int = 1) -> dt.date:
    """First occurrence of weekday ``dow`` in month; ``week``=1..5."""
    first = dt.date(y, m, 1)
    offset = (dow - first.weekday()) % 7
    d = first + dt.timedelta(days=offset + 7 * (week - 1))
    return d


class EconomicCalendar:
    """Approximate high-impact event calendar.

    Uses known scheduling heuristics (NFP = first Friday, US CPI ~mid-month,
    FOMC on the 8 standard meeting months). For production, point this at a
    real calendar feed (e.g. TradingEconomics / Forexlive) by extending
    ``events()``.
    """

    FOMC_MONTHS = [1, 3, 5, 6, 7, 9, 10, 12]

    def events(self, start: dt.date, end: dt.date) -> List[Dict[str, object]]:
        evts: List[Dict[str, object]] = []
        y = start.year
        while y <= end.year:
            for m in range(1, 13):
                d = dt.date(y, m, 1)
                if not (start <= d <= end):
                    continue
                # NFP: first Friday
                nfp = _first_weekday(y, m, 4, 1)
                evts.append({"name": "US Non-Farm Payrolls", "date": nfp,
                             "time_utc": dt.datetime(nfp.year, nfp.month, nfp.day, 12, 30),
                             "impact": 0.9})
                # US CPI: heuristic mid-month
                cpi_day = min(m, 28)
                cpi = dt.date(y, m, cpi_day)
                evts.append({"name": "US CPI", "date": cpi,
                             "time_utc": dt.datetime(cpi.year, cpi.month, cpi.day, 12, 30),
                             "impact": 0.8})
                # FOMC
                if m in self.FOMC_MONTHS:
                    fomc = _first_weekday(y, m, 2, 1) + dt.timedelta(days=13)
                    fomc = _clamp_to_month(fomc, y, m)
                    evts.append({"name": "FOMC Rate Decision", "date": fomc,
                                 "time_utc": dt.datetime(fomc.year, fomc.month, fomc.day, 18, 0),
                                 "impact": 1.0})
            y += 1
        evts = [e for e in evts if start <= e["date"] <= end]
        evts.sort(key=lambda e: e["date"])
        return evts

    def event_risk(self, now: Optional[dt.datetime] = None,
                   blackout_hours: float = 2.0) -> EventRisk:
        now = now or dt.datetime.now(dt.timezone.utc)
        start = (now - dt.timedelta(days=1)).date()
        end = (now + dt.timedelta(days=45)).date()
        best_risk = 0.0
        best_name = "none"
        best_within = 1e9
        for e in self.events(start, end):
            t = e["time_utc"]
            if t.tzinfo is None:
                t = t.replace(tzinfo=dt.timezone.utc)
            hours_away = (t - now).total_seconds() / 3600.0
            # blackout window before + after
            if -1 <= hours_away <= blackout_hours:
                risk = float(e["impact"])
                if risk > best_risk:
                    best_risk = risk
                    best_name = str(e["name"])
                    best_within = max(hours_away, 0.0)
        if best_risk <= 0:
            return EventRisk(0.0, "none", 0.0)
        return EventRisk(best_risk, best_name, best_within)


def _clamp_to_month(d: dt.date, y: int, m: int) -> dt.date:
    """Ensure a computed FOMC date stays within its month."""
    import calendar
    last = calendar.monthrange(y, m)[1]
    if d.month != m:
        return dt.date(y, m, last)
    return d


class NewsSentiment:
    def __init__(self, api_key: str = "", base_url: str = "", timeout: float = 10.0):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY", "")
        self.base_url = base_url or os.environ.get("NEWS_API_BASE", "https://gnews.io/api/v4/search")
        self.timeout = timeout
        self.cache: Dict[str, tuple] = {}

    def sentiment(self, coin: str, hours: int = 24) -> dict:
        """Return {score, events, risk} for a coin from recent headlines."""
        cache_key = f"{coin}:{hours}"
        if cache_key in self.cache:
            return self.cache[cache_key][0]
        if not self.api_key:
            neutral = {"score": 0.0, "headlines": 0, "risk": 0.0, "source": "off"}
            self.cache[cache_key] = (neutral, None)
            return neutral
        try:
            r = requests.get(
                self.base_url,
                params={"q": f"{coin} crypto OR {coin} bitcoin OR {coin}",
                        "lang": "en", "max": 10, "apikey": self.api_key},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json().get("articles", [])
            text = " ".join((a.get("title", "") + " " + a.get("description", "")) for a in data).lower()
            pos = sum(1 for w in _POS_WORDS if w in text)
            neg = sum(1 for w in _NEG_WORDS if w in text)
            denom = max(pos + neg, 1)
            score = (pos - neg) / denom
            risk = 0.0 if score == 0 else max(0.0, 1.0 - score) if score < 0 else 0.0
            if neg > 3 or ("hack" in text or "exploit" in text or "ban" in text):
                risk = max(risk, 0.7)
            out = {"score": score, "headlines": len(data), "risk": risk, "source": "gnews"}
            self.cache[cache_key] = (out, dt.datetime.now(dt.timezone.utc))
            return out
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("News fetch failed for %s: %s", coin, exc)
            out = {"score": 0.0, "headlines": 0, "risk": 0.0, "source": "error"}
            self.cache[cache_key] = (out, None)
            return out

    def aggregate_risk(self, coins: List[str]) -> float:
        if not coins:
            return 0.0
        return max((self.sentiment(c).get("risk", 0.0) for c in coins), default=0.0)

"""Economic calendar for the news-blackout filter.

Scrapes ForexFactory's public weekly calendar. Cached weekly. We only need:
    - timestamp (UTC)
    - currency (USD primarily; EUR/CNY for context)
    - impact (low/medium/high)
    - event name
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class EventRow:
    ts_utc: datetime
    currency: str
    impact: str
    event: str


def fetch_calendar(start: datetime, end: datetime) -> pd.DataFrame:
    """Return calendar events in [start, end) as a DataFrame."""
    raise NotImplementedError


def in_blackout(
    ts: datetime,
    events: pd.DataFrame,
    minutes_before: int,
    minutes_after: int,
    impact_levels: list[str],
) -> bool:
    """True if `ts` falls inside a blackout window around a matching event."""
    raise NotImplementedError

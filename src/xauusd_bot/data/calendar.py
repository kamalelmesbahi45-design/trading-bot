"""Economic calendar for the news-blackout filter.

Two surfaces:

* Live (forward) feed: ForexFactory's free weekly CSV via FairEconomy:
      https://nfs.faireconomy.media/ff_calendar_thisweek.csv
      https://nfs.faireconomy.media/ff_calendar_nextweek.csv

* Historical (backfill) feed: scrape FF's calendar HTML week-by-week.
  Stubbed here -- implement before the news-blackout filter is enabled
  in backtest. Live/paper trading works on the current/next-week feed.

Schema:
    ts_utc   : pandas tz-aware UTC timestamp
    currency : ISO 4217 (USD, EUR, ...)
    impact   : "low" | "medium" | "high"
    event    : event title (string)
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from xauusd_bot.data.storage import load_parquet, save_parquet

FF_CURRENT_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.csv"
FF_NEXT_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.csv"

_IMPACT_MAP = {
    "high": "high",
    "medium": "medium",
    "low": "low",
    "holiday": "low",
    "non-economic": "low",
    "": "low",
}

_COLUMNS = ["ts_utc", "currency", "impact", "event"]


@dataclass(frozen=True)
class EventRow:
    ts_utc: datetime
    currency: str
    impact: str
    event: str


def parse_ff_csv(text: str) -> pd.DataFrame:
    """Parse the FairEconomy/FF weekly CSV. Pure function, no network.

    The CSV header columns are:
        Title,Country,Date,Time,Impact,Forecast,Previous,URL
    Date is m/d/yyyy, Time is h:mma (e.g. '8:30am'). Times are US/Eastern in FF.
    Some rows have Time='All Day' or empty -- those become midnight UTC.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, object]] = []
    for r in reader:
        ts = _ff_row_to_utc(r.get("Date", ""), r.get("Time", ""))
        if ts is None:
            continue
        impact_raw = (r.get("Impact") or "").strip().lower()
        rows.append(
            {
                "ts_utc": ts,
                "currency": (r.get("Country") or "").strip().upper(),
                "impact": _IMPACT_MAP.get(impact_raw, "low"),
                "event": (r.get("Title") or "").strip(),
            }
        )
    if not rows:
        return _empty_events_frame()
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df.sort_values("ts_utc").reset_index(drop=True)


def _ff_row_to_utc(date_str: str, time_str: str) -> datetime | None:
    """Convert FF (US/Eastern) date+time to a tz-aware UTC datetime."""
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%m/%d/%Y").date()
    except ValueError:
        return None

    if not time_str or time_str.lower() in ("all day", "tentative"):
        naive = datetime(d.year, d.month, d.day, 0, 0)
    else:
        try:
            t = datetime.strptime(time_str.upper().replace(" ", ""), "%I:%M%p").time()
        except ValueError:
            return None
        naive = datetime(d.year, d.month, d.day, t.hour, t.minute)

    # Localise to US/Eastern, convert to UTC.
    ts: datetime = (
        pd.Timestamp(naive)
        .tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        .tz_convert("UTC")
        .to_pydatetime()
    )
    return ts


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "currency": pd.Series(dtype="object"),
            "impact": pd.Series(dtype="object"),
            "event": pd.Series(dtype="object"),
        }
    )


def fetch_current_window(cache_dir: Path | None = None, timeout_s: float = 15.0) -> pd.DataFrame:
    """Fetch this-week + next-week CSVs and return a unified events frame."""
    frames: list[pd.DataFrame] = []
    for url in (FF_CURRENT_URL, FF_NEXT_URL):
        try:
            r = httpx.get(url, timeout=timeout_s)
            r.raise_for_status()
            frames.append(parse_ff_csv(r.text))
        except httpx.HTTPError as e:
            logger.warning(f"calendar fetch failed for {url}: {e}")
    if not frames:
        return _empty_events_frame()
    df = pd.concat(frames).drop_duplicates(subset=["ts_utc", "currency", "event"]).reset_index(drop=True)

    if cache_dir is not None:
        out = cache_dir / "calendar" / f"current_{datetime.utcnow():%Y%m%d}.parquet"
        save_parquet(df, out)
    return df


def fetch_calendar(start: datetime, end: datetime, cache_dir: Path | None = None) -> pd.DataFrame:
    """Return calendar events in [start, end).

    Falls back to the current-week feed when the requested range is within ±10 days
    of now. Otherwise raises NotImplementedError; historical scraping will be
    implemented before the news filter is enabled in backtest mode.
    """
    now = datetime.utcnow()
    if start >= now - timedelta(days=10) and end <= now + timedelta(days=14):
        events = fetch_current_window(cache_dir=cache_dir)
        mask = (events["ts_utc"] >= pd.Timestamp(start, tz="UTC")) & (
            events["ts_utc"] < pd.Timestamp(end, tz="UTC")
        )
        return events.loc[mask].reset_index(drop=True)

    raise NotImplementedError(
        "Historical calendar backfill not implemented yet. Live/paper trading uses "
        "fetch_current_window(); backtest with news filter is blocked until the FF "
        "HTML scraper lands. Set filters.news.enabled=false to backtest without it."
    )


def in_blackout(
    ts: datetime,
    events: pd.DataFrame,
    minutes_before: int,
    minutes_after: int,
    impact_levels: list[str],
    currencies: list[str] | None = None,
) -> bool:
    """True if `ts` falls inside a blackout window around a matching event."""
    if events is None or events.empty:
        return False
    if currencies is None:
        currencies = ["USD"]  # XAUUSD: dollar events dominate

    target = pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")
    before = pd.Timedelta(minutes=minutes_before)
    after = pd.Timedelta(minutes=minutes_after)

    impacts = set(impact_levels)
    ccys = set(currencies)
    mask = events["impact"].isin(impacts) & events["currency"].isin(ccys)
    if not mask.any():
        return False
    window = events.loc[mask, "ts_utc"]
    return bool(((window - before <= target) & (target <= window + after)).any())


def load_cached_calendar(path: Path) -> pd.DataFrame:
    return load_parquet(path)

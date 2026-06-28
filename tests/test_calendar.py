"""Offline tests for FF CSV parsing + blackout window logic. No network."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from xauusd_bot.data.calendar import _empty_events_frame, in_blackout, parse_ff_csv

SAMPLE_CSV = """Title,Country,Date,Time,Impact,Forecast,Previous,URL
"Non-Farm Employment Change",USD,1/5/2024,8:30am,High,180K,199K,https://example.com/nfp
"Unemployment Rate",USD,1/5/2024,8:30am,High,3.8%,3.7%,https://example.com/ur
"Average Hourly Earnings",USD,1/5/2024,8:30am,Medium,0.3%,0.4%,https://example.com/ahe
"Bank Holiday",JPY,1/8/2024,All Day,Holiday,,,https://example.com/hol
"Empty Time",EUR,1/8/2024,,Low,,,https://example.com/x
"Bad Date",USD,not-a-date,8:30am,High,,,https://example.com/y
"""


def test_parse_ff_csv_basic() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    assert len(df) == 5  # bad-date row dropped
    assert set(df.columns) == {"ts_utc", "currency", "impact", "event"}

    nfp = df[df["event"] == "Non-Farm Employment Change"].iloc[0]
    # 8:30am US/Eastern on Jan 5 2024 = 13:30 UTC (winter standard time)
    assert nfp["ts_utc"] == pd.Timestamp("2024-01-05 13:30", tz="UTC")
    assert nfp["impact"] == "high"
    assert nfp["currency"] == "USD"


def test_parse_ff_csv_holiday_and_all_day_become_midnight() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    hol = df[df["event"] == "Bank Holiday"].iloc[0]
    # "All Day" -> midnight US/Eastern on date = 05:00 UTC (winter)
    assert hol["ts_utc"] == pd.Timestamp("2024-01-08 05:00", tz="UTC")
    assert hol["impact"] == "low"   # holiday demoted to low


def test_parse_ff_csv_impact_mapping() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    assert df[df["event"] == "Non-Farm Employment Change"]["impact"].iloc[0] == "high"
    assert df[df["event"] == "Average Hourly Earnings"]["impact"].iloc[0] == "medium"
    assert df[df["event"] == "Bank Holiday"]["impact"].iloc[0] == "low"


def test_parse_ff_csv_empty_input() -> None:
    df = parse_ff_csv("Title,Country,Date,Time,Impact,Forecast,Previous,URL\n")
    assert df.empty
    assert set(df.columns) == {"ts_utc", "currency", "impact", "event"}


def test_in_blackout_no_events_false() -> None:
    assert not in_blackout(
        datetime(2024, 1, 5, 13, 30, tzinfo=UTC),
        _empty_events_frame(),
        minutes_before=15,
        minutes_after=15,
        impact_levels=["high"],
    )


def test_in_blackout_inside_window_true() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    # NFP at 13:30 UTC, blackout 15min before/after -> 13:15 to 13:45 is blocked
    for ts in [
        datetime(2024, 1, 5, 13, 15, tzinfo=UTC),   # left edge
        datetime(2024, 1, 5, 13, 30, tzinfo=UTC),   # exact
        datetime(2024, 1, 5, 13, 45, tzinfo=UTC),   # right edge
        datetime(2024, 1, 5, 13, 32, tzinfo=UTC),   # middle
    ]:
        assert in_blackout(ts, df, 15, 15, ["high"]), ts


def test_in_blackout_outside_window_false() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    for ts in [
        datetime(2024, 1, 5, 13, 14, tzinfo=UTC),
        datetime(2024, 1, 5, 13, 46, tzinfo=UTC),
        datetime(2024, 1, 6, 13, 30, tzinfo=UTC),
    ]:
        assert not in_blackout(ts, df, 15, 15, ["high"]), ts


def test_in_blackout_respects_impact_filter() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    # The 13:30 USD row exists with impact "high" AND "medium" (different events)
    # Asking only for medium events: still blackout because AHE is medium at 13:30
    assert in_blackout(
        datetime(2024, 1, 5, 13, 30, tzinfo=UTC), df, 15, 15, ["medium"]
    )
    # Asking only for low: not blackout (no USD low events at that minute)
    assert not in_blackout(
        datetime(2024, 1, 5, 13, 30, tzinfo=UTC), df, 15, 15, ["low"]
    )


def test_in_blackout_respects_currency_filter() -> None:
    df = parse_ff_csv(SAMPLE_CSV)
    # Empty-time EUR event at 05:00 UTC on 1/8 (low impact). Default currencies=["USD"] -> no blackout.
    ts = datetime(2024, 1, 8, 5, 0, tzinfo=UTC)
    assert not in_blackout(ts, df, 15, 15, ["low"])
    # Including EUR -> blackout
    assert in_blackout(ts, df, 15, 15, ["low"], currencies=["EUR"])

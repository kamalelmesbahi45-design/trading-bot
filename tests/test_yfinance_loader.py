"""Offline tests for yfinance loader normalisation + wide reshape. No network."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from xauusd_bot.data.yfinance_loader import TICKERS, _empty_frame, _normalise_one, to_wide


def _fake_yf_frame(naive_tz: bool = False) -> pd.DataFrame:
    """Mimic yfinance's single-ticker output."""
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 14:00",
            "2024-01-02 15:00",
        ],
        name="Date",
    )
    if not naive_tz:
        idx = idx.tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "Open":   [100.0, 101.0],
            "High":   [101.5, 102.5],
            "Low":    [ 99.5, 100.5],
            "Close":  [101.0, 102.0],
            "Volume": [10_000, 12_000],
        },
        index=idx,
    )


def test_tickers_dict_covers_required_assets() -> None:
    for k in ("DXY", "VIX", "US10Y", "SPX", "OIL", "BTC", "GOLD_FUT"):
        assert k in TICKERS


def test_normalise_one_tz_naive_input_localises_to_utc() -> None:
    out = _normalise_one(_fake_yf_frame(naive_tz=True), "DXY")
    assert str(out.index.tz) == "UTC"
    assert (out["ticker"] == "DXY").all()
    assert list(out.columns) == ["ticker", "open", "high", "low", "close", "volume"]


def test_normalise_one_tz_aware_input_converts_to_utc() -> None:
    out = _normalise_one(_fake_yf_frame(naive_tz=False), "SPX")
    assert str(out.index.tz) == "UTC"
    # 14:00 New York = 19:00 UTC (no DST in January past 2007)
    assert out.index[0] == pd.Timestamp("2024-01-02 19:00", tz="UTC")


def test_normalise_one_empty_input_returns_empty_frame() -> None:
    out = _normalise_one(pd.DataFrame(), "DXY")
    assert out.empty
    assert "ticker" in out.columns


def test_empty_frame_schema() -> None:
    df = _empty_frame()
    assert df.empty
    assert "ticker" in df.columns
    for c in ("open", "high", "low", "close", "volume"):
        assert c in df.columns


def test_to_wide_pivots_close_by_ticker() -> None:
    a = _normalise_one(_fake_yf_frame(), "DXY")
    b = _normalise_one(_fake_yf_frame(), "VIX")
    b = b * 1  # noop -- variety not required
    long_df = pd.concat([a, b]).sort_index()
    wide = to_wide(long_df, field="close")
    assert "DXY" in wide.columns
    assert "VIX" in wide.columns
    assert len(wide) == 2  # two timestamps
    assert datetime(2024, 1, 2, 19, tzinfo=UTC) in wide.index


def test_to_wide_empty_returns_empty() -> None:
    wide = to_wide(_empty_frame(), field="close")
    assert wide.empty

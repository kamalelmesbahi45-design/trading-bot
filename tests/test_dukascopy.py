"""Offline tests for the Dukascopy bi5 codec and helpers. No network."""
from __future__ import annotations

import lzma
from datetime import UTC, datetime

from xauusd_bot.data.dukascopy import (
    PRICE_SCALE,
    TICK_STRUCT,
    _decode_bi5,
    _empty_tick_frame,
    _hour_url,
    _iter_hours,
    ticks_to_ohlcv,
)


def _encode_ticks(ticks: list[tuple[int, int, int, float, float]]) -> bytes:
    """Build a synthetic raw-LZMA bi5 payload like Dukascopy serves."""
    payload = b"".join(TICK_STRUCT.pack(*t) for t in ticks)
    return lzma.compress(payload, format=lzma.FORMAT_ALONE)


def test_decode_empty_payload_returns_empty_frame() -> None:
    df = _decode_bi5(b"", datetime(2024, 1, 2, 8))
    assert df.empty
    assert list(df.columns) == ["bid", "ask", "bid_vol", "ask_vol"]
    assert df.index.name == "ts"


def test_decode_lzma_empty_after_decompress_returns_empty_frame() -> None:
    payload = lzma.compress(b"", format=lzma.FORMAT_ALONE)
    df = _decode_bi5(payload, datetime(2024, 1, 2, 8))
    assert df.empty


def test_decode_round_trips_synthetic_ticks() -> None:
    hour = datetime(2024, 1, 2, 8)
    raw = [
        # (ms_offset, ask_int, bid_int, ask_vol, bid_vol)
        (0, int(2055.123 * PRICE_SCALE), int(2054.998 * PRICE_SCALE), 1.0, 1.5),
        (500, int(2055.200 * PRICE_SCALE), int(2055.050 * PRICE_SCALE), 0.5, 0.7),
        (1_000, int(2055.180 * PRICE_SCALE), int(2055.030 * PRICE_SCALE), 0.2, 0.4),
    ]
    df = _decode_bi5(_encode_ticks(raw), hour)
    assert len(df) == 3
    assert df.index.tz == UTC
    # ms offsets translate to absolute timestamps from hour start
    assert df.index[0] == datetime(2024, 1, 2, 8, 0, 0, tzinfo=UTC)
    assert df.index[1] == datetime(2024, 1, 2, 8, 0, 0, 500_000, tzinfo=UTC)
    # Prices scaled back
    assert abs(df["ask"].iloc[0] - 2055.123) < 1e-9
    assert abs(df["bid"].iloc[0] - 2054.998) < 1e-9
    # Volumes preserved as float
    assert abs(df["bid_vol"].iloc[0] - 1.5) < 1e-6


def test_hour_url_uses_zero_indexed_month() -> None:
    url = _hour_url("XAUUSD", datetime(2024, 1, 2, 8))   # Jan -> 00
    assert url.endswith("XAUUSD/2024/00/02/08h_ticks.bi5")
    url2 = _hour_url("XAUUSD", datetime(2024, 12, 31, 23))  # Dec -> 11
    assert url2.endswith("XAUUSD/2024/11/31/23h_ticks.bi5")


def test_iter_hours_inclusive_start_exclusive_end() -> None:
    hours = list(_iter_hours(datetime(2024, 1, 2, 8), datetime(2024, 1, 2, 11)))
    assert hours == [
        datetime(2024, 1, 2, 8),
        datetime(2024, 1, 2, 9),
        datetime(2024, 1, 2, 10),
    ]


def test_ticks_to_ohlcv_builds_bars_from_mid() -> None:
    hour = datetime(2024, 1, 2, 8)
    # 3 ticks within the first minute, 1 in the second minute
    raw = [
        (0,         int(2055.000 * PRICE_SCALE), int(2054.500 * PRICE_SCALE), 1.0, 1.0),
        (15_000,    int(2056.000 * PRICE_SCALE), int(2055.500 * PRICE_SCALE), 1.0, 1.0),
        (45_000,    int(2054.000 * PRICE_SCALE), int(2053.500 * PRICE_SCALE), 1.0, 1.0),
        (75_000,    int(2057.000 * PRICE_SCALE), int(2056.500 * PRICE_SCALE), 1.0, 1.0),
    ]
    ticks = _decode_bi5(_encode_ticks(raw), hour)
    bars = ticks_to_ohlcv(ticks, "1min")
    assert len(bars) == 2
    first = bars.iloc[0]
    # mid = (bid+ask)/2 -> 2054.75, 2055.75, 2053.75; OHLC -> 2054.75 / 2055.75 / 2053.75 / 2053.75
    assert abs(first["open"] - 2054.75) < 1e-6
    assert abs(first["high"] - 2055.75) < 1e-6
    assert abs(first["low"] - 2053.75) < 1e-6
    assert abs(first["close"] - 2053.75) < 1e-6
    assert abs(first["volume"] - 6.0) < 1e-6  # 3 ticks * (1+1) volume each


def test_ticks_to_ohlcv_empty_input() -> None:
    bars = ticks_to_ohlcv(_empty_tick_frame(), "1min")
    assert bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_tick_struct_size_is_20() -> None:
    assert TICK_STRUCT.size == 20
    # Pack/unpack a known record
    packed = TICK_STRUCT.pack(1234, 99999, 88888, 1.0, 2.0)
    assert len(packed) == 20
    assert TICK_STRUCT.unpack(packed) == (1234, 99999, 88888, 1.0, 2.0)

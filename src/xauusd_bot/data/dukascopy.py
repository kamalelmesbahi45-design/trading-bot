"""Dukascopy XAUUSD tick downloader.

URL pattern (per UTC hour, month is 0-indexed):
    https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM-1:02}/{DD:02}/{HH:02}h_ticks.bi5

Wire format:
    - File is raw LZMA-compressed (lzma.FORMAT_ALONE), not xz.
    - Decoded payload is a stream of 20-byte records: struct ">iiiff":
        ms_offset_int32, ask_int, bid_int, ask_vol_f32, bid_vol_f32
    - For XAUUSD prices are scaled by PRICE_SCALE (3 decimals).
    - ms_offset is milliseconds from the hour start (UTC).

Weekends and market-closed hours legitimately return 404 or an empty body.
"""
from __future__ import annotations

import asyncio
import lzma
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential_jitter

from xauusd_bot.data.storage import hour_cache_path, load_parquet, save_parquet

DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"
SYMBOL = "XAUUSD"
PRICE_SCALE = 1000.0
TICK_STRUCT = struct.Struct(">iiiff")
TICK_SIZE = TICK_STRUCT.size  # 20

_TICK_COLUMNS = ["bid", "ask", "bid_vol", "ask_vol"]


@dataclass
class DukascopyConfig:
    cache_dir: Path
    max_workers: int = 8
    retries: int = 3
    timeout_s: float = 30.0
    symbol: str = SYMBOL


def _hour_url(symbol: str, dt: datetime) -> str:
    """Build the bi5 URL for a UTC hour. Dukascopy months are 0-indexed."""
    return (
        f"{DUKASCOPY_BASE}/{symbol}/"
        f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/"
        f"{dt.hour:02d}h_ticks.bi5"
    )


def _decode_bi5(payload: bytes, hour_start: datetime, price_scale: float = PRICE_SCALE) -> pd.DataFrame:
    """Decode raw LZMA payload to a tick DataFrame indexed by UTC timestamp.

    Empty payload (market closed) yields an empty DataFrame with the right schema.
    """
    if not payload:
        return _empty_tick_frame()

    decompressed = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    n = len(decompressed) // TICK_SIZE
    if n == 0:
        return _empty_tick_frame()

    rows = TICK_STRUCT.iter_unpack(decompressed[: n * TICK_SIZE])
    timestamps: list[datetime] = []
    bids: list[float] = []
    asks: list[float] = []
    bid_vols: list[float] = []
    ask_vols: list[float] = []
    hour_ts = hour_start.replace(tzinfo=UTC) if hour_start.tzinfo is None else hour_start
    for ms_offset, ask_int, bid_int, ask_vol, bid_vol in rows:
        timestamps.append(hour_ts + timedelta(milliseconds=ms_offset))
        asks.append(ask_int / price_scale)
        bids.append(bid_int / price_scale)
        ask_vols.append(float(ask_vol))
        bid_vols.append(float(bid_vol))

    return pd.DataFrame(
        {"bid": bids, "ask": asks, "bid_vol": bid_vols, "ask_vol": ask_vols},
        index=pd.DatetimeIndex(timestamps, name="ts", tz="UTC"),
    )


def _empty_tick_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="ts", tz="UTC")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in _TICK_COLUMNS}, index=idx)


async def _fetch_one(
    client: httpx.AsyncClient,
    hour: datetime,
    cfg: DukascopyConfig,
    sem: asyncio.Semaphore,
) -> pd.DataFrame:
    """Fetch a single hour, with cache and retry. Returns tick DataFrame (possibly empty)."""
    cache_file = hour_cache_path(cfg.cache_dir, cfg.symbol, hour.year, hour.month, hour.day, hour.hour)
    if cache_file.exists():
        return load_parquet(cache_file)

    url = _hour_url(cfg.symbol, hour)

    async def _do() -> bytes:
        r = await client.get(url, timeout=cfg.timeout_s)
        if r.status_code == 404:
            return b""
        r.raise_for_status()
        return r.content

    async with sem:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(cfg.retries),
                wait=wait_exponential_jitter(initial=1.0, max=8.0),
                reraise=True,
            ):
                with attempt:
                    payload = await _do()
                    break
            else:  # pragma: no cover - AsyncRetrying with reraise always either yields or raises
                payload = b""
        except RetryError as e:  # pragma: no cover - reraise=True bypasses this
            logger.warning(f"giving up on {url}: {e}")
            payload = b""

    df = _decode_bi5(payload, hour)
    save_parquet(df, cache_file)
    return df


async def _download_range_async(start: datetime, end: datetime, cfg: DukascopyConfig) -> pd.DataFrame:
    sem = asyncio.Semaphore(cfg.max_workers)
    hours = list(_iter_hours(start, end))
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_one(client, h, cfg, sem) for h in hours]
        frames = await asyncio.gather(*tasks)
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return _empty_tick_frame()
    out = pd.concat(non_empty).sort_index()
    return out


def _iter_hours(start: datetime, end: datetime) -> Iterator[datetime]:
    cur = start.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    end = end.replace(tzinfo=None)
    while cur < end:
        yield cur
        cur += timedelta(hours=1)


def download_ticks(start: datetime, end: datetime, cfg: DukascopyConfig) -> pd.DataFrame:
    """Synchronous wrapper for download_range_async. Caches per-hour to parquet."""
    return asyncio.run(_download_range_async(start, end, cfg))


def ticks_to_ohlcv(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample tick stream to OHLCV bars using the mid price.

    timeframe: pandas offset alias ('1min', '5min', '15min', '1h', '4h', '1D').
    Returns columns: open, high, low, close, volume (sum of bid_vol+ask_vol per bar).
    """
    if ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    mid = (ticks["bid"] + ticks["ask"]) * 0.5
    vol = ticks["bid_vol"].fillna(0.0) + ticks["ask_vol"].fillna(0.0)
    df = pd.DataFrame({"mid": mid, "vol": vol})

    agg = df.resample(timeframe, label="left", closed="left").agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
        volume=("vol", "sum"),
    )
    return agg.dropna(subset=["open"])

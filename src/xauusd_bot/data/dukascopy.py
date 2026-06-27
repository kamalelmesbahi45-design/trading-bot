"""Dukascopy XAUUSD tick downloader.

Endpoint pattern (hourly bi5 files):
    https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM-1:02}/{DD:02}/{HH:02}h_ticks.bi5

Each .bi5 is LZMA-compressed; decoded payload is records of:
    >iiiff   (ms_offset, ask_int, bid_int, ask_vol, bid_vol)
prices stored as ints scaled by 1000 for XAUUSD (3 decimal places).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"
SYMBOL = "XAUUSD"
PRICE_SCALE = 1000.0  # XAUUSD 3-decimal scaling


@dataclass
class DukascopyConfig:
    cache_dir: Path
    max_workers: int = 8
    retries: int = 3
    timeout_s: float = 30.0


def download_ticks(start: datetime, end: datetime, cfg: DukascopyConfig) -> pd.DataFrame:
    """Download XAUUSD tick data in [start, end). Cached per-hour file under cfg.cache_dir.

    Returns a DataFrame indexed by UTC timestamp with columns: bid, ask, bid_vol, ask_vol.
    """
    raise NotImplementedError


def ticks_to_ohlcv(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample tick stream to OHLCV bars using mid price.

    timeframe: pandas offset alias ('1min', '5min', '15min', '1h', '4h', '1D').
    """
    raise NotImplementedError

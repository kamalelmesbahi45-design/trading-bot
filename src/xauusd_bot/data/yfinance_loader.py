"""Cross-asset feed via yfinance for the macro/regime layer.

Default tickers:
    DXY:      DX-Y.NYB (Yahoo's US dollar index symbol)
    VIX:      ^VIX
    US10Y:    ^TNX  (yield * 10, e.g. 4.25% -> 42.5; divided in normalisation)
    SPX:      ^GSPC
    OIL:      CL=F
    BTC:      BTC-USD
    GOLD_FUT: GC=F  (sanity vs Dukascopy spot)

We persist a tidy long DataFrame: columns ['ticker','open','high','low','close','volume']
indexed by UTC timestamp. Wide views are derived from this in features/.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from xauusd_bot.data.storage import cache_path, load_parquet, save_parquet

TICKERS: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "US10Y": "^TNX",
    "SPX": "^GSPC",
    "OIL": "CL=F",
    "BTC": "BTC-USD",
    "GOLD_FUT": "GC=F",
}

_OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class YFinanceConfig:
    cache_dir: Path
    interval: str = "1h"          # "1h", "1d"
    auto_adjust: bool = True


def _normalise_one(df: pd.DataFrame, ticker_label: str) -> pd.DataFrame:
    """Normalise a single-ticker yfinance frame to our schema.

    yfinance returns columns like ['Open','High','Low','Close','Adj Close','Volume']
    indexed by tz-aware (or tz-naive) DatetimeIndex.
    """
    if df.empty:
        return _empty_frame()

    df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
    keep = [c for c in _OHLCV if c in df.columns]
    out = df[keep].copy()
    # Ensure tz-aware UTC
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "ts"
    out.insert(0, "ticker", ticker_label)
    return out


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="ts", tz="UTC")
    return pd.DataFrame(
        {"ticker": pd.Series(dtype="object"), **{c: pd.Series(dtype="float64") for c in _OHLCV}},
        index=idx,
    )


def fetch_cross_asset(
    start: datetime,
    end: datetime,
    cfg: YFinanceConfig,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch the configured tickers and return one tidy DataFrame.

    Each ticker is fetched independently so a single failing symbol doesn't poison
    the whole batch. Results are cached per (ticker, interval) to parquet.
    """
    selected = tickers or list(TICKERS.keys())
    frames: list[pd.DataFrame] = []
    for label in selected:
        if label not in TICKERS:
            logger.warning(f"unknown cross-asset ticker '{label}', skipping")
            continue
        df = _fetch_one_cached(label, TICKERS[label], start, end, cfg)
        if not df.empty:
            frames.append(df)
    if not frames:
        return _empty_frame()
    return pd.concat(frames).sort_index()


def _fetch_one_cached(
    label: str,
    yahoo_symbol: str,
    start: datetime,
    end: datetime,
    cfg: YFinanceConfig,
) -> pd.DataFrame:
    period = f"{cfg.interval}_{start:%Y%m%d}_{end:%Y%m%d}"
    out = cache_path(cfg.cache_dir, "cross_asset", label, period)
    if out.exists():
        return load_parquet(out)

    raw = yf.download(
        yahoo_symbol,
        start=start,
        end=end,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        threads=False,
    )
    # yfinance sometimes returns a single-level MultiIndex when threads=False but ticker is single string
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = _normalise_one(raw, label)
    save_parquet(df, out)
    return df


def to_wide(long_df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Reshape long output to wide: one column per ticker for a single field."""
    if long_df.empty:
        return pd.DataFrame()
    return long_df.pivot_table(index=long_df.index, columns="ticker", values=field)

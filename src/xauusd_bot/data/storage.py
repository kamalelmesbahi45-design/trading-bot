"""Parquet storage for ticks, bars, and cross-asset data. Cheap re-reads, deterministic cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write df to path. Creates parent directories. Uses pyarrow + zstd."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd")


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def cache_path(root: Path, kind: str, symbol: str, period: str) -> Path:
    """Canonical cache path: <root>/<kind>/<symbol>/<period>.parquet"""
    return root / kind / symbol / f"{period}.parquet"


def hour_cache_path(root: Path, symbol: str, year: int, month: int, day: int, hour: int) -> Path:
    """Per-hour tick cache: <root>/ticks/<symbol>/YYYY/MM/DD/HH.parquet"""
    return root / "ticks" / symbol / f"{year:04d}" / f"{month:02d}" / f"{day:02d}" / f"{hour:02d}.parquet"

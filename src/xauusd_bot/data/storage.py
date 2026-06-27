"""Parquet storage for ticks, bars, and cross-asset data. Cheap re-reads, deterministic cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def cache_path(root: Path, kind: str, symbol: str, period: str) -> Path:
    """Canonical cache path: <root>/<kind>/<symbol>/<period>.parquet"""
    return root / kind / symbol / f"{period}.parquet"

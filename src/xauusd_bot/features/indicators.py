"""Classical technical indicators. Vectorised, no look-ahead.

Each function takes a price/HLC DataFrame and returns a Series indexed identically.
"""
from __future__ import annotations

import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    raise NotImplementedError


def ema(series: pd.Series, period: int) -> pd.Series:
    raise NotImplementedError


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    raise NotImplementedError


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (mid, upper, lower)."""
    raise NotImplementedError


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Returns (upper_band, lower_band)."""
    raise NotImplementedError


def keltner(high: pd.Series, low: pd.Series, close: pd.Series, ema_p: int = 20, atr_p: int = 14, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    raise NotImplementedError

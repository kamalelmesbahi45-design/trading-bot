"""XAUUSD-internal regime features: vol regime, trend strength, session label."""
from __future__ import annotations

from datetime import time

import pandas as pd


def atr_pct(close: pd.Series, atr_series: pd.Series) -> pd.Series:
    """ATR as percent of price. Used to filter dead chop / parabolic vol."""
    raise NotImplementedError


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Trend-strength score 0..100."""
    raise NotImplementedError


def session_label(ts: pd.DatetimeIndex) -> pd.Series:
    """Label each bar: 'asia' / 'london' / 'ny' / 'overlap' / 'off'."""
    raise NotImplementedError


LONDON_OPEN = time(7, 0)   # UTC
NY_OPEN = time(12, 0)
NY_CLOSE = time(21, 0)

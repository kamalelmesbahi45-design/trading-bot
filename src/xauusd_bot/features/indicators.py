"""Classical technical indicators. Vectorised, no look-ahead.

Each function takes price Series/columns and returns a Series indexed identically.
Pandas-only (no numba/talib) so they JIT cleanly in tests and stay portable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_period(period: int) -> None:
    if period <= 0:
        raise ValueError("period must be positive")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR = max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range -- Wilder's smoothing (EWMA with alpha = 1/period)."""
    _check_period(period)
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    _check_period(period)
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    _check_period(period)
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    _check_period(period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there are no losses in the window, RSI saturates to 100.
    return out.where(avg_loss > 0, 100.0)


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (mid, upper, lower)."""
    _check_period(period)
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Returns (upper_band, lower_band). Look-ahead-free: uses [i-period, i-1]."""
    _check_period(period)
    upper = high.shift(1).rolling(period, min_periods=period).max()
    lower = low.shift(1).rolling(period, min_periods=period).min()
    return upper, lower


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_p: int = 20,
    atr_p: int = 14,
    k: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (mid, upper, lower)."""
    mid = ema(close, ema_p)
    a = atr(high, low, close, atr_p)
    return mid, mid + k * a, mid - k * a


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ADX -- trend-strength 0..100."""
    _check_period(period)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=high.index)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

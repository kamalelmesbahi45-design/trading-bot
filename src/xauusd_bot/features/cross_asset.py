"""Cross-asset features for the macro/regime layer.

Inputs: daily-ish OHLC series for DXY, US10Y, VIX, and XAUUSD spot/futures.
Outputs: pandas Series indexed identically to the input, forward-fillable.
"""
from __future__ import annotations

import pandas as pd

from xauusd_bot.features.indicators import ema


def dxy_trend(dxy: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """Returns +1 if DXY EMA(fast) > EMA(slow) (USD up trend = gold negative),
    -1 otherwise. NaN during warm-up.
    """
    ef = ema(dxy, fast)
    es = ema(dxy, slow)
    diff = ef - es
    out = diff.where(diff.isna(), diff.apply(lambda x: 1.0 if x > 0 else -1.0))
    return out


def real_yield_proxy(us10y: pd.Series, breakeven_proxy: pd.Series | None = None) -> pd.Series:
    """Real-yield proxy. If no TIPS breakeven supplied, falls back to nominal change.

    Returns the YoY change (or first-difference if <250 obs) -- positive values are
    'rising real yields' which is gold-negative.
    """
    real = us10y - breakeven_proxy if breakeven_proxy is not None else us10y
    if len(real) >= 250:
        return real.diff(250)
    return real.diff()


def vix_regime(vix: pd.Series, low: float = 15.0, high: float = 25.0) -> pd.Series:
    """0=calm, 1=normal, 2=stressed."""
    out = pd.Series(1.0, index=vix.index)
    out = out.where(vix >= low, 0.0)
    out = out.where(vix <= high, 2.0)
    return out


def gold_dxy_beta(gold: pd.Series, dxy: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta of gold returns to DXY returns. Negative is typical.

    A breakdown (beta swinging towards 0 or positive) flags a regime change.
    """
    g_ret = gold.pct_change()
    d_ret = dxy.pct_change()
    cov = g_ret.rolling(window).cov(d_ret)
    var = d_ret.rolling(window).var()
    return (cov / var).replace([float("inf"), -float("inf")], pd.NA)

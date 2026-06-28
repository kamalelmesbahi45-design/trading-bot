"""Time-series momentum signals + volatility targeting.

The documented CTA edge (Moskowitz-Ooi-Pedersen 2012; AQR "A Century of Evidence
on Trend-Following"): go long markets with positive trailing return, short those
with negative, size each inversely to its own volatility so every position
contributes roughly equal risk, then scale the whole book to a target vol.

No look-ahead: every signal/weight at date t uses only data up to and including
t, and is applied to the return from t to t+1 by the backtester.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_vol(returns: pd.DataFrame, halflife: int = 33, min_periods: int = 20) -> pd.DataFrame:
    """Annualised EWMA volatility per asset. halflife ~33d ~= 1.5 trading months."""
    var = returns.ewm(halflife=halflife, min_periods=min_periods).var()
    return np.sqrt(var * TRADING_DAYS)


def tsmom_signal(
    panel: pd.DataFrame,
    lookbacks: tuple[int, ...] = (63, 126, 252),
) -> pd.DataFrame:
    """Blended time-series-momentum signal in [-1, 1] per asset/date.

    For each lookback L, raw = sign(close_t / close_{t-L} - 1). Average across
    lookbacks (so a market trending on all horizons gets +-1; mixed gets a
    fractional value). NaN until the longest lookback is available.
    """
    sigs = []
    for lb in lookbacks:
        trailing = panel / panel.shift(lb) - 1.0
        sigs.append(np.sign(trailing))
    blended = sum(sigs) / len(sigs)
    return blended


def vol_target_weights(
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    per_asset_vol_target: float = 0.10,
    max_leverage_per_asset: float = 2.0,
) -> pd.DataFrame:
    """Translate signals into per-asset weights via inverse-vol scaling.

    weight_i = signal_i * (per_asset_vol_target / vol_i), capped at
    max_leverage_per_asset in absolute value. Assets with missing vol get 0.
    """
    inv = per_asset_vol_target / vol.replace(0.0, np.nan)
    w = signal * inv
    w = w.clip(lower=-max_leverage_per_asset, upper=max_leverage_per_asset)
    return w.fillna(0.0)


def scale_to_portfolio_vol(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_annual_vol: float = 0.10,
    lookback: int = 126,
    max_gross_leverage: float = 4.0,
) -> pd.DataFrame:
    """Scale the whole book each day so realised portfolio vol ~= target.

    Uses the trailing realised vol of the *unscaled* strategy returns. The
    scalar is shifted by one day so it only uses past information. Gross
    leverage (sum |weight|) is capped to keep the book sane.
    """
    raw_port_ret = (weights.shift(1) * returns).sum(axis=1)
    realised = raw_port_ret.rolling(lookback, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    scalar = (target_annual_vol / realised.replace(0.0, np.nan)).shift(1)
    scalar = scalar.clip(upper=10.0).fillna(0.0)
    scaled = weights.mul(scalar, axis=0)

    gross = scaled.abs().sum(axis=1)
    over = gross > max_gross_leverage
    if over.any():
        shrink = pd.Series(1.0, index=scaled.index)
        shrink[over] = max_gross_leverage / gross[over]
        scaled = scaled.mul(shrink, axis=0)
    return scaled

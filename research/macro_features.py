"""Macro features derived from the joined daily frame. All look-ahead-free.

The daily frame must contain at least: gold_close, dxy. Optional: dgs10, t10yie,
dfii10, vix. Missing series produce NaN features; downstream filters degrade
gracefully rather than crashing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def real_yield(df: pd.DataFrame) -> pd.Series:
    """DGS10 - T10YIE (preferred) or DFII10 direct."""
    if "dgs10" in df and "t10yie" in df:
        return (df["dgs10"] - df["t10yie"]).rename("real_yield")
    if "dfii10" in df:
        return df["dfii10"].rename("real_yield")
    return pd.Series(np.nan, index=df.index, name="real_yield")


def real_yield_falling(df: pd.DataFrame, win: int = 20) -> pd.Series:
    ry = real_yield(df)
    return (ry.diff(win) < 0).astype(float).rename("ry_falling")


def real_yield_regime(df: pd.DataFrame, win: int = 250) -> pd.Series:
    """+1 bullish gold (real yield negative-and-falling), -1 bearish, 0 neutral."""
    ry = real_yield(df)
    if ry.isna().all():
        return pd.Series(np.nan, index=df.index, name="ry_regime")
    z = _zscore(ry, win)
    falling = ry.diff(win // 5) < 0
    regime = pd.Series(0.0, index=df.index)
    regime[(z < -0.5) & falling] = 1.0
    regime[(z > 0.5) & ~falling] = -1.0
    return regime.rename("ry_regime")


def dxy_trend(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """+1 USD up trend (gold negative), -1 USD down trend (gold positive)."""
    if "dxy" not in df:
        return pd.Series(np.nan, index=df.index, name="dxy_trend")
    d = df["dxy"]
    f = d.ewm(span=fast, min_periods=fast).mean()
    s = d.ewm(span=slow, min_periods=slow).mean()
    diff = f - s
    return pd.Series(np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0)),
                     index=df.index, name="dxy_trend")


def dxy_falling(df: pd.DataFrame, win: int = 20) -> pd.Series:
    if "dxy" not in df:
        return pd.Series(np.nan, index=df.index, name="dxy_falling")
    return (df["dxy"].diff(win) < 0).astype(float).rename("dxy_falling")


def vix_regime(df: pd.DataFrame, low: float = 15.0, high: float = 25.0) -> pd.Series:
    if "vix" not in df:
        return pd.Series(np.nan, index=df.index, name="vix_regime")
    v = df["vix"]
    return pd.Series(np.where(v < low, 0.0, np.where(v > high, 2.0, 1.0)),
                     index=df.index, name="vix_regime")


def gold_dxy_beta(df: pd.DataFrame, win: int = 60) -> pd.Series:
    if "dxy" not in df:
        return pd.Series(np.nan, index=df.index, name="gold_dxy_beta")
    g = df["gold_close"].pct_change()
    d = df["dxy"].pct_change()
    cov = g.rolling(win).cov(d)
    var = d.rolling(win).var()
    return (cov / var.replace(0.0, np.nan)).rename("gold_dxy_beta")


def gold_dxy_decorrelation(df: pd.DataFrame, win: int = 60, beta_thresh: float = -0.1) -> pd.Series:
    """Returns 1 when 60d gold/DXY beta is no-longer-inverse (>= -0.1). Marks regime breaks."""
    b = gold_dxy_beta(df, win)
    return (b > beta_thresh).astype(float).rename("gold_dxy_decor")


def divergence_signal(df: pd.DataFrame, win: int = 20) -> pd.Series:
    """Gold up AND DXY up together over `win` days -> strong other-driver -> +1 (bullish-continue).
    Gold down AND DXY down -> -1 (bearish-continue). Else 0."""
    if "dxy" not in df:
        return pd.Series(np.nan, index=df.index, name="divergence")
    g = df["gold_close"].diff(win)
    d = df["dxy"].diff(win)
    out = pd.Series(0.0, index=df.index)
    out[(g > 0) & (d > 0)] = 1.0
    out[(g < 0) & (d < 0)] = -1.0
    return out.rename("divergence")


def vix_spike(df: pd.DataFrame, z_thresh: float = 1.5, win: int = 60) -> pd.Series:
    """1 on days where VIX z-score over `win` exceeds `z_thresh`."""
    if "vix" not in df:
        return pd.Series(np.nan, index=df.index, name="vix_spike")
    z = _zscore(df["vix"], win)
    return (z > z_thresh).astype(float).rename("vix_spike")


def adx_proxy(df: pd.DataFrame, win: int = 20) -> pd.Series:
    """Simple trend-strength proxy: range of close over `win` / mean ATR-ish.
    Higher = stronger trend. Used to filter MR strategies."""
    if "gold_close" not in df:
        return pd.Series(np.nan, index=df.index, name="adx_proxy")
    rng = df["gold_close"].rolling(win).max() - df["gold_close"].rolling(win).min()
    atr_proxy = (df["gold_high"] - df["gold_low"]).rolling(win).mean()
    return (rng / atr_proxy.replace(0.0, np.nan)).rename("adx_proxy")


def is_first_two_weeks_january(df: pd.DataFrame) -> pd.Series:
    """Seasonality flag: 1 during Jan 2-14, 0 otherwise."""
    idx = df.index
    months = idx.month
    days = idx.day
    flag = ((months == 1) & (days >= 2) & (days <= 14)).astype(float)
    return pd.Series(flag, index=df.index, name="jan_2w")


def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature frame."""
    feats = pd.concat([
        real_yield(df),
        real_yield_falling(df),
        real_yield_regime(df),
        dxy_trend(df),
        dxy_falling(df),
        vix_regime(df),
        vix_spike(df),
        gold_dxy_beta(df),
        gold_dxy_decorrelation(df),
        divergence_signal(df),
        adx_proxy(df),
        is_first_two_weeks_january(df),
    ], axis=1)
    return feats

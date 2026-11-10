"""Indicators. All computed with information available at the bar's close (no look-ahead)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14, ohlc: bool = True) -> pd.Series:
    if ohlc:
        pc = df["Close"].shift()
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    else:
        # close-only data: mean absolute close-to-close move (documented proxy, ~0.8 sigma)
        tr = df["Close"].diff().abs()
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_features(df: pd.DataFrame, ohlc: bool, periods_per_year: int = 252) -> pd.DataFrame:
    f = df.copy()
    c = f["Close"]
    r = np.log(c).diff()
    f["ret"] = r
    f["atr"] = atr(f, 14, ohlc)
    f["sma5"] = c.rolling(5).mean()
    f["sma20"] = c.rolling(20).mean()
    f["sma50"] = c.rolling(50).mean()
    f["sma200"] = c.rolling(200).mean()
    f["ema20"] = c.ewm(span=20, adjust=False).mean()
    f["ema50"] = c.ewm(span=50, adjust=False).mean()
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)
    f["hh20"] = f["High"].rolling(20).max().shift(1)      # PRIOR 20-bar high
    f["ll20"] = f["Low"].rolling(20).min().shift(1)
    f["hh55"] = f["High"].rolling(55).max().shift(1)
    f["ll55"] = f["Low"].rolling(55).min().shift(1)
    rng = (f["High"] - f["Low"]).replace(0, np.nan)
    f["ibs"] = ((c - f["Low"]) / rng).clip(0, 1)            # internal bar strength
    # time-series momentum (Moskowitz-Ooi-Pedersen): 1, 3, 12 month
    ppm = periods_per_year / 12
    for m in (1, 3, 12):
        f[f"mom{m}"] = c / c.shift(int(round(ppm * m))) - 1
    # volatility
    f["vol20"] = r.rolling(20).std() * np.sqrt(periods_per_year)
    f["ewvol"] = np.sqrt((r ** 2).ewm(span=30, adjust=False).mean()) * np.sqrt(periods_per_year)
    f["vol_pct"] = f["vol20"].rolling(periods_per_year, min_periods=120).rank(pct=True)
    # Bollinger squeeze
    sd = c.rolling(20).std()
    f["bbw"] = 4 * sd / f["sma20"]
    f["bbw_pct"] = f["bbw"].rolling(periods_per_year, min_periods=120).rank(pct=True)
    f["squeeze_recent"] = (f["bbw_pct"] < 0.15).rolling(5).max().astype(bool)
    return f


def vol_regime(p: float) -> str:
    if p != p:
        return "n/a"
    if p >= 0.95:
        return "Extreme"
    if p >= 0.75:
        return "High"
    if p <= 0.25:
        return "Low"
    return "Normal"

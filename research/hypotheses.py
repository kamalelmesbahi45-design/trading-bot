"""Edge hypotheses for XAUUSD daily.

Each Hypothesis is a small dataclass with:
    * id, name, mechanism (text)
    * generator(features, bars) -> pd.Series of {+1 long, -1 short, 0 flat} positions
      indexed identically to bars.

The tester rebuilds them into trades and applies the same fill simulator and
cost model used in live trading.

All position series MUST be look-ahead-free: at row i they may only use rows[:i].
The tester enforces this by computing returns on bars.shift(-1) of close-to-close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from research import macro_features as mf


@dataclass
class Hypothesis:
    id: str
    name: str
    mechanism: str
    generator: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
    requires: tuple[str, ...] = ()    # required columns in features


# ---------- helpers ----------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, min_periods=n).mean()


def _donchian_break(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    """+1 long when close > prior-n-high, -1 short when close < prior-n-low. Look-ahead-free."""
    upper = high.shift(1).rolling(n).max()
    lower = low.shift(1).rolling(n).min()
    out = pd.Series(0.0, index=close.index)
    out[close > upper] = 1.0
    out[close < lower] = -1.0
    return out


def _bollinger_fade(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """Mean-reversion: +1 long when close < lower, -1 short when close > upper, else 0."""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    out = pd.Series(0.0, index=close.index)
    out[close < (mid - k * sd)] = 1.0
    out[close > (mid + k * sd)] = -1.0
    return out


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    l = (-d).clip(lower=0).ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    rs = g / l.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _hold_n_days(signal: pd.Series, n: int) -> pd.Series:
    """Hold any non-zero signal for n days then flatten (unless reversed)."""
    out = signal.copy().astype(float)
    days_left = 0
    cur = 0.0
    vals = []
    for s in signal.values:
        if s != 0 and s != cur:
            cur = float(s)
            days_left = n
        elif days_left > 0:
            days_left -= 1
            if days_left == 0:
                cur = 0.0
        vals.append(cur)
    return pd.Series(vals, index=signal.index)


# ---------- H1: real-yield regime-conditional trend ----------

def h1_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """Long-only Donchian, but ONLY when real-yield regime is bullish gold (+1)."""
    don = _donchian_break(bars["close"], bars["high"], bars["low"], 20)
    regime = feats.get("ry_regime")
    if regime is None:
        return pd.Series(0.0, index=bars.index)
    sig = don.where(regime > 0, 0.0)
    sig[sig < 0] = 0.0           # long-only in bullish regime
    return _hold_n_days(sig, 10)


# ---------- H2: DXY-divergence momentum (both gold and DXY up = other-driver) ----------

def h2_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    div = feats.get("divergence", pd.Series(0.0, index=bars.index))
    don = _donchian_break(bars["close"], bars["high"], bars["low"], 20)
    sig = pd.Series(0.0, index=bars.index)
    # If both rising together AND price made a 20d high -> long continuation
    sig[(div > 0) & (don > 0)] = 1.0
    sig[(div < 0) & (don < 0)] = -1.0
    return _hold_n_days(sig, 10)


# ---------- H3: DXY-trend filter on Donchian (long only when DXY down-trend) ----------

def h3_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    don = _donchian_break(bars["close"], bars["high"], bars["low"], 20)
    dxy_t = feats.get("dxy_trend", pd.Series(0.0, index=bars.index))
    sig = pd.Series(0.0, index=bars.index)
    sig[(don > 0) & (dxy_t < 0)] = 1.0   # gold breakout while DXY down-trending
    sig[(don < 0) & (dxy_t > 0)] = -1.0  # gold breakdown while DXY up-trending
    return _hold_n_days(sig, 10)


# ---------- H4: VIX-stressed long-gold tilt ----------

def h4_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    vix_r = feats.get("vix_regime", pd.Series(np.nan, index=bars.index))
    # When VIX is stressed (regime 2), bias long gold above 20d SMA
    sma20 = _sma(bars["close"], 20)
    sig = pd.Series(0.0, index=bars.index)
    sig[(vix_r >= 2) & (bars["close"] > sma20)] = 1.0
    return _hold_n_days(sig, 5)


# ---------- H5: VIX spike long-gold ----------

def h5_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    spike = feats.get("vix_spike", pd.Series(0.0, index=bars.index))
    sig = pd.Series(0.0, index=bars.index)
    sig[spike > 0] = 1.0    # long gold on any VIX z>1.5 day
    return _hold_n_days(sig, 5)


# ---------- H6: gold-DXY decorrelation regime change -> long-gold ----------

def h6_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    decor = feats.get("gold_dxy_decor", pd.Series(0.0, index=bars.index))
    sig = pd.Series(0.0, index=bars.index)
    sig[decor > 0] = 1.0   # beta no longer inverse -> other-driver (central banks, geopolitics)
    return _hold_n_days(sig, 10)


# ---------- H7: pre-FOMC drift (Tue/Wed before 3rd Wed) ----------

def h7_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """Approximation: long gold on Mondays + Tuesdays in months with FOMC meetings
    (8 per year, roughly every 6 weeks). This is a coarse proxy without FOMC calendar.
    For honest test, we use ALL weeks (this dilutes the edge -> a true pre-FOMC drift
    would survive even diluted)."""
    dows = bars.index.dayofweek
    sig = pd.Series(np.where((dows == 0) | (dows == 1), 1.0, 0.0), index=bars.index)
    return _hold_n_days(sig.where(sig > 0, 0.0), 1)


# ---------- H8: COT-extreme reversal (proxy: gold % above 50d SMA) ----------

def h8_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """Without real COT data, proxy 'spec stretched' as gold > 10% above 50d SMA -> fade short.
    Below -7% -> long."""
    sma50 = _sma(bars["close"], 50)
    z = (bars["close"] - sma50) / sma50
    sig = pd.Series(0.0, index=bars.index)
    sig[z > 0.10] = -1.0
    sig[z < -0.07] = 1.0
    return _hold_n_days(sig, 5)


# ---------- H9: Bollinger MR when ADX-proxy is LOW (range regime) ----------

def h9_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    bb = _bollinger_fade(bars["close"], 20, 2.0)
    adx = feats.get("adx_proxy", pd.Series(np.nan, index=bars.index))
    sig = bb.where(adx < adx.quantile(0.4), 0.0)
    return _hold_n_days(sig, 3)


# ---------- H10: Asian-range fade (close in Asia, fade at NY open) ----------
# Skipped on daily-only data; only meaningful on intraday.

# ---------- H11: January seasonality long ----------

def h11_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    flag = feats.get("jan_2w", pd.Series(0.0, index=bars.index))
    return flag * 1.0


# ---------- H12: combined real-yield-falling AND DXY-falling -> long ----------

def h12_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    ry_f = feats.get("ry_falling", pd.Series(np.nan, index=bars.index))
    dxy_f = feats.get("dxy_falling", pd.Series(np.nan, index=bars.index))
    sig = pd.Series(0.0, index=bars.index)
    sig[(ry_f > 0) & (dxy_f > 0)] = 1.0
    return _hold_n_days(sig, 10)


# ---------- H13: RSI mean-reversion with real-yield bullish regime ----------

def h13_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    rsi = _rsi(bars["close"], 14)
    regime = feats.get("ry_regime", pd.Series(np.nan, index=bars.index))
    sig = pd.Series(0.0, index=bars.index)
    sig[(rsi < 30) & (regime > 0)] = 1.0   # only buy oversold in bullish regime
    return _hold_n_days(sig, 5)


# ---------- H14: SMA200 trend filter + 20d momentum ----------

def h14_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    sma200 = _sma(bars["close"], 200)
    mom20 = bars["close"].pct_change(20)
    sig = pd.Series(0.0, index=bars.index)
    sig[(bars["close"] > sma200) & (mom20 > 0)] = 1.0     # uptrend + recent up = long
    sig[(bars["close"] < sma200) & (mom20 < 0)] = -1.0    # downtrend + recent down = short
    return _hold_n_days(sig, 10)


# ---------- BH: always-long benchmark (NOT a strategy, just a reference) ----------

def bh_gen(feats: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=bars.index)


# ---------- registry ----------

HYPOTHESES: list[Hypothesis] = [
    Hypothesis("H1", "Donchian + bullish real-yield regime",
               "Trend-follow only when real rates favour gold",
               h1_gen),
    Hypothesis("H2", "Gold/DXY divergence continuation",
               "Both rising together = other driver (CB buying, geo) -> continuation",
               h2_gen),
    Hypothesis("H3", "Donchian + DXY counter-trend filter",
               "Gold breakout only when DXY in downtrend",
               h3_gen),
    Hypothesis("H4", "VIX-stressed long-gold above 20d",
               "Risk-off + uptrend confirmation = safe-haven flow",
               h4_gen),
    Hypothesis("H5", "VIX-spike long-gold",
               "Bond/equity panic days -> gold demand",
               h5_gen),
    Hypothesis("H6", "Gold/DXY decorrelation regime",
               "When beta no longer inverse, other driver dominates",
               h6_gen),
    Hypothesis("H7", "Mon+Tue drift (pre-FOMC proxy)",
               "Documented pre-event drift; coarse calendar-day proxy",
               h7_gen),
    Hypothesis("H8", "Extension-from-SMA50 mean reversion",
               "Spec-positioning proxy: extreme extension reverts",
               h8_gen),
    Hypothesis("H9", "Bollinger MR in low-trend regime",
               "Fade bands only when not trending (ADX-proxy bottom 40%)",
               h9_gen),
    Hypothesis("H11", "January 1st-half seasonality",
               "Calendar effect: rebalancing into commodities Jan 2-14",
               h11_gen),
    Hypothesis("H12", "Real-yield-falling AND DXY-falling",
               "Joint macro tailwind",
               h12_gen),
    Hypothesis("H13", "RSI oversold + bullish real-yield regime",
               "Buy dips only in supportive macro",
               h13_gen),
    Hypothesis("H14", "SMA200 trend + 20d momentum",
               "Classic dual-confirmation trend-follower",
               h14_gen),
    Hypothesis("BH", "Buy-and-hold benchmark",
               "Reference for whether any strategy actually beats holding gold",
               bh_gen),
]

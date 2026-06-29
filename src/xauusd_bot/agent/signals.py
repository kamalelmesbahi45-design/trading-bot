"""Day-trading and swing-trading signals.

Two horizons, each producing a per-asset score in [-1, 1] and a side label.

* swing_signal:
    Blended trend (3/6/12-month TSMOM, the AQR / Moskowitz-Ooi-Pedersen edge)
    plus cross-sectional rank within the universe (long the strongest, short
    the weakest). Cross-sectional rank stabilises the book when absolute trend
    is weak across the board.

* day_signal:
    Short-horizon (5d/20d) momentum, RSI(14) extremes for mean-reversion in
    range markets, ATR breakout vs the 20-day Donchian channel, and a relative
    volume (RVOL) filter so we only act when the move has participation.
    Mean-reversion is gated to assets currently NOT in a strong trend (avoids
    fading a real breakout), trend continuation is gated to assets in trend.

All functions are pure: input frames in, output series/frames out. No globals,
no look-ahead -- every value at timestamp ``t`` uses data up to and including
``t``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_dn = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def realised_vol(close: pd.Series, halflife: int = 33) -> float:
    r = close.pct_change().dropna()
    if r.empty:
        return float("nan")
    v = r.ewm(halflife=halflife, min_periods=10).std().iloc[-1]
    return float(v * np.sqrt(TRADING_DAYS))


def donchian(ohlc: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    hi = ohlc["high"].rolling(period).max()
    lo = ohlc["low"].rolling(period).min()
    return hi, lo


def rvol(volume: pd.Series, period: int = 20) -> float:
    v = volume.dropna()
    if v.empty:
        return 1.0
    base = v.rolling(period).mean()
    if base.empty or not np.isfinite(base.iloc[-1]) or base.iloc[-1] == 0:
        return 1.0
    return float(v.iloc[-1] / base.iloc[-1])


def trailing_return(close: pd.Series, lookback: int) -> float:
    s = close.dropna()
    if len(s) <= lookback:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - lookback] - 1.0)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
@dataclass
class SignalRead:
    score: float            # [-1, 1]
    side: str               # LONG / SHORT / FLAT
    components: dict[str, float]
    strategy: str           # "swing_trend" | "day_momentum" | "day_mean_revert" | "day_breakout"


def _side(score: float, threshold: float = 0.15) -> str:
    if score > threshold:
        return "LONG"
    if score < -threshold:
        return "SHORT"
    return "FLAT"


def swing_signal(close: pd.Series, xs_rank: float = 0.0) -> SignalRead:
    """Blended 3/6/12m TSMOM with optional cross-sectional rank kicker."""
    r3 = trailing_return(close, 63)
    r6 = trailing_return(close, 126)
    r12 = trailing_return(close, 252)
    comps = [np.sign(x) for x in (r3, r6, r12) if x == x]  # drop NaN
    raw = float(np.mean(comps)) if comps else 0.0
    score = 0.7 * raw + 0.3 * xs_rank
    score = float(np.clip(score, -1.0, 1.0))
    return SignalRead(
        score=score,
        side=_side(score),
        components={
            "ret_3m": 0.0 if r3 != r3 else r3,
            "ret_6m": 0.0 if r6 != r6 else r6,
            "ret_12m": 0.0 if r12 != r12 else r12,
            "xs_rank": xs_rank,
        },
        strategy="swing_trend",
    )


def day_signal(ohlc: pd.DataFrame) -> SignalRead:
    """Short-horizon signal: combines momentum, mean-reversion, breakout.

    Picks the best-aligned of three sub-strategies based on the current
    micro-regime (trend strength) of the asset itself. Cleanest mental model:

        strong trend  -> momentum or breakout continuation
        weak / ranging -> RSI mean-reversion at extremes
    """
    close = ohlc["close"].dropna()
    if len(close) < 60:
        return SignalRead(0.0, "FLAT", {}, "day_momentum")

    # short-horizon momentum
    r5 = trailing_return(close, 5) or 0.0
    r20 = trailing_return(close, 20) or 0.0
    mom_score = float(np.clip(0.5 * np.sign(r5) + 0.5 * np.sign(r20), -1.0, 1.0))

    # mean-reversion via RSI
    rsi_now = float(rsi(close).iloc[-1])
    mr_score = 0.0
    if rsi_now <= 30.0:
        mr_score = float(np.clip((30.0 - rsi_now) / 20.0, 0.0, 1.0))  # long
    elif rsi_now >= 70.0:
        mr_score = -float(np.clip((rsi_now - 70.0) / 20.0, 0.0, 1.0))  # short

    # Donchian breakout (20d): close above/below channel
    hi, lo = donchian(ohlc, 20)
    br_score = 0.0
    if not np.isnan(hi.iloc[-1]) and not np.isnan(lo.iloc[-1]):
        if close.iloc[-1] >= hi.iloc[-2]:
            br_score = 1.0
        elif close.iloc[-1] <= lo.iloc[-2]:
            br_score = -1.0

    # regime of the asset itself: is it trending? (50/200 ma alignment)
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50
    trending = bool(np.sign(close.iloc[-1] - ma50) == np.sign(ma50 - ma200) != 0)

    # volume gate (only used if column present and non-NaN)
    rv = 1.0
    if "volume" in ohlc.columns and ohlc["volume"].notna().any():
        rv = rvol(ohlc["volume"])

    # pick strategy
    if abs(br_score) >= 1.0 and rv >= 1.1:
        score = br_score * min(1.0, rv / 1.5)
        strat = "day_breakout"
        comps = {"breakout": br_score, "rvol": rv, "rsi": rsi_now}
    elif trending and abs(mom_score) > 0.0:
        score = mom_score
        strat = "day_momentum"
        comps = {"ret_5d": r5, "ret_20d": r20, "rvol": rv, "rsi": rsi_now}
    elif not trending and abs(mr_score) > 0.0:
        score = mr_score
        strat = "day_mean_revert"
        comps = {"rsi": rsi_now, "ret_5d": r5}
    else:
        score = 0.0
        strat = "day_momentum"
        comps = {"ret_5d": r5, "ret_20d": r20, "rsi": rsi_now, "rvol": rv}

    return SignalRead(
        score=float(np.clip(score, -1.0, 1.0)),
        side=_side(score),
        components={k: round(float(v), 4) for k, v in comps.items()},
        strategy=strat,
    )


def cross_sectional_rank(scores: dict[str, float]) -> dict[str, float]:
    """Translate raw scores to [-1, 1] cross-sectional rank.

    Top half map to positive, bottom half negative; magnitudes proportional to
    distance from the median. Used to add a "long the strongest, short the
    weakest" tilt on top of pure TSMOM.
    """
    if not scores:
        return {}
    s = pd.Series(scores).dropna()
    if s.empty:
        return {k: 0.0 for k in scores}
    ranks = s.rank(method="average")
    n = len(s)
    centered = (ranks - (n + 1) / 2) / ((n - 1) / 2 if n > 1 else 1)
    return {k: float(v) for k, v in centered.items()}

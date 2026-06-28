"""Session breakouts (UTC):

LondonOpenBreakout:
    Mark the range during a fixed window (default 07:00-08:00 UTC), then on the
    bar where close breaks above range_high * (1 + buffer) -> LONG; below
    range_low * (1 - buffer) -> SHORT. Each side fires at most once per day; the
    setup expires expiry_minutes after the range close.

AsianRangeBreakout:
    Same idea but the range is the prior Asia session (00:00-07:00 UTC). The
    breakout window is the subsequent London/NY session.
"""
from __future__ import annotations

from datetime import time
from typing import cast

import pandas as pd

from xauusd_bot.features.indicators import atr
from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Side, Signal

_LONDON_RANGE_START = time(7, 0)
_LONDON_RANGE_END = time(8, 0)
_ASIA_RANGE_START = time(0, 0)
_ASIA_RANGE_END = time(7, 0)


class _SessionBreakout(Strategy):
    """Common scaffold for session breakouts. Subclasses define the range window."""

    range_start: time
    range_end: time

    def __init__(self, expiry_minutes: int = 240, buffer_atr_mult: float = 0.1,
                 atr_period: int = 14) -> None:
        self.expiry_minutes = expiry_minutes
        self.buffer_atr_mult = buffer_atr_mult
        self.atr_period = atr_period

    def _range_for_day(self, bars: pd.DataFrame, day_ts: pd.Timestamp) -> tuple[float | None, float | None]:
        """Return (range_high, range_low) for the configured window on the given UTC date."""
        day = day_ts.date()
        mask = (bars.index.date == day) & (bars.index.time >= self.range_start) & (bars.index.time < self.range_end)
        window = bars[mask]
        if window.empty:
            return None, None
        return float(window["high"].max()), float(window["low"].min())

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        if len(bars) < self.atr_period + 2:
            return []
        last = bars.iloc[-1]
        ts = cast(pd.Timestamp, last.name)
        # Only fire AFTER the range has closed
        if ts.time() < self.range_end:
            return []
        # And only within the expiry window from the range end
        rng_end_dt = pd.Timestamp.combine(ts.date(), self.range_end).tz_localize(ts.tz) if ts.tz else pd.Timestamp.combine(ts.date(), self.range_end)
        if (ts - rng_end_dt).total_seconds() / 60.0 > self.expiry_minutes:
            return []

        hi, lo = self._range_for_day(bars, ts)
        if hi is None or lo is None:
            return []

        a = atr(bars["high"], bars["low"], bars["close"], self.atr_period).iloc[-1]
        if pd.isna(a):
            return []
        buf = self.buffer_atr_mult * float(a)
        close = float(last["close"])

        if close > hi + buf:
            return [Signal(ts=ts, strategy=self.name, side=Side.LONG, confidence=1.0)]
        if close < lo - buf:
            return [Signal(ts=ts, strategy=self.name, side=Side.SHORT, confidence=1.0)]
        return []


class LondonOpenBreakout(_SessionBreakout):
    name = "breakout_london"
    range_start = _LONDON_RANGE_START
    range_end = _LONDON_RANGE_END


class AsianRangeBreakout(_SessionBreakout):
    name = "breakout_asian"
    range_start = _ASIA_RANGE_START
    range_end = _ASIA_RANGE_END

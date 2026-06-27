"""Trend-following: Donchian channel breakout. H1 default.

Rules:
    LONG  when close > donchian_upper.shift(1)  (close above the prior period's high)
    SHORT when close < donchian_lower.shift(1)
Only one signal per bar. Stop levels are computed by the engine's StopPolicy
(ATR-based) so this strategy stays sizing-agnostic.
"""
from __future__ import annotations

import pandas as pd

from xauusd_bot.features.indicators import donchian
from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Side, Signal


class DonchianTrend(Strategy):
    name = "trend_donchian"

    def __init__(self, donchian_period: int = 20) -> None:
        if donchian_period < 2:
            raise ValueError("donchian_period must be >= 2")
        self.donchian_period = donchian_period

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        """Look at the most recent bar only and emit at most one signal for it.

        The engine calls this incrementally with bars[:i+1] for each i, so only
        the last row needs evaluating. This keeps work O(period) per bar.
        """
        if len(bars) <= self.donchian_period + 1:
            return []
        upper, lower = donchian(bars["high"], bars["low"], self.donchian_period)
        last = bars.iloc[-1]
        u = upper.iloc[-1]
        d = lower.iloc[-1]
        if pd.isna(u) or pd.isna(d):
            return []
        close = float(last["close"])
        ts = last.name
        if close > u:
            return [Signal(ts=ts, strategy=self.name, side=Side.LONG, confidence=1.0)]
        if close < d:
            return [Signal(ts=ts, strategy=self.name, side=Side.SHORT, confidence=1.0)]
        return []

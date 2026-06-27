"""Session breakouts:
    - London open range (07:00-08:00 UTC) -> break above/below at 08:00+
    - Asian range (00:00-07:00 UTC)        -> break either side after London open
"""
from __future__ import annotations

import pandas as pd

from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Signal


class LondonOpenBreakout(Strategy):
    name = "breakout_london"

    def __init__(self, range_minutes: int = 60, expiry_minutes: int = 240, buffer_atr_mult: float = 0.1) -> None:
        self.range_minutes = range_minutes
        self.expiry_minutes = expiry_minutes
        self.buffer_atr_mult = buffer_atr_mult

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        raise NotImplementedError


class AsianRangeBreakout(Strategy):
    name = "breakout_asian"

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        raise NotImplementedError

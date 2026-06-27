"""Trend-following: Donchian breakout with Keltner pullback entry. H1 default."""
from __future__ import annotations

import pandas as pd

from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Signal


class DonchianTrend(Strategy):
    name = "trend_donchian"

    def __init__(
        self,
        donchian_period: int = 20,
        atr_period: int = 14,
        atr_mult_sl: float = 2.0,
        atr_mult_tp: float = 3.0,
    ) -> None:
        self.donchian_period = donchian_period
        self.atr_period = atr_period
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        raise NotImplementedError

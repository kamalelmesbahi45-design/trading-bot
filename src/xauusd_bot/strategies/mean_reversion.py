"""Mean-reversion: Bollinger fade with RSI divergence confirmation. M15 default.

Only enabled when regime says range/low-trend. Disabled in propfirm_strict by default.
"""
from __future__ import annotations

import pandas as pd

from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Signal


class BollingerMR(Strategy):
    name = "mr_bollinger"

    def __init__(
        self,
        bb_period: int = 20,
        bb_k: float = 2.0,
        rsi_period: int = 14,
        rsi_buy_below: float = 30.0,
        rsi_sell_above: float = 70.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_k = bb_k
        self.rsi_period = rsi_period
        self.rsi_buy_below = rsi_buy_below
        self.rsi_sell_above = rsi_sell_above

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        raise NotImplementedError

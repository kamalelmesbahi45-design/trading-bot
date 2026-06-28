"""Mean-reversion: Bollinger fade with RSI confirmation.

Logic at the most recent bar:
    LONG  when close < lower band AND RSI < rsi_buy_below
    SHORT when close > upper band AND RSI > rsi_sell_above

Only emits one signal per bar and only on the latest row. Look-ahead-free.
The engine's regime filter (ATR-pct band) is the main protection against
running this in a trending market; disabled in propfirm_strict by default.
"""
from __future__ import annotations

import pandas as pd

from xauusd_bot.features.indicators import bollinger, rsi
from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Side, Signal


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
        if bb_period < 2 or rsi_period < 2:
            raise ValueError("bb_period and rsi_period must be >= 2")
        self.bb_period = bb_period
        self.bb_k = bb_k
        self.rsi_period = rsi_period
        self.rsi_buy_below = rsi_buy_below
        self.rsi_sell_above = rsi_sell_above

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        min_bars = max(self.bb_period, self.rsi_period) + 2
        if len(bars) < min_bars:
            return []
        _, upper, lower = bollinger(bars["close"], self.bb_period, k=self.bb_k)
        r = rsi(bars["close"], self.rsi_period)
        last = bars.iloc[-1]
        u, d, rsi_now = upper.iloc[-1], lower.iloc[-1], r.iloc[-1]
        if pd.isna(u) or pd.isna(d) or pd.isna(rsi_now):
            return []
        close = float(last["close"])
        ts = last.name
        if close < d and rsi_now < self.rsi_buy_below:
            return [Signal(ts=ts, strategy=self.name, side=Side.LONG, confidence=1.0)]
        if close > u and rsi_now > self.rsi_sell_above:
            return [Signal(ts=ts, strategy=self.name, side=Side.SHORT, confidence=1.0)]
        return []

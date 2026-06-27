"""Transaction-cost models for XAUUSD: spread, slippage, commission, swap.

Why dynamic spread matters: gold often shows 25-30c spread in London/NY but
60-100c+ around NFP/FOMC and 80c+ during Asian thin liquidity. A static-spread
backtest will overstate edge by a wide margin.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class SpreadModel(ABC):
    @abstractmethod
    def spread(self, ts: datetime, base_price: float) -> float:
        ...


class StaticSpread(SpreadModel):
    def __init__(self, points: int) -> None:
        self.points = points  # XAUUSD: 1 point = $0.01 in price

    def spread(self, ts: datetime, base_price: float) -> float:
        raise NotImplementedError


class DynamicSpread(SpreadModel):
    """Session- and event-aware. Tighter London/NY, wider Asia, wider near high-impact news."""
    def __init__(self, base_points: int, event_widen_x: float = 4.0, asia_widen_x: float = 2.0) -> None:
        self.base_points = base_points
        self.event_widen_x = event_widen_x
        self.asia_widen_x = asia_widen_x

    def spread(self, ts: datetime, base_price: float) -> float:
        raise NotImplementedError


def commission(qty_lots: float, per_lot_usd: float) -> float:
    return qty_lots * per_lot_usd


def swap(qty_lots: float, side_pct_per_night: float, equity: float, nights: int) -> float:
    raise NotImplementedError

"""Transaction-cost models for XAUUSD: spread, slippage, commission, swap.

Gold is brutal on retail spreads. A static-spread backtest overstates edge by a lot:
real spreads sit ~25-30c London/NY but blow out to 60-100c+ around NFP/FOMC and
80c+ in thin Asian sessions. The DynamicSpread model captures session widening; the
news widening (event_widen_x) is wired through so the engine can apply it when an
event window is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

# XAUUSD price convention: 1 point = $0.01
POINT_VALUE = 0.01


class SpreadModel(ABC):
    @abstractmethod
    def spread(self, ts: datetime, base_price: float, in_event_window: bool = False) -> float:
        """Return half-spread + half-spread, i.e. the full bid-ask gap in price units."""


class StaticSpread(SpreadModel):
    def __init__(self, points: int) -> None:
        self.points = points

    def spread(self, ts: datetime, base_price: float, in_event_window: bool = False) -> float:
        return self.points * POINT_VALUE


class DynamicSpread(SpreadModel):
    """Session- and event-aware spread.

    base   : applied during London/NY (07-21 UTC).
    asia   : applied during Asia (00-07 UTC), widened by `asia_widen_x`.
    event  : during a news-event window the spread widens by `event_widen_x`.
    """
    def __init__(self, base_points: int, event_widen_x: float = 4.0, asia_widen_x: float = 2.0) -> None:
        self.base_points = base_points
        self.event_widen_x = event_widen_x
        self.asia_widen_x = asia_widen_x

    def spread(self, ts: datetime, base_price: float, in_event_window: bool = False) -> float:
        hour = ts.hour
        mult = self.asia_widen_x if 0 <= hour < 7 else 1.0
        if in_event_window:
            mult = max(mult, self.event_widen_x)
        return self.base_points * POINT_VALUE * mult


def commission(qty_lots: float, per_lot_usd: float) -> float:
    """Per-side commission in USD."""
    return max(0.0, qty_lots) * per_lot_usd


def swap(qty_lots: float, side_pct_per_night: float, equity: float, nights: int) -> float:
    """Overnight swap cost. Signed: negative for long when long_swap is negative.

    Approximate model: swap_pct_per_night is applied to the notional value of the
    position which we proxy by (lots * contract_value_proxy). Real brokers use a
    points-per-lot model that varies daily; this is intentionally simple for v1
    and will be refined when calibrated against a specific broker's swap feed.
    """
    if nights <= 0 or qty_lots <= 0:
        return 0.0
    notional = qty_lots * equity * 0.01  # 1% of equity per lot is a rough proxy
    return notional * side_pct_per_night * nights

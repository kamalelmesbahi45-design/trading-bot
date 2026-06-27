"""Fill simulator. Same surface as live execution so engine code path is identical.

Realistic XAUUSD assumptions:
    - MARKET BUY fills at ask + slippage_points * 0.01
    - MARKET SELL fills at bid - slippage_points * 0.01
    - Stops gap-through the level when price moves past it intra-bar
    - No partial fills (XAUUSD spot is plenty liquid for retail size)
"""
from __future__ import annotations

from xauusd_bot.backtest.costs import SpreadModel
from xauusd_bot.types import Bar, Fill, Order


class FillSimulator:
    def __init__(
        self,
        spread_model: SpreadModel,
        slippage_points: int,
        commission_per_lot: float,
        contract_size: int = 100,
    ) -> None:
        self.spread_model = spread_model
        self.slippage_points = slippage_points
        self.commission_per_lot = commission_per_lot
        self.contract_size = contract_size

    def fill_market(self, order: Order, bar: Bar) -> Fill:
        raise NotImplementedError

    def maybe_fill_sl_tp(self, position_side: str, sl: float | None, tp: float | None, bar: Bar) -> tuple[bool, float | None, str | None]:
        """Return (was_hit, fill_price, reason)."""
        raise NotImplementedError

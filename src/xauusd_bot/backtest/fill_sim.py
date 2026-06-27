"""Fill simulator. Same surface as live execution so engine code path is identical.

Convention:
    - MARKET BUY fills at  bar.open + spread/2 + slippage
    - MARKET SELL fills at bar.open - spread/2 - slippage
    - Intra-bar SL/TP: if bar.low <= sl <= bar.high (long stops), the stop is hit
      at the stop level (not the worst tick) -- this is the "conservative-but-fair"
      assumption used by most backtesters. When both SL and TP could hit in the
      same bar, we resolve pessimistically: SL hit first.
"""
from __future__ import annotations

from dataclasses import dataclass

from xauusd_bot.backtest.costs import POINT_VALUE, SpreadModel, commission
from xauusd_bot.types import Bar, Fill, Order, Side


@dataclass(frozen=True)
class StopHit:
    fill_price: float
    reason: str   # "sl" | "tp"


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

    def fill_market(self, order: Order, bar: Bar, in_event_window: bool = False) -> Fill:
        """Fill a MARKET order at `bar.open` with spread/2 + slippage applied."""
        sp = self.spread_model.spread(bar.ts, bar.open, in_event_window=in_event_window)
        slip = self.slippage_points * POINT_VALUE
        if order.side is Side.LONG:
            price = bar.open + sp * 0.5 + slip
        else:
            price = bar.open - sp * 0.5 - slip
        return Fill(
            ts=bar.ts,
            order_id=order.id or "",
            side=order.side,
            qty=order.qty,
            price=price,
            commission=commission(order.qty, self.commission_per_lot),
            slippage=slip,
        )

    def maybe_fill_sl_tp(
        self,
        side: Side,
        sl: float | None,
        tp: float | None,
        bar: Bar,
    ) -> StopHit | None:
        """Check intra-bar whether SL or TP was hit. Conservative: SL wins ties."""
        if side is Side.LONG:
            sl_hit = sl is not None and bar.low <= sl
            tp_hit = tp is not None and bar.high >= tp
            if sl_hit:
                return StopHit(fill_price=sl, reason="sl")  # type: ignore[arg-type]
            if tp_hit:
                return StopHit(fill_price=tp, reason="tp")  # type: ignore[arg-type]
            return None

        # SHORT
        sl_hit = sl is not None and bar.high >= sl
        tp_hit = tp is not None and bar.low <= tp
        if sl_hit:
            return StopHit(fill_price=sl, reason="sl")  # type: ignore[arg-type]
        if tp_hit:
            return StopHit(fill_price=tp, reason="tp")  # type: ignore[arg-type]
        return None

    def close_market(self, side: Side, qty: float, bar: Bar, in_event_window: bool = False) -> Fill:
        """Market exit for an existing position. Long closes at bid, short at ask."""
        sp = self.spread_model.spread(bar.ts, bar.open, in_event_window=in_event_window)
        slip = self.slippage_points * POINT_VALUE
        price = bar.open - sp * 0.5 - slip if side is Side.LONG else bar.open + sp * 0.5 + slip
        return Fill(
            ts=bar.ts,
            order_id="",
            side=side,
            qty=qty,
            price=price,
            commission=commission(qty, self.commission_per_lot),
            slippage=slip,
        )

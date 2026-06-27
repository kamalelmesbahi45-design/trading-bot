"""Stop-loss / take-profit policies."""
from __future__ import annotations

from dataclasses import dataclass

from xauusd_bot.types import Position, Side


@dataclass
class StopPolicy:
    atr_mult_sl: float
    atr_mult_tp: float
    breakeven_at_r: float
    trailing: bool

    def initial_levels(self, side: Side, entry: float, atr: float) -> tuple[float, float]:
        """Returns (sl_price, tp_price)."""
        raise NotImplementedError

    def update(self, pos: Position, mark: float, atr: float) -> tuple[float | None, float | None]:
        """Return new (sl, tp). None if unchanged."""
        raise NotImplementedError

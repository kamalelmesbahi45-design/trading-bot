"""Stop-loss / take-profit policies.

Initial: ATR-based SL/TP at entry.
Update : breakeven once unrealised PnL >= breakeven_at_r * 1R; optional ATR trailing
         that only ever tightens the stop, never loosens it.
"""
from __future__ import annotations

from dataclasses import dataclass

from xauusd_bot.types import Position, Side


@dataclass
class StopPolicy:
    atr_mult_sl: float
    atr_mult_tp: float
    breakeven_at_r: float = 1.0
    trailing: bool = True

    def initial_levels(self, side: Side, entry: float, atr: float) -> tuple[float, float]:
        """Return (sl_price, tp_price)."""
        if atr <= 0:
            raise ValueError("atr must be positive")
        sl_delta = self.atr_mult_sl * atr
        tp_delta = self.atr_mult_tp * atr
        if side is Side.LONG:
            return entry - sl_delta, entry + tp_delta
        return entry + sl_delta, entry - tp_delta

    def update(self, pos: Position, mark: float, atr: float) -> tuple[float | None, float | None]:
        """Return (new_sl, new_tp). Either may be None when unchanged.

        Rules:
          1. Once breakeven is hit (R-multiple >= breakeven_at_r), SL moves to entry
             (only on the favourable side).
          2. If trailing is enabled, SL tightens to (mark -/+ atr_mult_sl * atr) but
             never loosens.
          3. TP is left unchanged here (set at entry).
        """
        if atr <= 0:
            return None, None
        new_sl = pos.sl_price
        r = pos.r_multiple(mark)

        # Breakeven move
        if r is not None and r >= self.breakeven_at_r:
            be = pos.entry_price
            new_sl = self._tighten(pos.side, new_sl, be)

        # Trailing tightens with mark
        if self.trailing:
            trail = mark - self.atr_mult_sl * atr if pos.side is Side.LONG else mark + self.atr_mult_sl * atr
            new_sl = self._tighten(pos.side, new_sl, trail)

        # Only report change if it actually moved
        if new_sl == pos.sl_price:
            return None, None
        return new_sl, None

    @staticmethod
    def _tighten(side: Side, current: float | None, candidate: float) -> float:
        if current is None:
            return candidate
        return max(current, candidate) if side is Side.LONG else min(current, candidate)

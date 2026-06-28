"""Position sizing. Three sizers behind a common interface.

XAUUSD economics (assumed):
    1 standard lot = 100 oz.
    A $1 move in price * 1 lot = $100 PnL.
    Risk at stop (USD) = lots * stop_distance_price * contract_size.

All sizers return *lots* (broker-quantised by `lot_step`, default 0.01).
Negative or NaN inputs return 0.0.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass


@dataclass
class SizingInputs:
    equity: float
    stop_distance_price: float        # |entry - sl| in price units
    contract_size: int = 100          # XAUUSD = 100 oz / lot


def _quantise(lots: float, lot_step: float, lot_min: float) -> float:
    """Floor `lots` to the broker's lot step and clamp to the minimum lot size.

    Uses an epsilon nudge to absorb float artefacts like 2.0/0.01 == 199.999...
    """
    if not math.isfinite(lots) or lots <= 0:
        return 0.0
    stepped = math.floor(lots / lot_step + 1e-9) * lot_step
    if stepped + 1e-12 < lot_min:
        return 0.0
    return round(stepped, 4)


def _lots_for_risk(risk_usd: float, x: SizingInputs) -> float:
    """Convert a target USD risk to lots given XAUUSD economics."""
    denom = x.stop_distance_price * x.contract_size
    if denom <= 0:
        return 0.0
    return risk_usd / denom


class Sizer(ABC):
    name: str
    lot_step: float = 0.01
    lot_min: float = 0.01

    @abstractmethod
    def lots(self, x: SizingInputs) -> float:
        ...


class FixedFractional(Sizer):
    """Risk a fixed pct of equity per trade, capped by hard_cap."""
    name = "fixed_fractional"

    def __init__(self, pct_per_trade: float, hard_cap_pct: float,
                 lot_step: float = 0.01, lot_min: float = 0.01) -> None:
        self.pct_per_trade = pct_per_trade
        self.hard_cap_pct = hard_cap_pct
        self.lot_step = lot_step
        self.lot_min = lot_min

    def lots(self, x: SizingInputs) -> float:
        pct = min(self.pct_per_trade, self.hard_cap_pct)
        risk_usd = max(0.0, x.equity) * pct
        return _quantise(_lots_for_risk(risk_usd, x), self.lot_step, self.lot_min)


class FractionalKelly(Sizer):
    """Fractional Kelly from realised per-trade R-multiples.

    With R-multiples (loss = -1R), Kelly's optimal bet fraction is
        f* = p - (1 - p) / b
    where p = win rate and b = avg_win_R (the payout per unit risk).

    We multiply f* by `kelly_fraction` (e.g. 0.25), clamp to [0, hard_cap_pct],
    and fall back to FixedFractional(fallback_pct) until `min_trades` are in.
    Negative f* -> 0 lots (no trade), since the rolling edge is negative.
    """
    name = "fractional_kelly"

    def __init__(
        self,
        kelly_fraction: float,
        fallback_pct: float,
        hard_cap_pct: float,
        min_trades: int,
        lookback_trades: int,
        lot_step: float = 0.01,
        lot_min: float = 0.01,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.fallback_pct = fallback_pct
        self.hard_cap_pct = hard_cap_pct
        self.min_trades = min_trades
        self.lookback_trades = lookback_trades
        self.lot_step = lot_step
        self.lot_min = lot_min
        self._history: deque[float] = deque(maxlen=max(lookback_trades, min_trades))

    def update_stats(self, trade_returns_r: list[float]) -> None:
        """Append realised R-multiples (last N kept)."""
        for r in trade_returns_r:
            if math.isfinite(r):
                self._history.append(r)

    def _current_kelly(self) -> float | None:
        if len(self._history) < self.min_trades:
            return None
        wins = [r for r in self._history if r > 0]
        n = len(self._history)
        if n == 0 or not wins:
            return 0.0
        p = len(wins) / n
        b = sum(wins) / len(wins)
        if b <= 0:
            return 0.0
        f_star = p - (1.0 - p) / b
        return max(0.0, f_star)  # negative edge -> stand down

    def lots(self, x: SizingInputs) -> float:
        f = self._current_kelly()
        pct = self.fallback_pct if f is None else self.kelly_fraction * f
        pct = min(pct, self.hard_cap_pct)
        risk_usd = max(0.0, x.equity) * pct
        return _quantise(_lots_for_risk(risk_usd, x), self.lot_step, self.lot_min)


class VolTarget(Sizer):
    """Size to a target annualised PnL volatility.

    Requires a recent realised-returns series to estimate per-trade vol; left as a
    documented stub for v1. Configs default to fractional_kelly.
    """
    name = "vol_target"

    def __init__(self, target_annual_vol: float, hard_cap_pct: float,
                 lot_step: float = 0.01, lot_min: float = 0.01) -> None:
        self.target_annual_vol = target_annual_vol
        self.hard_cap_pct = hard_cap_pct
        self.lot_step = lot_step
        self.lot_min = lot_min

    def lots(self, x: SizingInputs) -> float:
        raise NotImplementedError(
            "VolTarget needs a rolling returns series to estimate per-trade vol; "
            "not in v1. Use FractionalKelly or FixedFractional."
        )

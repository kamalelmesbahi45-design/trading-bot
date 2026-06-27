"""Position sizing. Three sizers behind a common interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SizingInputs:
    equity: float
    stop_distance_price: float      # price distance to stop (in price units, not pips)
    contract_size: int              # XAUUSD = 100 oz / lot
    pip_value_per_lot: float = 1.0  # for XAUUSD: $1 per 0.01 move per lot at 1 oz... see fill_sim


class Sizer(ABC):
    name: str

    @abstractmethod
    def lots(self, x: SizingInputs) -> float:
        ...


class FixedFractional(Sizer):
    name = "fixed_fractional"

    def __init__(self, pct_per_trade: float, hard_cap_pct: float) -> None:
        self.pct_per_trade = pct_per_trade
        self.hard_cap_pct = hard_cap_pct

    def lots(self, x: SizingInputs) -> float:
        raise NotImplementedError


class FractionalKelly(Sizer):
    """Kelly fraction estimated from rolling per-strategy trade stats.

    f* = W/A - L/B  where W=win_rate, L=1-W, A=avg_win_R, B=avg_loss_R.
    We size lots so that loss_at_stop = equity * fraction * f*, capped by hard_cap.
    Only activates after `min_trades`; before that we fall back to FixedFractional.
    """
    name = "fractional_kelly"

    def __init__(
        self,
        kelly_fraction: float,
        fallback_pct: float,
        hard_cap_pct: float,
        min_trades: int,
        lookback_trades: int,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.fallback_pct = fallback_pct
        self.hard_cap_pct = hard_cap_pct
        self.min_trades = min_trades
        self.lookback_trades = lookback_trades

    def update_stats(self, trade_returns_r: list[float]) -> None:
        """Feed the sizer recent realised trade R-multiples to recompute Kelly."""
        raise NotImplementedError

    def lots(self, x: SizingInputs) -> float:
        raise NotImplementedError


class VolTarget(Sizer):
    """Size to a target annualised PnL volatility."""
    name = "vol_target"

    def __init__(self, target_annual_vol: float, hard_cap_pct: float) -> None:
        self.target_annual_vol = target_annual_vol
        self.hard_cap_pct = hard_cap_pct

    def lots(self, x: SizingInputs) -> float:
        raise NotImplementedError

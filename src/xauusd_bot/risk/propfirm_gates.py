"""Prop-firm guardrails. These can VETO a trade or KILL the bot.

Decisions:
    ALLOW : new trade is fine.
    VETO  : the proposed trade's worst case (immediate stop-out) would breach a
            daily- or total-drawdown limit. Skip this trade; bot keeps running.
    KILL  : the account is already past a hard limit, or has been killed before.
            Flatten everything and stop opening new trades until manual reset.

GateState carries the bookkeeping (HWM, day-start equity, trading days, killed
flag). Engine is expected to call `on_day_start` once per trading day and
`on_equity_update` whenever equity changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class GateDecision(StrEnum):
    ALLOW = "allow"
    VETO = "veto"
    KILL = "kill"


@dataclass
class GateState:
    starting_equity: float
    high_water_mark: float = 0.0
    day_start_equity: dict[date, float] = field(default_factory=dict)
    trading_days: set[date] = field(default_factory=set)
    killed: bool = False
    kill_reason: str | None = None

    def __post_init__(self) -> None:
        if self.high_water_mark <= 0.0:
            self.high_water_mark = self.starting_equity

    def on_day_start(self, day: date, equity: float) -> None:
        self.day_start_equity.setdefault(day, equity)

    def on_equity_update(self, equity: float) -> None:
        if equity > self.high_water_mark:
            self.high_water_mark = equity

    def on_trade_day(self, day: date) -> None:
        self.trading_days.add(day)

    def mark_killed(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason


@dataclass
class PropFirmGates:
    daily_loss_limit_pct: float | None
    max_drawdown_pct: float
    profit_target_pct: float | None = None
    min_trading_days: int | None = None

    def evaluate(
        self,
        now: datetime,
        current_equity: float,
        proposed_risk_amount: float,
        state: GateState,
    ) -> GateDecision:
        """Decide ALLOW/VETO/KILL for a proposed new trade at `now`."""
        if state.killed:
            return GateDecision.KILL

        # Total DD already breached?
        total_dd = (current_equity - state.starting_equity) / state.starting_equity
        if total_dd <= -self.max_drawdown_pct:
            state.mark_killed(f"max_drawdown breach: dd={total_dd:.4f}")
            return GateDecision.KILL

        # Daily already breached?
        if self.daily_loss_limit_pct is not None:
            day = now.date()
            day_start = state.day_start_equity.get(day, current_equity)
            day_pnl = (current_equity - day_start) / day_start if day_start > 0 else 0.0
            if day_pnl <= -self.daily_loss_limit_pct:
                state.mark_killed(f"daily_loss breach: pnl={day_pnl:.4f}")
                return GateDecision.KILL

            # Worst-case daily after this trade gets stopped out
            worst_eq = current_equity - max(0.0, proposed_risk_amount)
            worst_day = (worst_eq - day_start) / day_start if day_start > 0 else 0.0
            if worst_day < -self.daily_loss_limit_pct:
                return GateDecision.VETO

        # Worst-case total DD after stop-out
        worst_eq_total = current_equity - max(0.0, proposed_risk_amount)
        worst_total = (worst_eq_total - state.starting_equity) / state.starting_equity
        if worst_total < -self.max_drawdown_pct:
            return GateDecision.VETO

        return GateDecision.ALLOW

    def profit_target_hit(self, current_equity: float, state: GateState) -> bool:
        """For the challenge: are we at the profit target with enough trading days?"""
        if self.profit_target_pct is None:
            return False
        gain = (current_equity - state.starting_equity) / state.starting_equity
        if gain < self.profit_target_pct:
            return False
        return not (
            self.min_trading_days is not None
            and len(state.trading_days) < self.min_trading_days
        )

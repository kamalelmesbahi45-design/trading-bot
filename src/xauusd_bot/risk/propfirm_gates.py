"""Prop-firm guardrails. These can VETO a trade or KILL the bot.

Veto: new trade would push intraday equity below daily loss limit, or beyond max DD.
Kill: equity is already past hard limits -> flatten all + disable bot until manual reset.
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
    trading_days_count: int = 0
    killed: bool = False


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
        raise NotImplementedError

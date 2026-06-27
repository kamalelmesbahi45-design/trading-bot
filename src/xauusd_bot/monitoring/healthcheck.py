"""Healthcheck: heartbeat + invariant checks for the live trading loop.

Invariants:
    - last tick within `max_tick_age_s` seconds
    - bot-equity matches broker-equity within tolerance
    - no orphan positions (bot vs broker disagree)
    - no orders stuck in PENDING for > `max_pending_age_s`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HealthCheckInputs:
    now: datetime
    last_tick_ts: datetime | None
    bot_equity: float
    broker_equity: float
    bot_position_count: int
    broker_position_count: int
    pending_order_ages_s: list[float] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    healthy: bool
    last_tick_age_s: float
    equity_diff: float
    orphan_positions: int
    pending_orders: int
    issues: list[str]


def check(
    inp: HealthCheckInputs,
    max_tick_age_s: float = 60.0,
    equity_tolerance: float = 1.0,
    max_pending_age_s: float = 30.0,
) -> HealthCheckResult:
    issues: list[str] = []

    if inp.last_tick_ts is None:
        last_age = float("inf")
        issues.append("no_tick_yet")
    else:
        last_age = (inp.now - inp.last_tick_ts).total_seconds()
        if last_age > max_tick_age_s:
            issues.append(f"stale_tick:{last_age:.1f}s")

    eq_diff = inp.bot_equity - inp.broker_equity
    if abs(eq_diff) > equity_tolerance:
        issues.append(f"equity_mismatch:{eq_diff:+.2f}")

    orphan = abs(inp.bot_position_count - inp.broker_position_count)
    if orphan > 0:
        issues.append(f"orphan_positions:{orphan}")

    stuck = sum(1 for a in inp.pending_order_ages_s if a > max_pending_age_s)
    if stuck:
        issues.append(f"stuck_pending:{stuck}")

    return HealthCheckResult(
        healthy=not issues,
        last_tick_age_s=last_age,
        equity_diff=eq_diff,
        orphan_positions=orphan,
        pending_orders=stuck,
        issues=issues,
    )

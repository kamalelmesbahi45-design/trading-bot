"""Healthcheck: heartbeat + invariant checks.

Invariants:
    - last tick within N seconds
    - equity matches MT5 reported equity within tolerance
    - no orphan positions (engine vs broker disagree)
    - no orders stuck in PENDING > N seconds
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class HealthCheckResult:
    healthy: bool
    last_tick_age_s: float
    equity_diff: float
    orphan_positions: int
    pending_orders: int
    issues: list[str]


def check(now: datetime) -> HealthCheckResult:
    raise NotImplementedError

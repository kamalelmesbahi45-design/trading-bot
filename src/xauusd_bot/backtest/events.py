"""Event types for the engine queue."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from xauusd_bot.types import Bar, Fill, Order, Tick


@dataclass(frozen=True)
class BarEvent:
    ts: datetime
    bar: Bar


@dataclass(frozen=True)
class TickEvent:
    ts: datetime
    tick: Tick


@dataclass(frozen=True)
class OrderEvent:
    ts: datetime
    order: Order


@dataclass(frozen=True)
class FillEvent:
    ts: datetime
    fill: Fill


@dataclass(frozen=True)
class TimerEvent:
    """Periodic tick for housekeeping: stop updates, EOD checks, regime refresh."""
    ts: datetime
    kind: str

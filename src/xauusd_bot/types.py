"""Core domain types shared across modules. Keep this layer dependency-free."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Tick:
    ts: datetime
    bid: float
    ask: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) * 0.5

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Signal:
    """A strategy's intent. The portfolio/risk layer turns this into an Order."""
    ts: datetime
    strategy: str
    side: Side
    confidence: float = 1.0           # 0..1, used by ML filter / allocator
    sl_price: float | None = None
    tp_price: float | None = None
    meta: dict[str, float] = field(default_factory=dict)


@dataclass
class Order:
    ts: datetime
    strategy: str
    side: Side
    type: OrderType
    qty: float                        # lots
    price: float | None = None        # None for market
    sl_price: float | None = None
    tp_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    id: str | None = None


@dataclass
class Fill:
    ts: datetime
    order_id: str
    side: Side
    qty: float
    price: float
    commission: float
    slippage: float


@dataclass
class Position:
    strategy: str
    side: Side
    qty: float
    entry_price: float
    entry_ts: datetime
    sl_price: float | None = None
    tp_price: float | None = None

    def unrealized_pnl(self, mark: float, contract_size: int) -> float:
        sign = 1 if self.side is Side.LONG else -1
        return sign * (mark - self.entry_price) * self.qty * contract_size


Timeframe = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

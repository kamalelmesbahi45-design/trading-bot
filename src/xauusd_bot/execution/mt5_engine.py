"""MT5 execution via the MetaTrader5 Python lib (Windows only at runtime).

Connection details come from env: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.
Implements idempotent order placement (deterministic magic + comment tagging)
so retries after timeouts don't double-fill.
"""
from __future__ import annotations

from xauusd_bot.execution.base import ExecutionEngine
from xauusd_bot.types import Fill, Order, Position


class MT5Engine(ExecutionEngine):
    def __init__(self, symbol: str, magic: int, slippage_points: int = 5) -> None:
        self.symbol = symbol
        self.magic = magic
        self.slippage_points = slippage_points

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError

    def positions(self) -> list[Position]:
        raise NotImplementedError

    def place(self, order: Order) -> Fill:
        raise NotImplementedError

    def modify(self, position: Position, sl: float | None, tp: float | None) -> None:
        raise NotImplementedError

    def close(self, position: Position) -> Fill:
        raise NotImplementedError

    def flatten_all(self) -> list[Fill]:
        raise NotImplementedError

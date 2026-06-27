"""Paper execution. Reuses backtest fill simulator against live tick feed.

State persisted in JSON so a crash/restart doesn't lose open positions.
"""
from __future__ import annotations

from pathlib import Path

from xauusd_bot.execution.base import ExecutionEngine
from xauusd_bot.types import Fill, Order, Position


class PaperEngine(ExecutionEngine):
    def __init__(self, starting_equity: float, state_path: Path) -> None:
        self.starting_equity = starting_equity
        self.state_path = state_path
        self._equity = starting_equity
        self._positions: list[Position] = []

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def equity(self) -> float:
        return self._equity

    def positions(self) -> list[Position]:
        return list(self._positions)

    def place(self, order: Order) -> Fill:
        raise NotImplementedError

    def modify(self, position: Position, sl: float | None, tp: float | None) -> None:
        raise NotImplementedError

    def close(self, position: Position) -> Fill:
        raise NotImplementedError

    def flatten_all(self) -> list[Fill]:
        raise NotImplementedError

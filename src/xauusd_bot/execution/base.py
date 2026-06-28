"""ExecutionEngine ABC. Live and paper conform to the same surface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from xauusd_bot.types import Fill, Order, Position


class ExecutionEngine(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def place(self, order: Order) -> Fill: ...

    @abstractmethod
    def modify(self, position: Position, sl: float | None, tp: float | None) -> None: ...

    @abstractmethod
    def close(self, position: Position) -> Fill: ...

    @abstractmethod
    def flatten_all(self) -> list[Fill]: ...

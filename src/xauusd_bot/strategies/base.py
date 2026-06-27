"""Strategy ABC. Implementations consume bars (and optional context) and emit Signals."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from xauusd_bot.types import Signal


@dataclass
class StrategyContext:
    """Everything a strategy might need beyond its own bars: cross-asset features, regime, macro score."""
    cross_asset: pd.DataFrame | None = None
    regime: pd.DataFrame | None = None
    macro_score: pd.Series | None = None


class Strategy(ABC):
    """Stateless strategy: bars in, list of signals out.

    Implementations MUST be look-ahead-free: at bar i they may only use rows[:i+1].
    """

    name: str

    @abstractmethod
    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        ...

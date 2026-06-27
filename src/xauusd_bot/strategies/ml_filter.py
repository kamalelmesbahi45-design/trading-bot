"""ML signal-quality filter.

Wraps any upstream Strategy. For each generated Signal, builds a feature vector
(price action, regime, cross-asset) and queries an XGBoost/LightGBM model that
predicts P(trade is profitable beyond N R-multiples). If P < threshold, drop.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Signal


class MLFilter(Strategy):
    name = "ml_filter"

    def __init__(self, inner: Strategy, model_path: Path, threshold: float = 0.55) -> None:
        self.inner = inner
        self.model_path = model_path
        self.threshold = threshold
        self._model = None  # lazy-loaded

    def _load(self) -> None:
        raise NotImplementedError

    def _featurise(self, bars: pd.DataFrame, sig: Signal, ctx: StrategyContext | None) -> pd.Series:
        raise NotImplementedError

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        raise NotImplementedError

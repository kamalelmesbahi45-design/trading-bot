"""Performance metrics. All produced from an equity curve + trade log."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PerformanceStats:
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    avg_R: float
    expectancy_R: float
    profit_factor: float
    n_trades: int
    deflated_sharpe: float | None = None


def compute_stats(equity_curve: pd.Series, trades: pd.DataFrame) -> PerformanceStats:
    raise NotImplementedError


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float:
    """Bailey & López de Prado deflated Sharpe ratio. Corrects for multiple-testing inflation."""
    raise NotImplementedError

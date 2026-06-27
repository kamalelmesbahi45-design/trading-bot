"""Monte Carlo stress tests.

Two complementary methods:
    1. TRADE SHUFFLE: permute the order of realised trades N times. Tests if the
       observed drawdown is path-dependent or luck.
    2. BOOTSTRAP RESAMPLE: sample-with-replacement from trade R-multiples N times,
       build synthetic equity curves of same length. Tests robustness of CAGR/DD.

Outputs:
    - Distribution of max DD, CAGR, Sharpe
    - Probability of ruin (curve breaches max_drawdown_pct)
    - 95th percentile drawdown
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MonteCarloResult:
    dd_distribution: pd.Series
    cagr_distribution: pd.Series
    sharpe_distribution: pd.Series
    p_ruin: float
    p95_drawdown: float


def shuffle_trades(trades: pd.DataFrame, n_runs: int = 10_000, seed: int = 0) -> MonteCarloResult:
    raise NotImplementedError


def bootstrap_trades(trades: pd.DataFrame, n_runs: int = 10_000, n_per_run: int | None = None, seed: int = 0) -> MonteCarloResult:
    raise NotImplementedError

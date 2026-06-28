"""Monte Carlo stress tests on trade history.

Two methods:
    shuffle_trades  : permute the realised trade order N times. Tests whether the
                      observed drawdown is path-dependent or just unlucky/ordering.
    bootstrap_trades: sample-with-replacement from realised R-multiples N times to
                      build synthetic equity curves of the same length. Tests
                      robustness of CAGR / max DD / Sharpe to the trade sample.

Outputs distributions for max DD, CAGR, Sharpe, plus probability of ruin
(curve breaches `ruin_dd_pct`) and 95th-percentile drawdown.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    dd_distribution: pd.Series
    cagr_distribution: pd.Series
    sharpe_distribution: pd.Series
    p_ruin: float
    p95_drawdown: float


def _equity_from_pnl(starting_equity: float, pnl: np.ndarray) -> np.ndarray:
    return starting_equity + np.cumsum(pnl)


def _max_dd_from_equity(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))  # type: ignore[no-any-return]


def _cagr_from_equity(equity: np.ndarray, n_periods: int, periods_per_year: float) -> float:
    if n_periods <= 0 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def _sharpe_from_returns(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _runs(
    trades: pd.DataFrame,
    n_runs: int,
    seed: int,
    starting_equity: float,
    ruin_dd_pct: float,
    sampler: Callable[[np.random.Generator, np.ndarray, int], np.ndarray],
    periods_per_year: float,
) -> MonteCarloResult:
    if trades.empty:
        empty = pd.Series(dtype=float)
        return MonteCarloResult(empty, empty, empty, 0.0, 0.0)
    pnl = trades["pnl"].astype(float).to_numpy()
    n_trades = len(pnl)
    rng = np.random.default_rng(seed)

    dds = np.zeros(n_runs)
    cagrs = np.zeros(n_runs)
    sharpes = np.zeros(n_runs)
    n_ruined = 0

    for k in range(n_runs):
        sample = sampler(rng, pnl, n_trades)
        eq = _equity_from_pnl(starting_equity, sample)
        dd = _max_dd_from_equity(eq)
        dds[k] = dd
        if dd <= -ruin_dd_pct:
            n_ruined += 1
        cagrs[k] = _cagr_from_equity(eq, n_trades, periods_per_year)
        prev = eq[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(prev > 0, np.diff(eq) / np.where(prev > 0, prev, 1.0), 0.0)
        sharpes[k] = _sharpe_from_returns(rets, periods_per_year)

    return MonteCarloResult(
        dd_distribution=pd.Series(dds, name="max_dd"),
        cagr_distribution=pd.Series(cagrs, name="cagr"),
        sharpe_distribution=pd.Series(sharpes, name="sharpe"),
        p_ruin=float(n_ruined / n_runs),
        p95_drawdown=float(np.quantile(dds, 0.05)),  # worst 5% DDs (lower bound)
    )


def shuffle_trades(
    trades: pd.DataFrame,
    n_runs: int = 10_000,
    seed: int = 0,
    starting_equity: float = 10_000.0,
    ruin_dd_pct: float = 0.20,
    periods_per_year: float = 252.0,
) -> MonteCarloResult:
    def sampler(rng: np.random.Generator, pnl: np.ndarray, n: int) -> np.ndarray:
        return rng.permutation(pnl)
    return _runs(trades, n_runs, seed, starting_equity, ruin_dd_pct, sampler, periods_per_year)


def bootstrap_trades(
    trades: pd.DataFrame,
    n_runs: int = 10_000,
    n_per_run: int | None = None,
    seed: int = 0,
    starting_equity: float = 10_000.0,
    ruin_dd_pct: float = 0.20,
    periods_per_year: float = 252.0,
) -> MonteCarloResult:
    target_n = n_per_run

    def sampler(rng: np.random.Generator, pnl: np.ndarray, n: int) -> np.ndarray:
        size = target_n if target_n is not None else n
        return rng.choice(pnl, size=size, replace=True)
    return _runs(trades, n_runs, seed, starting_equity, ruin_dd_pct, sampler, periods_per_year)

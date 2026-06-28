"""Monte Carlo distribution tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.optimize.monte_carlo import (
    MonteCarloResult,
    bootstrap_trades,
    shuffle_trades,
)


def _toy_trades(n: int = 100, win_p: float = 0.55, win_r: float = 1.5, loss_r: float = -1.0,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pnl = np.where(rng.random(n) < win_p, win_r, loss_r) * 100.0  # R in dollar terms
    return pd.DataFrame({"pnl": pnl, "r_multiple": pnl / 100.0})


def test_shuffle_deterministic_with_seed() -> None:
    trades = _toy_trades(50, seed=42)
    a = shuffle_trades(trades, n_runs=100, seed=1, starting_equity=10_000.0)
    b = shuffle_trades(trades, n_runs=100, seed=1, starting_equity=10_000.0)
    pd.testing.assert_series_equal(a.dd_distribution, b.dd_distribution)


def test_shuffle_distributions_have_correct_length() -> None:
    trades = _toy_trades(30, seed=0)
    res = shuffle_trades(trades, n_runs=200, seed=0, starting_equity=10_000.0)
    assert isinstance(res, MonteCarloResult)
    assert len(res.dd_distribution) == 200
    assert len(res.cagr_distribution) == 200
    assert len(res.sharpe_distribution) == 200


def test_bootstrap_p_ruin_is_high_with_tight_limit() -> None:
    # Trades that go heavily negative; small starting equity -> high ruin probability
    trades = pd.DataFrame({"pnl": [-200.0] * 20, "r_multiple": [-1.0] * 20})
    res = bootstrap_trades(trades, n_runs=500, seed=0, starting_equity=1_000.0,
                            ruin_dd_pct=0.10)
    assert res.p_ruin >= 0.95


def test_bootstrap_p_ruin_is_low_with_strongly_positive_trades() -> None:
    trades = pd.DataFrame({"pnl": [200.0] * 20, "r_multiple": [1.5] * 20})
    res = bootstrap_trades(trades, n_runs=500, seed=0, starting_equity=10_000.0,
                            ruin_dd_pct=0.20)
    assert res.p_ruin == 0.0


def test_empty_trades_returns_zero_metrics() -> None:
    res = shuffle_trades(pd.DataFrame(columns=["pnl", "r_multiple"]), n_runs=10, seed=0)
    assert res.p_ruin == 0.0
    assert res.dd_distribution.empty


def test_p95_drawdown_is_in_range() -> None:
    trades = _toy_trades(50, win_p=0.5)
    res = shuffle_trades(trades, n_runs=300, seed=3, starting_equity=10_000.0)
    # p95 is the 5th percentile of (negative) dd values
    assert -1.0 <= res.p95_drawdown <= 0.0

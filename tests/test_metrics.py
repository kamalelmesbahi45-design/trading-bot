"""Performance metric tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.optimize.metrics import (
    cagr,
    compute_stats,
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
)


def _curve(values: list[float], freq: str = "1D") -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_max_drawdown_basic() -> None:
    c = _curve([100.0, 110.0, 105.0, 90.0, 120.0])
    # peak = [100, 110, 110, 110, 120]; dd = [0, 0, -4.5%, -18.18%, 0]; min = -18.18%
    assert abs(max_drawdown(c) - (90.0 / 110.0 - 1.0)) < 1e-9


def test_max_drawdown_empty_curve() -> None:
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


def test_cagr_on_two_year_doubling() -> None:
    idx = pd.date_range("2022-01-01", periods=3, freq="365D", tz="UTC")
    c = pd.Series([10000.0, 14142.13, 20000.0], index=idx)
    g = cagr(c)
    # roughly 41% CAGR over 2 years to double; allow some tolerance for 365d span
    assert 0.35 < g < 0.50


def test_sharpe_zero_when_returns_constant() -> None:
    c = _curve([100.0] * 10)
    assert sharpe(c) == 0.0


def test_sharpe_positive_on_rising_curve() -> None:
    # Drift large enough to dominate seed noise -> realised Sharpe well > 1
    rng = np.random.default_rng(7)
    rets = 0.005 + rng.normal(0, 0.01, 252)
    eq = pd.Series(np.cumprod(1 + rets) * 10_000, index=pd.date_range("2024-01-01", periods=252, freq="1D", tz="UTC"))
    s = sharpe(eq)
    assert s > 1.0


def test_sortino_zero_when_no_downside() -> None:
    c = _curve([100.0, 101.0, 102.0, 103.0, 104.0])
    assert sortino(c) == 0.0


def test_deflated_sharpe_returns_finite_probability() -> None:
    d = deflated_sharpe(sharpe_obs=1.5, n_trials=50, n_obs=252, skew=-0.5, kurt=4.0)
    assert 0.0 <= d <= 1.0


def test_deflated_sharpe_handles_zero_obs() -> None:
    import math
    d = deflated_sharpe(sharpe_obs=1.0, n_trials=10, n_obs=0, skew=0.0, kurt=3.0)
    assert math.isnan(d)


def test_compute_stats_empty_inputs() -> None:
    s = compute_stats(pd.Series(dtype=float), pd.DataFrame())
    assert s.n_trades == 0


def test_compute_stats_with_trades() -> None:
    c = _curve([10_000.0 + i * 10 for i in range(30)])
    trades = pd.DataFrame({
        "pnl":        [50.0, -30.0, 80.0, -40.0, 60.0],
        "r_multiple": [1.0, -1.0, 1.5, -1.0, 1.2],
    })
    s = compute_stats(c, trades)
    assert s.n_trades == 5
    assert abs(s.win_rate - 0.6) < 1e-9
    # profit factor = 190 / 70
    assert abs(s.profit_factor - 190.0 / 70.0) < 1e-9

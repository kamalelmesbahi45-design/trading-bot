"""HTML report and plot smoke tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_bot.optimize.monte_carlo import shuffle_trades
from xauusd_bot.reports.html_report import render_report
from xauusd_bot.reports.plots import (
    drawdown_fig,
    equity_curve_fig,
    mc_distribution_fig,
    monthly_returns_heatmap,
    per_strategy_contribution,
)


def _curve_and_trades():
    idx = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0008, 0.01, 200)
    eq = pd.Series(10_000.0 * np.cumprod(1 + returns), index=idx)
    trades = pd.DataFrame({
        "pnl":       [50.0, -30.0, 80.0, -40.0, 60.0, -20.0, 100.0],
        "r_multiple":[1.0, -1.0, 1.5, -1.0, 1.2, -1.0, 2.0],
        "strategy":  ["t","t","t","mr","mr","brk","brk"],
    })
    return eq, trades


def test_plots_build_without_error() -> None:
    eq, trades = _curve_and_trades()
    assert equity_curve_fig(eq).data[0].mode == "lines"
    assert drawdown_fig(eq).data[0].mode == "lines"
    monthly_returns_heatmap(eq)   # no exception
    per_strategy_contribution(trades)


def test_plots_handle_empty_inputs() -> None:
    empty = pd.Series(dtype=float)
    fig = equity_curve_fig(empty)
    assert fig is not None
    assert per_strategy_contribution(pd.DataFrame()) is not None
    assert mc_distribution_fig(pd.Series(dtype=float), "x") is not None
    assert monthly_returns_heatmap(empty) is not None


def test_render_report_writes_html(tmp_path: Path) -> None:
    eq, trades = _curve_and_trades()
    mc = shuffle_trades(trades, n_runs=100, seed=0, starting_equity=10_000.0)
    out = render_report(tmp_path, eq, trades, mc_result=mc)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
    assert "Summary" in content
    assert "Probability of ruin" in content


def test_render_report_without_mc(tmp_path: Path) -> None:
    eq, trades = _curve_and_trades()
    out = render_report(tmp_path, eq, trades)
    assert out.exists()
    assert "Probability of ruin" not in out.read_text(encoding="utf-8")

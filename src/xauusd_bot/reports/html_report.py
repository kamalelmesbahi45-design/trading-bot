"""Render the full HTML report from a backtest / WFO / MC result bundle.

Sections:
    - Summary stats (CAGR, Sharpe, Sortino, Calmar, max DD, profit factor, deflated Sharpe)
    - Equity curve
    - Drawdown
    - Monthly returns heatmap
    - Per-strategy contribution
    - Trade distribution (R-multiples, hold times, win/loss)
    - Walk-forward OOS curve + parameter stability
    - Monte Carlo distributions + probability of ruin
    - Regime breakdown (performance by quant_regime label)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_report(
    out_dir: Path,
    equity: pd.Series,
    trades: pd.DataFrame,
    stats: object,
    wfo_result: object | None = None,
    mc_result: object | None = None,
    title: str = "XAUUSD bot report",
) -> Path:
    """Render to <out_dir>/report.html. Returns the file path."""
    raise NotImplementedError

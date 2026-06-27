"""Plotly figures used by the HTML report."""
from __future__ import annotations

import pandas as pd


def equity_curve_fig(equity: pd.Series) -> object:
    raise NotImplementedError


def drawdown_fig(equity: pd.Series) -> object:
    raise NotImplementedError


def monthly_returns_heatmap(equity: pd.Series) -> object:
    raise NotImplementedError


def per_strategy_contribution(trades: pd.DataFrame) -> object:
    raise NotImplementedError


def mc_distribution_fig(distribution: pd.Series, title: str) -> object:
    raise NotImplementedError


def param_stability_heatmap(param_history: pd.DataFrame) -> object:
    raise NotImplementedError

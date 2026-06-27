"""Plotly figures used by the HTML report. Returns Figure objects, not HTML."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def equity_curve_fig(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="equity"))
    fig.update_layout(title=title, xaxis_title="time", yaxis_title="equity (USD)",
                      template="plotly_white", height=400)
    return fig


def drawdown_fig(equity: pd.Series, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        peak = equity.cummax()
        dd = (equity / peak - 1.0) * 100.0
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="dd %",
                                 fill="tozeroy"))
    fig.update_layout(title=title, xaxis_title="time", yaxis_title="drawdown (%)",
                      template="plotly_white", height=300)
    return fig


def monthly_returns_heatmap(equity: pd.Series, title: str = "Monthly Returns") -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        fig.update_layout(title=title, template="plotly_white", height=300)
        return fig
    monthly = equity.resample("ME").last().pct_change().dropna() * 100.0
    if monthly.empty:
        fig.update_layout(title=title, template="plotly_white", height=300)
        return fig
    table = monthly.copy()
    df = pd.DataFrame({
        "year": table.index.year,
        "month": table.index.month,
        "ret": table.values,
    })
    pivot = df.pivot_table(index="year", columns="month", values="ret")
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=[str(m) for m in pivot.columns], y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0,
        texttemplate="%{z:.1f}%", colorbar={"title": "%"},
    ))
    fig.update_layout(title=title, xaxis_title="month", yaxis_title="year",
                      template="plotly_white", height=380)
    return fig


def per_strategy_contribution(trades: pd.DataFrame, title: str = "P&L by Strategy") -> go.Figure:
    fig = go.Figure()
    if trades is None or trades.empty or "strategy" not in trades.columns:
        fig.update_layout(title=title, template="plotly_white", height=300)
        return fig
    agg = trades.groupby("strategy")["pnl"].sum().sort_values()
    fig.add_trace(go.Bar(x=agg.values, y=agg.index, orientation="h"))
    fig.update_layout(title=title, xaxis_title="net pnl (USD)", yaxis_title="strategy",
                      template="plotly_white", height=300)
    return fig


def mc_distribution_fig(distribution: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if distribution.empty:
        fig.update_layout(title=title, template="plotly_white", height=300)
        return fig
    fig.add_trace(go.Histogram(x=distribution.values, nbinsx=50))
    fig.update_layout(title=title, xaxis_title=str(distribution.name or ""),
                      yaxis_title="count", template="plotly_white", height=300)
    return fig


def param_stability_heatmap(param_history: pd.DataFrame, title: str = "Parameter Stability") -> go.Figure:
    fig = go.Figure()
    if param_history is None or param_history.empty:
        fig.update_layout(title=title, template="plotly_white", height=300)
        return fig
    fig.add_trace(go.Heatmap(
        z=param_history.values, x=list(param_history.columns), y=[str(i) for i in param_history.index],
    ))
    fig.update_layout(title=title, xaxis_title="param", yaxis_title="fold",
                      template="plotly_white", height=400)
    return fig

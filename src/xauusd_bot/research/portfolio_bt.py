"""Vectorised multi-asset daily-rebalanced portfolio backtest.

Convention (no look-ahead): weights decided at close of day t are held over day
t+1, so realised PnL on day t+1 = weight_t * return_{t+1}. Costs are charged on
the change in weights (turnover) at each rebalance.

Cost model (round-trip, in return space, per unit of |weight| traded):
    cost = turnover * cost_bps / 1e4
Default 5 bps is realistic for liquid ETFs; crypto/some FX a touch higher but
we keep one number for transparency and stress it in the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    per_asset_pnl: pd.DataFrame
    turnover: pd.Series
    metadata: dict[str, float] = field(default_factory=dict)


def run_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    starting_equity: float = 100_000.0,
    cost_bps: float = 5.0,
    rebalance: str = "D",
) -> PortfolioResult:
    """Backtest a weight matrix against a returns matrix.

    weights, returns: same index/columns. weights at t applied to returns at t+1.
    rebalance: 'D' daily, 'W' weekly (hold weights between rebalance dates to cut
    turnover/costs).
    """
    weights = weights.reindex_like(returns).fillna(0.0)

    if rebalance.upper().startswith("W"):
        # Hold the most recent rebalance weights through the week.
        mask = weights.index.to_series().dt.dayofweek == 0  # Monday rebalance
        held = weights.where(mask).ffill().fillna(0.0)
        eff_weights = held
    else:
        eff_weights = weights

    # PnL on day t+1 from weights at t
    lagged = eff_weights.shift(1).fillna(0.0)
    per_asset = lagged * returns.fillna(0.0)
    gross = per_asset.sum(axis=1)

    # Turnover = sum |w_t - w_{t-1}|; cost charged that day
    turnover = (eff_weights - eff_weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 1e4)
    net = gross - costs

    equity = starting_equity * (1.0 + net).cumprod()

    ann_ret = float((1.0 + net.mean()) ** TRADING_DAYS - 1.0)
    ann_vol = float(net.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = equity.cummax()
    max_dd = float((equity / peak - 1.0).min())

    return PortfolioResult(
        equity_curve=equity,
        gross_returns=gross,
        net_returns=net,
        weights=eff_weights,
        per_asset_pnl=per_asset,
        turnover=turnover,
        metadata={
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_turnover": float(turnover.mean()),
            "final_equity": float(equity.iloc[-1]) if len(equity) else starting_equity,
            "cost_bps": cost_bps,
        },
    )


def equal_weight_buy_hold(
    returns: pd.DataFrame,
    starting_equity: float = 100_000.0,
) -> pd.Series:
    """Benchmark: equal-weight, long-only, daily-rebalanced buy-and-hold across
    whatever assets are live each day."""
    live = returns.notna()
    n_live = live.sum(axis=1).replace(0, np.nan)
    w = live.div(n_live, axis=0).fillna(0.0)
    port = (w.shift(1) * returns.fillna(0.0)).sum(axis=1)
    return starting_equity * (1.0 + port).cumprod()

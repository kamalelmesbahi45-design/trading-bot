"""Generate a sample HTML dashboard end-to-end on synthetic XAUUSD-like data.

Run: python scripts/demo_dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_bot.backtest.engine import BacktestEngine
from xauusd_bot.config import load_config
from xauusd_bot.optimize.metrics import compute_stats
from xauusd_bot.optimize.monte_carlo import shuffle_trades
from xauusd_bot.reports.html_report import render_report


def synthetic_xauusd_bars(years: float = 2.0, seed: int = 7) -> pd.DataFrame:
    """Synthetic gold-like H1 bars: drift + vol clustering. No baked-in edge.

    Real XAUUSD has ~10-15% annualised vol and gentle drift. Daily ranges
    around 0.6-1.0% of price. This mimics that without embedding trend bursts
    that would let a trend-follower look unrealistically good.
    """
    n = int(years * 24 * 252)
    idx = pd.date_range("2022-01-03 00:00", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)

    # Hourly vol ~= annualised 14% / sqrt(252*24) ~= 0.0018
    base_vol = 0.0018
    # GARCH-like clustering: vol mean-reverts around base
    vol_state = np.zeros(n)
    vol_state[0] = base_vol
    for i in range(1, n):
        vol_state[i] = 0.95 * vol_state[i - 1] + 0.05 * base_vol \
                       + 0.10 * abs(rng.normal(0, base_vol))
    drift = 0.00002  # ~5%/yr nominal drift (gold-ish)
    shocks = rng.normal(0, 1, n) * vol_state
    log_close = np.cumsum(drift + shocks)
    close = pd.Series(2000.0 * np.exp(log_close), index=idx)

    rng2 = np.random.default_rng(seed + 1)
    range_pct = 0.0008 + 0.0004 * rng2.random(n)
    half = close * range_pct * 0.5
    high = close + half * (1 + 0.5 * rng2.random(n))
    low = close - half * (1 + 0.5 * rng2.random(n))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([high, open_, close], axis=1).max(axis=1)
    low = pd.concat([low, open_, close], axis=1).min(axis=1)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 0.0}, index=idx)


def main() -> None:
    out_dir = Path("reports/output/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/personal_aggressive.yaml")
    # Loosen so trades fire on synthetic data
    cfg.filters.session.enabled = False
    cfg.filters.regime.enabled = False
    cfg.filters.news.enabled = False
    cfg.risk.sizer = "fixed_fractional"
    cfg.risk.per_trade_pct = 0.005
    cfg.risk.hard_cap_per_trade_pct = 0.01
    cfg.risk.max_concurrent_positions = 1
    cfg.risk.trailing_stop = True
    cfg.stops.atr_mult_sl = 2.0
    cfg.stops.atr_mult_tp = 3.0
    cfg.stops.breakeven_at_r = 1.0
    cfg.execution.spread_model = "static"
    cfg.execution.spread_static_points = 25
    cfg.execution.slippage_points = 3

    print("generating synthetic 2y XAUUSD-like H1 bars...")
    bars = synthetic_xauusd_bars(years=2.0)
    print(f"bars: {len(bars):,}  range: {bars.index[0]} ... {bars.index[-1]}")

    print("running backtest...")
    res = BacktestEngine(cfg).run(bars)
    print(f"trades closed: {len(res.trades)}  final equity: ${float(res.metadata['final_equity']):,.2f}")

    print("running Monte Carlo (5000 trade-order shuffles)...")
    mc = shuffle_trades(res.trades, n_runs=5000, seed=0,
                       starting_equity=cfg.account.starting_equity,
                       ruin_dd_pct=cfg.risk.max_drawdown_pct,
                       periods_per_year=252.0)
    print(f"P(ruin): {mc.p_ruin:.2%}  P95 DD: {mc.p95_drawdown:.2%}")

    stats = compute_stats(res.equity_curve, res.trades)
    print(f"Sharpe: {stats.sharpe:.2f}  Sortino: {stats.sortino:.2f}  MaxDD: {stats.max_drawdown:.2%}  "
          f"WinRate: {stats.win_rate:.2%}  PF: {stats.profit_factor:.2f}")

    out = render_report(out_dir, res.equity_curve, res.trades, stats=stats, mc_result=mc,
                       title="XAUUSD bot -- demo (synthetic data, personal_aggressive)")
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()

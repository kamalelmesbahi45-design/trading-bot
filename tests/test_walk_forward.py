"""Walk-forward validator tests on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.config import load_config
from xauusd_bot.optimize.walk_forward import WalkForward


def _long_trending_bars(n_days: int = 900) -> pd.DataFrame:
    """Daily-ish bars with mild trend + noise so Donchian fires occasionally."""
    idx = pd.date_range("2020-01-01 10:00", periods=n_days, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, n_days)  # slight positive drift
    close = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    high = close * 1.005
    low = close * 0.995
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx)


def _cfg():
    cfg = load_config("configs/personal_aggressive.yaml")
    cfg.filters.session.enabled = False
    cfg.filters.regime.enabled = False
    cfg.filters.news.enabled = False
    cfg.risk.sizer = "fixed_fractional"
    cfg.risk.per_trade_pct = 0.005
    cfg.risk.hard_cap_per_trade_pct = 0.01
    cfg.risk.trailing_stop = False
    cfg.risk.max_concurrent_positions = 1
    cfg.stops.atr_mult_sl = 2.0
    cfg.stops.atr_mult_tp = 3.0
    cfg.stops.breakeven_at_r = 100.0
    cfg.execution.spread_model = "static"
    cfg.execution.spread_static_points = 20
    cfg.execution.slippage_points = 2
    return cfg


def test_walkforward_produces_folds_and_oos_curve() -> None:
    bars = _long_trending_bars(900)
    wf = WalkForward(_cfg(), is_years=0.5, oos_months=2, step_months=2)
    res = wf.run(bars, start=bars.index[0].to_pydatetime(), end=bars.index[-1].to_pydatetime())
    assert len(res.fold_results) >= 2
    # OOS curve should be non-empty after the first IS+OOS window
    assert not res.oos_equity.empty


def test_walkforward_empty_bars() -> None:
    wf = WalkForward(_cfg())
    res = wf.run(pd.DataFrame(), start=pd.Timestamp("2024-01-01").to_pydatetime(),
                 end=pd.Timestamp("2024-12-31").to_pydatetime())
    assert res.oos_equity.empty
    assert res.fold_results == []


def test_stability_score_bounded() -> None:
    bars = _long_trending_bars(800)
    wf = WalkForward(_cfg(), is_years=0.5, oos_months=2, step_months=2)
    res = wf.run(bars, start=bars.index[0].to_pydatetime(), end=bars.index[-1].to_pydatetime())
    assert 0.0 <= res.stability_score <= 1.0

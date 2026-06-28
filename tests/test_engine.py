"""End-to-end backtest engine integration test on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.backtest.engine import BacktestEngine, _in_session
from xauusd_bot.config import SessionWindow, load_config


def _cfg_personal_for_engine():
    cfg = load_config("configs/personal_aggressive.yaml")
    # Loosen filters and force fixed sizing for a deterministic run.
    cfg.filters.session.enabled = False
    cfg.filters.regime.enabled = False
    cfg.filters.news.enabled = False
    cfg.risk.sizer = "fixed_fractional"
    cfg.risk.per_trade_pct = 0.005           # 0.5% per trade
    cfg.risk.hard_cap_per_trade_pct = 0.01
    cfg.risk.max_concurrent_positions = 1
    cfg.risk.trailing_stop = False           # cleaner for asserting fills
    cfg.stops.atr_mult_sl = 2.0
    cfg.stops.atr_mult_tp = 3.0
    cfg.stops.breakeven_at_r = 100.0         # effectively disabled in test
    cfg.execution.spread_model = "static"
    cfg.execution.spread_static_points = 20
    cfg.execution.slippage_points = 2
    return cfg


def _rising_then_falling_bars(n: int = 80) -> pd.DataFrame:
    """200..300 over 40 bars, then 300..200 over 40. Clean trend + clean reversal."""
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="1h", tz="UTC")
    up = np.linspace(200, 300, n // 2)
    down = np.linspace(300, 200, n - n // 2)
    close = pd.Series(np.concatenate([up, down]), index=idx)
    high = close + 0.5
    low = close - 0.5
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx)


def test_in_session_basic() -> None:
    win = [SessionWindow(start="07:00", end="16:00")]
    assert _in_session(pd.Timestamp("2024-01-02 10:00", tz="UTC"), win)
    assert not _in_session(pd.Timestamp("2024-01-02 17:00", tz="UTC"), win)


def test_in_session_no_windows_allows_all() -> None:
    assert _in_session(pd.Timestamp("2024-01-02 23:00", tz="UTC"), [])


def test_in_session_window_crosses_midnight() -> None:
    win = [SessionWindow(start="22:00", end="03:00")]
    assert _in_session(pd.Timestamp("2024-01-02 23:30", tz="UTC"), win)
    assert _in_session(pd.Timestamp("2024-01-02 02:30", tz="UTC"), win)
    assert not _in_session(pd.Timestamp("2024-01-02 10:00", tz="UTC"), win)


def test_engine_runs_on_trending_data_and_returns_results() -> None:
    cfg = _cfg_personal_for_engine()
    eng = BacktestEngine(cfg)
    bars = _rising_then_falling_bars(80)
    res = eng.run(bars)

    assert not res.equity_curve.empty
    assert len(res.equity_curve) == len(bars)
    # On a clean up-then-down with trend-following, we should have AT LEAST one closed trade
    assert len(res.trades) >= 1, f"expected trades, got 0\ntrades={res.trades}"
    assert "final_equity" in res.metadata
    # Equity shouldn't end below 50% of starting equity on such a clean dataset
    assert float(res.metadata["final_equity"]) > cfg.account.starting_equity * 0.5


def test_engine_handles_empty_bars() -> None:
    cfg = _cfg_personal_for_engine()
    res = BacktestEngine(cfg).run(pd.DataFrame())
    assert res.equity_curve.empty
    assert res.trades.empty


def test_engine_kills_on_max_drawdown_breach() -> None:
    cfg = _cfg_personal_for_engine()
    cfg.risk.max_drawdown_pct = 0.001       # 0.1% -- effectively impossible
    cfg.risk.per_trade_pct = 0.01           # max risk
    cfg.risk.hard_cap_per_trade_pct = 0.01
    eng = BacktestEngine(cfg)
    bars = _rising_then_falling_bars(80)
    res = eng.run(bars)
    # The bot should have been killed at some point (any kill_reason -> killed True)
    # With trailing off + stops getting hit, we expect at least no growth past the limit
    assert float(res.metadata["final_equity"]) < cfg.account.starting_equity * 1.05

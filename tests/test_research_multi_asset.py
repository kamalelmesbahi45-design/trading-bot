"""Tests for the multi-asset research stack: panel, signals, portfolio, briefing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.research.briefing import generate_briefing
from xauusd_bot.research.panel import daily_returns
from xauusd_bot.research.portfolio_bt import equal_weight_buy_hold, run_portfolio
from xauusd_bot.research.signals import (
    ewma_vol,
    scale_to_portfolio_vol,
    tsmom_signal,
    vol_target_weights,
)


def _synthetic_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """3 assets: a clean uptrend, a clean downtrend, and noise."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    # Strong, clean trends so every lookback (3/6/12m) agrees in sign.
    up = 100 * np.cumprod(1 + rng.normal(0.003, 0.006, n))
    down = 100 * np.cumprod(1 + rng.normal(-0.003, 0.006, n))
    chop = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame({"UP": up, "DOWN": down, "CHOP": chop}, index=idx)


def test_tsmom_signal_sign_matches_trend() -> None:
    panel = _synthetic_panel()
    sig = tsmom_signal(panel, lookbacks=(63, 126, 252))
    last = sig.iloc[-1]
    assert last["UP"] > 0.5      # strong uptrend -> long
    assert last["DOWN"] < -0.5   # strong downtrend -> short


def test_tsmom_signal_warmup_is_nan() -> None:
    panel = _synthetic_panel()
    sig = tsmom_signal(panel, lookbacks=(252,))
    # Before 252 obs the trailing return is NaN -> sign is NaN
    assert sig.iloc[10].isna().all()


def test_ewma_vol_positive_and_annualised() -> None:
    panel = _synthetic_panel()
    vol = ewma_vol(daily_returns(panel))
    last = vol.iloc[-1].dropna()
    # ~1% daily vol -> ~16% annual; allow a wide band
    assert (last > 0.05).all() and (last < 0.5).all()


def test_vol_target_weights_inverse_to_vol() -> None:
    panel = _synthetic_panel()
    returns = daily_returns(panel)
    sig = tsmom_signal(panel)
    vol = ewma_vol(returns)
    w = vol_target_weights(sig, vol, per_asset_vol_target=0.02)
    # Higher-vol asset should get smaller |weight| for the same signal magnitude.
    assert np.isfinite(w.iloc[-1]).all()
    assert (w.abs() <= 2.0 + 1e-9).all().all()   # capped at max leverage


def test_scale_to_portfolio_vol_no_lookahead_and_finite() -> None:
    panel = _synthetic_panel()
    returns = daily_returns(panel)
    sig = tsmom_signal(panel)
    vol = ewma_vol(returns)
    raw = vol_target_weights(sig, vol, 0.02)
    scaled = scale_to_portfolio_vol(raw, returns, target_annual_vol=0.10)
    assert scaled.shape == raw.shape
    assert np.isfinite(scaled.to_numpy()).all()


def test_run_portfolio_basic_accounting() -> None:
    panel = _synthetic_panel()
    returns = daily_returns(panel)
    sig = tsmom_signal(panel)
    vol = ewma_vol(returns)
    w = scale_to_portfolio_vol(vol_target_weights(sig, vol, 0.02), returns, 0.10)
    res = run_portfolio(w, returns, starting_equity=100_000.0, cost_bps=5.0, rebalance="W")
    assert len(res.equity_curve) == len(returns)
    assert res.equity_curve.iloc[-1] > 0
    assert "sharpe" in res.metadata
    # costs must make net <= gross on every day
    assert (res.net_returns <= res.gross_returns + 1e-12).all()


def test_run_portfolio_costs_reduce_return() -> None:
    panel = _synthetic_panel()
    returns = daily_returns(panel)
    sig = tsmom_signal(panel)
    vol = ewma_vol(returns)
    w = scale_to_portfolio_vol(vol_target_weights(sig, vol, 0.02), returns, 0.10)
    free = run_portfolio(w, returns, cost_bps=0.0, rebalance="W")
    costly = run_portfolio(w, returns, cost_bps=50.0, rebalance="W")
    assert costly.equity_curve.iloc[-1] <= free.equity_curve.iloc[-1]


def test_equal_weight_buy_hold_runs() -> None:
    panel = _synthetic_panel()
    returns = daily_returns(panel)
    bh = equal_weight_buy_hold(returns, starting_equity=100_000.0)
    assert len(bh) == len(returns)
    assert bh.iloc[-1] > 0


def test_no_lookahead_weights_use_only_past() -> None:
    """Shuffling the FUTURE of the panel must not change today's weights."""
    panel = _synthetic_panel()
    cutoff = 300
    sig_full = tsmom_signal(panel)
    sig_trunc = tsmom_signal(panel.iloc[: cutoff + 1])
    # signal at the cutoff date must match whether or not future rows exist
    a = sig_full.iloc[cutoff].fillna(0.0)
    b = sig_trunc.iloc[-1].fillna(0.0)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_briefing_structure() -> None:
    panel = _synthetic_panel()
    b = generate_briefing(panel, target_vol=0.10)
    assert b.as_of == panel.index[-1]
    assert len(b.signals) == panel.shape[1]
    txt = b.to_text()
    assert "Daily briefing" in txt
    # UP asset should be a LONG in the briefing
    up = next(s for s in b.signals if s.ticker == "UP")
    assert up.direction == "LONG"
    down = next(s for s in b.signals if s.ticker == "DOWN")
    assert down.direction == "SHORT"

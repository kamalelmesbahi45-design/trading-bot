"""Cross-asset feature tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.features.cross_asset import dxy_trend, gold_dxy_beta, real_yield_proxy, vix_regime


def test_dxy_trend_up_when_fast_ema_above_slow() -> None:
    idx = pd.date_range("2024-01-01", periods=150, freq="1D", tz="UTC")
    dxy = pd.Series(np.linspace(100, 120, 150), index=idx)
    t = dxy_trend(dxy, fast=10, slow=50)
    assert t.iloc[-1] == 1.0


def test_dxy_trend_down_when_fast_below_slow() -> None:
    idx = pd.date_range("2024-01-01", periods=150, freq="1D", tz="UTC")
    dxy = pd.Series(np.linspace(120, 100, 150), index=idx)
    t = dxy_trend(dxy, fast=10, slow=50)
    assert t.iloc[-1] == -1.0


def test_vix_regime_classifies() -> None:
    vix = pd.Series([10.0, 18.0, 30.0])
    r = vix_regime(vix)
    assert (r.values == [0.0, 1.0, 2.0]).all()


def test_real_yield_proxy_short_series_uses_first_diff() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="1D", tz="UTC")
    y = pd.Series([4.0 + i * 0.01 for i in range(50)], index=idx)
    r = real_yield_proxy(y)
    assert pd.isna(r.iloc[0])
    assert r.iloc[1] == pytest_approx(0.01)


def test_real_yield_proxy_long_series_uses_yoy() -> None:
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    y = pd.Series([4.0 + i * 0.01 for i in range(400)], index=idx)
    r = real_yield_proxy(y)
    assert pd.isna(r.iloc[100])
    assert r.iloc[300] > 0


def test_gold_dxy_beta_is_inverse_for_perfectly_inverse_series() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.005, 200)
    gold = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    dxy = pd.Series(100 * np.cumprod(1 - rets), index=idx)
    beta = gold_dxy_beta(gold, dxy, window=60).dropna()
    assert beta.iloc[-1] < 0


def pytest_approx(x, tol: float = 1e-9):
    class _A:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(other - self.v) < tol
    return _A(x)

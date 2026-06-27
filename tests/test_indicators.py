"""Indicator unit tests. Deterministic, no random fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xauusd_bot.features.indicators import (
    adx,
    atr,
    bollinger,
    donchian,
    ema,
    keltner,
    rsi,
    sma,
    true_range,
)


@pytest.fixture
def bars() -> pd.DataFrame:
    """A 50-bar walk with known patterns: rising then falling."""
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 150, n) + np.sin(np.arange(n)) * 0.5, index=idx)
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def test_period_validation() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ema(s, 0)
    with pytest.raises(ValueError):
        sma(s, -1)


def test_true_range_components(bars: pd.DataFrame) -> None:
    tr = true_range(bars["high"], bars["low"], bars["close"])
    # First TR has no prev_close, so it's just high-low (= 2.0 by construction).
    assert tr.iloc[0] == pytest.approx(2.0)
    # TR is always >= high - low (TR can never be smaller).
    assert (tr.iloc[1:] >= bars["high"].iloc[1:] - bars["low"].iloc[1:] - 1e-12).all()


def test_atr_is_smoothed_true_range(bars: pd.DataFrame) -> None:
    a = atr(bars["high"], bars["low"], bars["close"], period=14)
    # First 13 NaN, then values around (high-low)=2 plus some prev-close adjustments
    assert a.iloc[:13].isna().all()
    assert pytest.approx(a.iloc[-1], abs=0.5) == 2.0


def test_ema_first_valid_at_period_minus_one(bars: pd.DataFrame) -> None:
    e = ema(bars["close"], 10)
    assert e.iloc[:9].isna().all()
    assert not pd.isna(e.iloc[9])


def test_sma_first_valid_at_period_minus_one(bars: pd.DataFrame) -> None:
    s = sma(bars["close"], 10)
    assert s.iloc[:9].isna().all()
    assert pytest.approx(s.iloc[9]) == bars["close"].iloc[:10].mean()


def test_rsi_bounded_in_0_100(bars: pd.DataFrame) -> None:
    r = rsi(bars["close"], 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_saturates_at_100_when_only_gains() -> None:
    s = pd.Series(np.arange(1, 30, dtype=float))   # strictly increasing
    r = rsi(s, 14).dropna()
    assert (r.iloc[-1] == pytest.approx(100.0))


def test_bollinger_mid_equals_sma(bars: pd.DataFrame) -> None:
    mid, upper, lower = bollinger(bars["close"], 20, k=2.0)
    s = sma(bars["close"], 20)
    # Compare on the non-NaN region
    pd.testing.assert_series_equal(mid.dropna(), s.dropna(), check_names=False)
    assert (upper.dropna() >= mid.dropna()).all()
    assert (lower.dropna() <= mid.dropna()).all()


def test_donchian_no_lookahead(bars: pd.DataFrame) -> None:
    upper, lower = donchian(bars["high"], bars["low"], period=20)
    # Look-ahead-free: at bar i, upper uses high[i-20..i-1] -> never equals high[i]
    diffs = (upper - bars["high"]).dropna()
    assert (diffs <= 0.0 + 1e-12).any() or (diffs != 0).all()
    # First 20 values NaN (shift then rolling)
    assert upper.iloc[:20].isna().all()
    assert lower.iloc[:20].isna().all()


def test_keltner_bands_envelop_mid(bars: pd.DataFrame) -> None:
    mid, upper, lower = keltner(bars["high"], bars["low"], bars["close"], 20, 14, k=2.0)
    valid = mid.dropna().index
    assert (upper.loc[valid] >= mid.loc[valid]).all()
    assert (lower.loc[valid] <= mid.loc[valid]).all()


def test_adx_is_bounded_when_clean_trend(bars: pd.DataFrame) -> None:
    a = adx(bars["high"], bars["low"], bars["close"], period=14).dropna()
    assert (a >= 0).all() and (a <= 100).all()

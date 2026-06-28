"""BollingerMR signal tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.strategies.mean_reversion import BollingerMR
from xauusd_bot.types import Side


def _oscillating_bars(n: int = 50, level: float = 100.0, amp: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    close = pd.Series(level + amp * np.sin(np.linspace(0, 6 * np.pi, n)), index=idx)
    return pd.DataFrame({"high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)


def test_no_signal_during_warmup() -> None:
    s = BollingerMR()
    assert s.generate(_oscillating_bars(10)) == []


def test_long_when_close_below_lower_band_and_rsi_low() -> None:
    s = BollingerMR(bb_period=10, rsi_period=10, rsi_buy_below=40, rsi_sell_above=60)
    n = 30
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    # Drop sharply at the end so close < lower band and RSI is low
    close = pd.Series([100.0] * (n - 5) + [99.0, 97.0, 94.0, 91.0, 88.0], index=idx)
    bars = pd.DataFrame({"high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)
    sigs = s.generate(bars)
    assert sigs and sigs[0].side is Side.LONG


def test_short_when_close_above_upper_band_and_rsi_high() -> None:
    s = BollingerMR(bb_period=10, rsi_period=10, rsi_buy_below=40, rsi_sell_above=60)
    n = 30
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    close = pd.Series([100.0] * (n - 5) + [101.0, 103.0, 106.0, 109.0, 112.0], index=idx)
    bars = pd.DataFrame({"high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)
    sigs = s.generate(bars)
    assert sigs and sigs[0].side is Side.SHORT


def test_no_signal_when_inside_band() -> None:
    s = BollingerMR()
    bars = _oscillating_bars(40, amp=0.1)   # tiny moves stay inside bands
    sigs = s.generate(bars)
    assert sigs == []

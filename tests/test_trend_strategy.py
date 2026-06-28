"""DonchianTrend signal tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_bot.strategies.trend import DonchianTrend
from xauusd_bot.types import Side


def _bars_flat(n: int = 30, level: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"high": [level + 0.5] * n, "low": [level - 0.5] * n, "close": [level] * n},
        index=idx,
    )


def test_no_signal_during_warmup() -> None:
    s = DonchianTrend(donchian_period=20)
    bars = _bars_flat(15)
    assert s.generate(bars) == []


def test_long_signal_on_breakout_above_prior_high() -> None:
    s = DonchianTrend(donchian_period=20)
    bars = _bars_flat(25)
    # Spike the last close above the previous Donchian upper
    bars.iloc[-1, bars.columns.get_loc("close")] = 102.0
    bars.iloc[-1, bars.columns.get_loc("high")] = 102.5
    sigs = s.generate(bars)
    assert len(sigs) == 1
    assert sigs[0].side is Side.LONG
    assert sigs[0].strategy == "trend_donchian"


def test_short_signal_on_break_below_prior_low() -> None:
    s = DonchianTrend(donchian_period=20)
    bars = _bars_flat(25)
    bars.iloc[-1, bars.columns.get_loc("close")] = 98.0
    bars.iloc[-1, bars.columns.get_loc("low")] = 97.5
    sigs = s.generate(bars)
    assert len(sigs) == 1
    assert sigs[0].side is Side.SHORT


def test_no_signal_when_within_range() -> None:
    s = DonchianTrend(donchian_period=20)
    bars = _bars_flat(25)
    sigs = s.generate(bars)
    assert sigs == []


def test_rising_series_eventually_signals_long() -> None:
    s = DonchianTrend(donchian_period=10)
    n = 30
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    bars = pd.DataFrame({"high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)
    sigs = s.generate(bars)
    # On a strictly rising series the last close exceeds the prior high band -> LONG
    assert sigs and sigs[0].side is Side.LONG

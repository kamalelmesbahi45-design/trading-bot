"""Session breakout tests (London + Asia)."""
from __future__ import annotations

import pandas as pd

from xauusd_bot.strategies.breakout import AsianRangeBreakout, LondonOpenBreakout
from xauusd_bot.types import Side


def _multi_day_bars(level: float = 100.0, n_days: int = 3) -> pd.DataFrame:
    """48 hours x n_days so we have ATR warm-up. Constant close baseline."""
    n = 24 * n_days
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"high": [level + 0.4] * n, "low": [level - 0.4] * n, "close": [level] * n},
        index=idx,
    )


def test_london_breakout_long_when_close_above_range_high() -> None:
    s = LondonOpenBreakout(buffer_atr_mult=0.0, atr_period=10)
    bars = _multi_day_bars(level=100.0, n_days=3)
    # Spike the most recent 08:00 close above the 07:00-range high
    day3_08 = pd.Timestamp("2024-01-04 08:00", tz="UTC")
    bars.loc[day3_08, "close"] = 101.0
    bars.loc[day3_08, "high"] = 101.1
    sigs = s.generate(bars.loc[: day3_08])
    assert sigs and sigs[0].side is Side.LONG


def test_london_breakout_short_when_close_below_range_low() -> None:
    s = LondonOpenBreakout(buffer_atr_mult=0.0, atr_period=10)
    bars = _multi_day_bars(level=100.0, n_days=3)
    day3_08 = pd.Timestamp("2024-01-04 08:00", tz="UTC")
    bars.loc[day3_08, "close"] = 98.0
    bars.loc[day3_08, "low"] = 97.9
    sigs = s.generate(bars.loc[: day3_08])
    assert sigs and sigs[0].side is Side.SHORT


def test_london_breakout_no_signal_during_range() -> None:
    s = LondonOpenBreakout(atr_period=10)
    bars = _multi_day_bars(level=100.0, n_days=3)
    day3_07_30 = pd.Timestamp("2024-01-04 07:00", tz="UTC")  # last bar still inside 07-08 range
    sigs = s.generate(bars.loc[: day3_07_30])
    assert sigs == []


def test_asian_breakout_uses_asia_window() -> None:
    s = AsianRangeBreakout(buffer_atr_mult=0.0, atr_period=10)
    bars = _multi_day_bars(level=100.0, n_days=3)
    day3_08 = pd.Timestamp("2024-01-04 08:00", tz="UTC")
    bars.loc[day3_08, "close"] = 102.0
    bars.loc[day3_08, "high"] = 102.1
    sigs = s.generate(bars.loc[: day3_08])
    assert sigs and sigs[0].side is Side.LONG

"""Sizer unit tests."""
from __future__ import annotations

import math

import pytest

from xauusd_bot.risk.sizing import (
    FixedFractional,
    FractionalKelly,
    SizingInputs,
    VolTarget,
    _lots_for_risk,
    _quantise,
)


def test_quantise_floors_to_step() -> None:
    assert _quantise(0.234, 0.01, 0.01) == 0.23
    assert _quantise(0.005, 0.01, 0.01) == 0.0  # below min
    assert _quantise(0.01, 0.01, 0.01) == 0.01
    assert _quantise(-0.5, 0.01, 0.01) == 0.0
    assert _quantise(math.nan, 0.01, 0.01) == 0.0


def test_lots_for_risk_zero_denom() -> None:
    assert _lots_for_risk(100.0, SizingInputs(equity=10_000, stop_distance_price=0.0)) == 0.0


def test_fixed_fractional_basic_gold_math() -> None:
    # $10k equity, 1% risk, $5 stop, 100 oz/lot -> 100 / (5*100) = 0.20 lots
    sizer = FixedFractional(pct_per_trade=0.01, hard_cap_pct=0.02)
    lots = sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0))
    assert lots == 0.20


def test_fixed_fractional_hard_cap_clamps() -> None:
    # Config asks for 5% but hard cap is 2% -> should size to 2%
    sizer = FixedFractional(pct_per_trade=0.05, hard_cap_pct=0.02)
    lots = sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0))
    # 10k * 2% = 200 / (5*100) = 0.40
    assert lots == 0.40


def test_fixed_fractional_negative_equity_zero_lots() -> None:
    sizer = FixedFractional(pct_per_trade=0.01, hard_cap_pct=0.02)
    assert sizer.lots(SizingInputs(equity=-100, stop_distance_price=5.0)) == 0.0


def test_fractional_kelly_fallback_when_below_min_trades() -> None:
    sizer = FractionalKelly(
        kelly_fraction=0.25, fallback_pct=0.01, hard_cap_pct=0.02,
        min_trades=30, lookback_trades=100,
    )
    sizer.update_stats([1.0, -1.0, 2.0])  # only 3 trades
    # Falls back to 1% fixed -> 100 / (5*100) = 0.20
    assert sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0)) == 0.20


def test_fractional_kelly_uses_stats_when_enough_trades() -> None:
    sizer = FractionalKelly(
        kelly_fraction=0.25, fallback_pct=0.01, hard_cap_pct=0.10,
        min_trades=10, lookback_trades=100,
    )
    # 10 trades: 6 wins of +2R, 4 losses of -1R.
    # p=0.6, b=2 -> f* = 0.6 - 0.4/2 = 0.4. fractional = 0.25 * 0.4 = 0.10 (capped at 0.10).
    sizer.update_stats([2.0]*6 + [-1.0]*4)
    # 10k * 0.10 = 1000 / (5*100) = 2.00 lots
    assert sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0)) == 2.0


def test_fractional_kelly_negative_edge_zero_lots() -> None:
    sizer = FractionalKelly(
        kelly_fraction=0.25, fallback_pct=0.01, hard_cap_pct=0.02,
        min_trades=10, lookback_trades=100,
    )
    # 10 trades: 2 wins of +1R, 8 losses of -1R -> p=0.2, b=1, f* = 0.2 - 0.8/1 = -0.6
    sizer.update_stats([1.0]*2 + [-1.0]*8)
    # negative edge clamps to 0 -> 0 lots
    assert sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0)) == 0.0


def test_fractional_kelly_hard_cap_clamps_when_fstar_huge() -> None:
    sizer = FractionalKelly(
        kelly_fraction=1.0, fallback_pct=0.01, hard_cap_pct=0.02,
        min_trades=10, lookback_trades=100,
    )
    # 10 wins of +5R, 0 losses... but we need at least 1 loss for ratio meaning.
    # 9 wins of +5R, 1 loss -> p=0.9, b=5 -> f* = 0.9 - 0.1/5 = 0.88. Capped at 2%.
    sizer.update_stats([5.0]*9 + [-1.0])
    # 10k * 0.02 = 200 / (5*100) = 0.4
    assert sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0)) == 0.4


def test_fractional_kelly_no_wins_returns_zero_lots() -> None:
    sizer = FractionalKelly(
        kelly_fraction=0.25, fallback_pct=0.01, hard_cap_pct=0.02,
        min_trades=5, lookback_trades=100,
    )
    sizer.update_stats([-1.0]*5)
    assert sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0)) == 0.0


def test_voltarget_documented_stub() -> None:
    sizer = VolTarget(target_annual_vol=0.15, hard_cap_pct=0.02)
    with pytest.raises(NotImplementedError):
        sizer.lots(SizingInputs(equity=10_000, stop_distance_price=5.0))

"""Cost-model unit tests."""
from __future__ import annotations

from datetime import datetime

from xauusd_bot.backtest.costs import DynamicSpread, StaticSpread, commission, swap


def test_static_spread_constant() -> None:
    s = StaticSpread(points=25)
    assert s.spread(datetime(2024, 1, 2, 8), 2000.0) == 0.25
    assert s.spread(datetime(2024, 1, 2, 3), 2000.0) == 0.25  # asia same as london


def test_dynamic_spread_widens_in_asia() -> None:
    s = DynamicSpread(base_points=25, asia_widen_x=2.0)
    london = s.spread(datetime(2024, 1, 2, 10), 2000.0)
    asia = s.spread(datetime(2024, 1, 2, 3), 2000.0)
    assert london == 0.25
    assert asia == 0.50


def test_dynamic_spread_widens_on_event() -> None:
    s = DynamicSpread(base_points=25, event_widen_x=4.0, asia_widen_x=2.0)
    normal = s.spread(datetime(2024, 1, 2, 13), 2000.0)
    event = s.spread(datetime(2024, 1, 2, 13), 2000.0, in_event_window=True)
    assert normal == 0.25
    assert event == 1.0


def test_dynamic_spread_event_dominates_asia() -> None:
    s = DynamicSpread(base_points=25, event_widen_x=4.0, asia_widen_x=2.0)
    asia_event = s.spread(datetime(2024, 1, 2, 3), 2000.0, in_event_window=True)
    # 25 * 0.01 * max(2, 4) = 1.0
    assert asia_event == 1.0


def test_commission_floors_at_zero_for_invalid_qty() -> None:
    assert commission(-1.0, 7.0) == 0.0
    assert abs(commission(0.10, 7.0) - 0.7) < 1e-9


def test_swap_zero_when_no_nights_or_qty() -> None:
    assert swap(0.10, -0.0002, 10_000, 0) == 0.0
    assert swap(0.0, -0.0002, 10_000, 1) == 0.0


def test_swap_sign_follows_side_pct() -> None:
    pos = swap(0.10, 0.0002, 10_000, 1)
    neg = swap(0.10, -0.0002, 10_000, 1)
    assert pos > 0 and neg < 0

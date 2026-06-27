"""Tests for core domain types."""
from __future__ import annotations

from datetime import datetime

from xauusd_bot.types import Position, Side


def test_position_unrealized_long() -> None:
    p = Position(
        strategy="t",
        side=Side.LONG,
        qty=0.10,
        entry_price=2000.0,
        entry_ts=datetime(2025, 1, 1),
    )
    pnl = p.unrealized_pnl(mark=2010.0, contract_size=100)
    # +10 price move * 0.10 lot * 100 oz = $100
    assert abs(pnl - 100.0) < 1e-9


def test_position_unrealized_short() -> None:
    p = Position(
        strategy="t",
        side=Side.SHORT,
        qty=0.20,
        entry_price=2000.0,
        entry_ts=datetime(2025, 1, 1),
    )
    pnl = p.unrealized_pnl(mark=1990.0, contract_size=100)
    # -10 price move * -1 (short) * 0.20 lot * 100 oz = $200
    assert abs(pnl - 200.0) < 1e-9

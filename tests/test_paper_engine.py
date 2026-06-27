"""PaperEngine tests."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from xauusd_bot.execution.paper import PaperEngine
from xauusd_bot.types import Order, OrderType, Side


def _engine(tmp_path: Path) -> PaperEngine:
    return PaperEngine(starting_equity=10_000.0, state_path=tmp_path / "state.json",
                        spread_points=20, slippage_points=2)


def test_place_market_long_at_ask_plus_slip(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                  type=OrderType.MARKET, qty=0.10)
    fill = eng.place(order)
    # ask 2000.20 + slip 2 pts = 2000.22
    assert abs(fill.price - 2000.22) < 1e-9
    assert eng.positions() and eng.positions()[0].side is Side.LONG


def test_place_market_short_at_bid_minus_slip(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.SHORT,
                  type=OrderType.MARKET, qty=0.10)
    fill = eng.place(order)
    # bid 2000.0 - slip 2 pts = 1999.98
    assert abs(fill.price - 1999.98) < 1e-9


def test_place_fails_without_market_data(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                  type=OrderType.MARKET, qty=0.10)
    with pytest.raises(RuntimeError):
        eng.place(order)


def test_on_quote_triggers_long_sl_hit(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                  type=OrderType.MARKET, qty=0.10, sl_price=1990.0, tp_price=2020.0)
    eng.place(order)
    # Bid falls below SL
    fills = eng.on_quote(bid=1989.0, ask=1989.20, ts=datetime(2024, 1, 2, 11))
    assert len(fills) == 1
    assert eng.positions() == []


def test_on_quote_triggers_long_tp_hit(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                  type=OrderType.MARKET, qty=0.10, sl_price=1990.0, tp_price=2010.0)
    eng.place(order)
    fills = eng.on_quote(bid=2010.50, ask=2010.70, ts=datetime(2024, 1, 2, 11))
    assert len(fills) == 1
    assert eng.positions() == []


def test_modify_updates_sl_tp(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                  type=OrderType.MARKET, qty=0.10, sl_price=1990.0)
    eng.place(order)
    pos = eng.positions()[0]
    eng.modify(pos, sl=1995.0, tp=2020.0)
    assert pos.sl_price == 1995.0
    assert pos.tp_price == 2020.0


def test_flatten_all_closes_all_positions(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    for _ in range(3):
        eng.place(Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                        type=OrderType.MARKET, qty=0.10))
    assert len(eng.positions()) == 3
    fills = eng.flatten_all()
    assert len(fills) == 3
    assert eng.positions() == []


def test_state_persists_and_reloads(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    eng = PaperEngine(starting_equity=10_000.0, state_path=state, spread_points=20, slippage_points=2)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    eng.place(Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                    type=OrderType.MARKET, qty=0.10, sl_price=1990.0, tp_price=2020.0))
    eng.disconnect()
    assert state.exists()

    # New instance should load the same equity + position
    eng2 = PaperEngine(starting_equity=99_999.0, state_path=state, spread_points=20, slippage_points=2)
    eng2.connect()
    assert len(eng2.positions()) == 1
    assert eng2.positions()[0].side is Side.LONG


def test_equity_reflects_unrealised(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    eng.place(Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                    type=OrderType.MARKET, qty=0.10))
    # commission deducted on entry, but no slip on equity recalc; equity ~= starting - commission
    eng.on_quote(bid=2020.0, ask=2020.20, ts=datetime(2024, 1, 2, 11))
    # mark = 2020.10; entry was 2000.22; +19.88 * 0.10 * 100 = +198.8 unrealised
    eq = eng.equity()
    assert eq > 10_000.0   # in the green
    assert eq < 11_000.0


def test_state_file_is_valid_json(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.connect()
    eng.on_quote(bid=2000.0, ask=2000.20, ts=datetime(2024, 1, 2, 10))
    eng.place(Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                    type=OrderType.MARKET, qty=0.10, sl_price=1990.0))
    data = json.loads((tmp_path / "state.json").read_text())
    assert "equity" in data
    assert "positions" in data
    assert len(data["positions"]) == 1

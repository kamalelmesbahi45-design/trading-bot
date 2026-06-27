"""Fill-simulator unit tests."""
from __future__ import annotations

from datetime import datetime

from xauusd_bot.backtest.costs import StaticSpread
from xauusd_bot.backtest.fill_sim import FillSimulator
from xauusd_bot.types import Bar, Order, OrderType, Side


def _bar(open_: float = 2000.0, high: float = 2005.0, low: float = 1995.0, close: float = 2002.0) -> Bar:
    return Bar(ts=datetime(2024, 1, 2, 10), open=open_, high=high, low=low, close=close, volume=0.0)


def _sim(spread_pts: int = 20, slip_pts: int = 5, commish: float = 7.0) -> FillSimulator:
    return FillSimulator(
        spread_model=StaticSpread(spread_pts),
        slippage_points=slip_pts,
        commission_per_lot=commish,
    )


def test_market_buy_fills_above_open() -> None:
    sim = _sim()
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.LONG,
                  type=OrderType.MARKET, qty=0.10)
    fill = sim.fill_market(order, _bar())
    # spread 20pts = 0.20 -> half = 0.10. slip 5pts = 0.05. fill = 2000 + 0.10 + 0.05 = 2000.15
    assert abs(fill.price - 2000.15) < 1e-9
    assert abs(fill.commission - 0.70) < 1e-9


def test_market_sell_fills_below_open() -> None:
    sim = _sim()
    order = Order(ts=datetime(2024,1,2,10), strategy="t", side=Side.SHORT,
                  type=OrderType.MARKET, qty=0.10)
    fill = sim.fill_market(order, _bar())
    # 2000 - 0.10 - 0.05 = 1999.85
    assert abs(fill.price - 1999.85) < 1e-9


def test_sl_hit_long() -> None:
    sim = _sim()
    hit = sim.maybe_fill_sl_tp(Side.LONG, sl=1996.0, tp=2010.0, bar=_bar(low=1995.0))
    assert hit is not None and hit.reason == "sl" and hit.fill_price == 1996.0


def test_tp_hit_long() -> None:
    sim = _sim()
    hit = sim.maybe_fill_sl_tp(Side.LONG, sl=1990.0, tp=2003.0, bar=_bar(high=2005.0))
    assert hit is not None and hit.reason == "tp" and hit.fill_price == 2003.0


def test_sl_priority_on_long_when_both_could_hit() -> None:
    sim = _sim()
    # SL=1996 within range AND TP=2004 within range -> SL wins (pessimistic)
    hit = sim.maybe_fill_sl_tp(Side.LONG, sl=1996.0, tp=2004.0, bar=_bar(low=1995.0, high=2005.0))
    assert hit is not None and hit.reason == "sl"


def test_sl_hit_short() -> None:
    sim = _sim()
    hit = sim.maybe_fill_sl_tp(Side.SHORT, sl=2004.0, tp=1990.0, bar=_bar(high=2005.0))
    assert hit is not None and hit.reason == "sl" and hit.fill_price == 2004.0


def test_tp_hit_short() -> None:
    sim = _sim()
    hit = sim.maybe_fill_sl_tp(Side.SHORT, sl=2010.0, tp=1996.0, bar=_bar(low=1995.0))
    assert hit is not None and hit.reason == "tp" and hit.fill_price == 1996.0


def test_no_hit_returns_none() -> None:
    sim = _sim()
    hit = sim.maybe_fill_sl_tp(Side.LONG, sl=1990.0, tp=2010.0, bar=_bar(low=1995.0, high=2005.0))
    assert hit is None


def test_close_market_long_sells_at_bid() -> None:
    sim = _sim()
    fill = sim.close_market(Side.LONG, qty=0.10, bar=_bar())
    # close long = sell at open - spread/2 - slip = 1999.85
    assert abs(fill.price - 1999.85) < 1e-9


def test_close_market_short_buys_at_ask() -> None:
    sim = _sim()
    fill = sim.close_market(Side.SHORT, qty=0.10, bar=_bar())
    assert abs(fill.price - 2000.15) < 1e-9

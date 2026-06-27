"""Prop-firm gate unit tests."""
from __future__ import annotations

from datetime import date, datetime

from xauusd_bot.risk.propfirm_gates import GateDecision, GateState, PropFirmGates


def _state(start_eq: float = 100_000.0) -> GateState:
    return GateState(starting_equity=start_eq)


def _ftmo_gates() -> PropFirmGates:
    # FTMO-style with safety buffer
    return PropFirmGates(
        daily_loss_limit_pct=0.04,
        max_drawdown_pct=0.08,
        profit_target_pct=0.08,
        min_trading_days=4,
    )


def test_allow_when_well_within_limits() -> None:
    g = _ftmo_gates()
    s = _state()
    s.on_day_start(date(2025, 1, 6), 100_000.0)
    d = g.evaluate(datetime(2025, 1, 6, 10), current_equity=100_000.0,
                   proposed_risk_amount=500.0, state=s)
    assert d == GateDecision.ALLOW


def test_kill_on_existing_max_dd_breach() -> None:
    g = _ftmo_gates()
    s = _state()
    s.on_day_start(date(2025, 1, 6), 100_000.0)
    # -9% already, max DD is -8%
    d = g.evaluate(datetime(2025, 1, 6, 10), current_equity=91_000.0,
                   proposed_risk_amount=0.0, state=s)
    assert d == GateDecision.KILL
    assert s.killed


def test_kill_on_existing_daily_breach() -> None:
    g = _ftmo_gates()
    s = _state()
    s.on_day_start(date(2025, 1, 6), 100_000.0)
    # -5% today, daily limit is -4%
    d = g.evaluate(datetime(2025, 1, 6, 10), current_equity=95_000.0,
                   proposed_risk_amount=0.0, state=s)
    assert d == GateDecision.KILL


def test_veto_when_worst_case_breaches_daily() -> None:
    g = _ftmo_gates()
    s = _state()
    s.on_day_start(date(2025, 1, 6), 100_000.0)
    # Currently -3.5% (96.5k), risk 1k would push to -4.5% > 4% daily limit
    d = g.evaluate(datetime(2025, 1, 6, 10), current_equity=96_500.0,
                   proposed_risk_amount=1_000.0, state=s)
    assert d == GateDecision.VETO
    assert not s.killed


def test_veto_when_worst_case_breaches_total_dd() -> None:
    g = _ftmo_gates()
    s = _state()
    s.on_day_start(date(2025, 1, 8), 93_000.0)  # day starts at -7% already
    # Risk 2k would push total to -9% > 8% max DD
    d = g.evaluate(datetime(2025, 1, 8, 10), current_equity=93_000.0,
                   proposed_risk_amount=2_000.0, state=s)
    assert d == GateDecision.VETO


def test_killed_stays_killed() -> None:
    g = _ftmo_gates()
    s = _state()
    s.mark_killed("manual")
    d = g.evaluate(datetime(2025, 1, 6, 10), current_equity=100_000.0,
                   proposed_risk_amount=0.0, state=s)
    assert d == GateDecision.KILL


def test_daily_limit_skipped_when_none() -> None:
    g = PropFirmGates(daily_loss_limit_pct=None, max_drawdown_pct=0.25)
    s = _state()
    # -10% today would breach 4% daily, but daily disabled -> ALLOW (within 25% total)
    d = g.evaluate(datetime(2025, 1, 6, 10), current_equity=90_000.0,
                   proposed_risk_amount=500.0, state=s)
    assert d == GateDecision.ALLOW


def test_hwm_updates_only_on_new_high() -> None:
    s = _state(100_000.0)
    s.on_equity_update(101_000.0)
    s.on_equity_update(99_000.0)
    s.on_equity_update(100_500.0)
    assert s.high_water_mark == 101_000.0


def test_profit_target_requires_min_days() -> None:
    g = _ftmo_gates()
    s = _state()
    # 8% gain reached but only 2 trading days -> not hit
    s.on_trade_day(date(2025, 1, 6))
    s.on_trade_day(date(2025, 1, 7))
    assert not g.profit_target_hit(108_000.0, s)
    # Now add 2 more days -> hit
    s.on_trade_day(date(2025, 1, 8))
    s.on_trade_day(date(2025, 1, 9))
    assert g.profit_target_hit(108_000.0, s)


def test_profit_target_hit_disabled_when_none() -> None:
    g = PropFirmGates(daily_loss_limit_pct=None, max_drawdown_pct=0.10, profit_target_pct=None)
    s = _state()
    assert not g.profit_target_hit(1_000_000.0, s)


def test_day_start_equity_locked_on_first_call() -> None:
    s = _state()
    d = date(2025, 1, 6)
    s.on_day_start(d, 100_000.0)
    s.on_day_start(d, 99_000.0)  # later call same day must NOT overwrite
    assert s.day_start_equity[d] == 100_000.0

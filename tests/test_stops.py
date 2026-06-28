"""Stop policy unit tests."""
from __future__ import annotations

from datetime import datetime

import pytest

from xauusd_bot.risk.stops import StopPolicy
from xauusd_bot.types import Position, Side


def _pos(side: Side, entry: float, sl: float, initial_sl: float | None = None) -> Position:
    return Position(
        strategy="t",
        side=side,
        qty=0.1,
        entry_price=entry,
        entry_ts=datetime(2025, 1, 1),
        sl_price=sl,
        initial_sl_price=initial_sl if initial_sl is not None else sl,
    )


def test_initial_levels_long() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0)
    sl, tp = p.initial_levels(Side.LONG, entry=2000.0, atr=5.0)
    assert sl == 1990.0
    assert tp == 2015.0


def test_initial_levels_short() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0)
    sl, tp = p.initial_levels(Side.SHORT, entry=2000.0, atr=5.0)
    assert sl == 2010.0
    assert tp == 1985.0


def test_initial_levels_rejects_nonpositive_atr() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0)
    with pytest.raises(ValueError):
        p.initial_levels(Side.LONG, entry=2000.0, atr=0.0)


def test_breakeven_moves_sl_to_entry_long() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0, breakeven_at_r=1.0, trailing=False)
    pos = _pos(Side.LONG, entry=2000.0, sl=1990.0)
    # 1R = $10. Mark at 2010 -> r=+1.0 -> breakeven triggers.
    new_sl, _ = p.update(pos, mark=2010.0, atr=5.0)
    assert new_sl == 2000.0


def test_breakeven_does_not_loosen_sl_short() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0, breakeven_at_r=1.0, trailing=False)
    pos = _pos(Side.SHORT, entry=2000.0, sl=2010.0)
    # 1R = $10. Mark at 1990 -> r=+1.0 -> breakeven triggers (SL down to entry).
    new_sl, _ = p.update(pos, mark=1990.0, atr=5.0)
    assert new_sl == 2000.0


def test_trailing_tightens_long() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0, breakeven_at_r=10.0, trailing=True)
    pos = _pos(Side.LONG, entry=2000.0, sl=1990.0)
    # Mark at 2030, atr 5 -> trail candidate = 2030 - 10 = 2020 (tighter than 1990).
    new_sl, _ = p.update(pos, mark=2030.0, atr=5.0)
    assert new_sl == 2020.0


def test_trailing_never_loosens_long() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0, breakeven_at_r=10.0, trailing=True)
    pos = _pos(Side.LONG, entry=2000.0, sl=1995.0)
    # Mark drops back to 2002, candidate = 1992 < current 1995 -> no change reported.
    new_sl, _ = p.update(pos, mark=2002.0, atr=5.0)
    assert new_sl is None


def test_trailing_never_loosens_short() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0, breakeven_at_r=10.0, trailing=True)
    pos = _pos(Side.SHORT, entry=2000.0, sl=2005.0)
    # Mark rises back to 1998, candidate = 2008 > current 2005 -> no loosening.
    new_sl, _ = p.update(pos, mark=1998.0, atr=5.0)
    assert new_sl is None


def test_no_changes_returned_when_sl_unchanged() -> None:
    p = StopPolicy(atr_mult_sl=2.0, atr_mult_tp=3.0, breakeven_at_r=10.0, trailing=False)
    pos = _pos(Side.LONG, entry=2000.0, sl=1990.0)
    new_sl, new_tp = p.update(pos, mark=2003.0, atr=5.0)
    assert new_sl is None and new_tp is None

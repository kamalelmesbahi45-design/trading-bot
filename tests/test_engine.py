"""Sanity tests: barrier logic, no look-ahead, BH procedure."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.edges import triple_barrier, Setup, benjamini_hochberg

def mk(prices_o, prices_h, prices_l, prices_c, atr=1.0):
    idx = pd.bdate_range("2020-01-01", periods=len(prices_c))
    f = pd.DataFrame(dict(Open=prices_o, High=prices_h, Low=prices_l, Close=prices_c), index=idx)
    f["atr"] = atr; f["vol_pct"] = 0.5
    return f

S = Setup("T", "t", "", True, 1.0, 2.0, 5, (1,), None, "", "", "", lambda r, d: [])

def test_target_hit_long():
    f = mk([100,100,101,102,103,104,105], [100,101,102,103,104,105,106], [100,99.5,100.5,101.5,102.5,103.5,104.5], [100,101,102,103,104,105,106])
    sig = pd.Series([True]+[False]*6, index=f.index)
    tr = triple_barrier(f, sig, 1, S, True, 0.0)
    assert len(tr) == 1 and tr.R.iloc[0] == 2.0 and tr.outcome.iloc[0] == "target"

def test_same_bar_both_is_stop():
    f = mk([100,100,100,100,100,100,100], [100,103,100,100,100,100,100], [100,98,100,100,100,100,100], [100]*7)
    sig = pd.Series([True]+[False]*6, index=f.index)
    tr = triple_barrier(f, sig, 1, S, True, 0.0)
    assert tr.R.iloc[0] == -1.0  # pessimistic

def test_gap_through_stop():
    f = mk([100,100,97,97,97,97,97], [100,100.5,97,97,97,97,97], [100,99.5,96,96,96,96,96], [100,100,97,97,97,97,97])
    sig = pd.Series([True]+[False]*6, index=f.index)
    tr = triple_barrier(f, sig, 1, S, True, 0.0)
    assert abs(tr.R.iloc[0] + 3.0) < 1e-9   # filled at the gap open, worse than -1R

def test_entry_is_next_open():
    f = mk([100,110,110,110,110,110,110], [100,110.5,110.5,110.5,110.5,110.5,110.5], [100,109.5,109.5,109.5,109.5,109.5,109.5], [100,110,110,110,110,110,110])
    sig = pd.Series([True]+[False]*6, index=f.index)
    tr = triple_barrier(f, sig, 1, S, True, 0.0)
    assert tr.outcome.iloc[0] == "time" and abs(tr.R.iloc[0]) < 1e-9  # no free +10 from the gap

def test_cost():
    f = mk([100]*7, [100]*7, [100]*7, [100]*7)
    sig = pd.Series([True]+[False]*6, index=f.index)
    tr = triple_barrier(f, sig, 1, S, True, 10.0)  # 10bp of 100 = 0.1 = 0.1R
    assert abs(tr.R.iloc[0] + 0.1) < 1e-9

def test_bh():
    assert benjamini_hochberg([0.001, 0.2, 0.03, 0.5], 0.1) == [True, False, True, False]

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k)

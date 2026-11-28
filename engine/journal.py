"""
Forward (live) track record. Every run appends the day's signals to data/signal_log.csv;
every later run re-scores them on the bars that came AFTER the signal, with exactly the
same rules as the backtest. This is the only truly out-of-sample evidence.
"""
from __future__ import annotations
import os
import pandas as pd

LOG = "data/signal_log.csv"
COLS = ["asof", "market", "setup", "dir", "grade", "entry", "stop", "target", "time_stop", "prob", "exp_R"]


def append(signals: list[dict], path: str = LOG):
    if not signals:
        return
    new = pd.DataFrame([{k: s[k] for k in COLS} for s in signals])
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates(subset=["asof", "market", "setup", "dir"], keep="first")
    new.to_csv(path, index=False)


def score(frames: dict, path: str = LOG) -> dict:
    """frames: market id -> daily OHLC DataFrame. Entry = first open after the signal bar."""
    if not os.path.exists(path):
        return dict(closed=[], open=0)
    log = pd.read_csv(path)
    closed, still_open = [], 0
    for _, s in log.iterrows():
        df = frames.get(s["market"])
        if df is None:
            continue
        after = df[df.index > pd.Timestamp(s["asof"])]
        if len(after) == 0:
            still_open += 1
            continue
        d = 1 if s["dir"] == "LONG" else -1
        E = after["Open"].iloc[0]
        risk = abs(s["entry"] - s["stop"])
        stop = E - d * risk
        tgt = E + d * abs(s["target"] - s["entry"])
        res = None
        for i, (t, b) in enumerate(after.iterrows()):
            if t > pd.Timestamp(s["time_stop"]):
                res = (d * (after["Close"].iloc[i - 1] - E) / risk, "time", str(after.index[i - 1].date()))
                break
            if i > 0 and d * (b.Open - stop) <= 0:
                res = (d * (b.Open - E) / risk, "stop", str(t.date())); break
            if (d > 0 and b.Low <= stop) or (d < 0 and b.High >= stop):
                res = (-1.0, "stop", str(t.date())); break
            if (d > 0 and b.High >= tgt) or (d < 0 and b.Low <= tgt):
                res = (abs(tgt - E) / risk, "target", str(t.date())); break
        if res is None:
            still_open += 1
            continue
        closed.append(dict(asof=s["asof"], market=s["market"], setup=s["setup"], dir=s["dir"], grade=s["grade"],
                           prob=s["prob"], R=round(float(res[0]), 3), outcome=res[1], exit=res[2]))
    return dict(closed=closed, open=still_open)

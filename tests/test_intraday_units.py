"""Fast checks of the day-trading engine (no network): order logic and a synthetic London session."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import intraday as I

assert I.stop_entry(np.array([1.0]), np.array([2.0]), np.array([0.0]), 0, 1, 1.5, 0.5)[1] == 0      # both sides -> whipsaw
assert I.run_trade(np.array([10, 10.0]), np.array([10, 10.0]), np.array([10, 8.0]), np.array([10, 9.0]), 0, 1, 9.0, None)[1] == "stop"
# synthetic USD/JPY day: quiet Asia 150.0-150.2, breaks up at 07:30 London
idx = pd.date_range("2026-03-02 00:00", "2026-03-02 17:00", freq="5min", tz="Europe/London", inclusive="left").tz_convert("UTC")
c = np.where(idx.tz_convert("Europe/London").hour < 7, 150.1, 150.4)
df = pd.DataFrame(dict(open=c, high=c + 0.1, low=c - 0.1, close=c), index=idx)
S = [s for s in I.SYSTEMS if s["id"] == "LDN_JPY"][0]
r = I.sys_london(S, df, pd.Timestamp("2026-03-02 18:00", tz="UTC"))
assert r["status"] == "DONE" and r["trade"]["dir"] == "LONG", r
print("intraday unit tests ok")

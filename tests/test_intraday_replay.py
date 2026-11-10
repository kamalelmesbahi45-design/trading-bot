"""
Replays the LIVE day-trading engine over historical Dukascopy 5-minute data and checks that it
produces the same trades (direction and R) as the research backtests in daytrade/.
Usage: python tests/test_intraday_replay.py <m5_dir> <research_out_dir_round1> <research_out_dir_round4>
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import intraday as I

M5, R1, R4 = sys.argv[1], sys.argv[2], sys.argv[3]
FILES = {"^NDX": "usatechidxusd", "^GSPC": "usa500idxusd", "GC=F": "xauusd", "^GDAXI": "deuidxeur", "JPY=X": "usdjpy"}
REF = {"LDN_JPY": f"{R1}/trades_LONDON__USDJPY__stop_mode-opp_target_mode-EOD_trend_filter-False.csv",
       "ORB_DAX": f"{R1}/trades_ORB_RANGE__GER40__W-30_target_mode-EOD_trend_filter-True.csv",
       "NOISE_NAS": f"{R1}/trades_NOISE_CLEAN__NAS100.csv",
       "BIG_NAS": f"{R4}/tr_NAS100_1330_2.0_half.csv", "BIG_SPX": f"{R4}/tr_SPX500_1330_2.0_half.csv", "BIG_XAU": f"{R4}/tr_XAUUSD_1000_2.0_half.csv"}
END = {"LDN_JPY": "17:05", "ORB_DAX": "17:35", "NOISE_NAS": "16:05", "BIG_NAS": "16:05", "BIG_SPX": "16:05", "BIG_XAU": "16:05"}
ok_all = True
for S in I.SYSTEMS:
    m5 = pd.read_parquet(f"{M5}/{FILES[S['yf']]}_m5.parquet").astype(float)
    m5.index = m5.index.tz_convert("UTC")
    ref = pd.read_csv(REF[S["id"]], parse_dates=["day"])
    ref = ref[(ref.day >= "2025-01-01") & (ref.day < "2025-07-01")]
    if S["id"] == "NOISE_NAS":     # several trades a day: compare the day's sum
        ref = ref.groupby("day").agg(R=("R", "sum"), dir=("dir", "first")).reset_index()
    days = pd.bdate_range("2025-01-02", "2025-06-27")
    match = miss = extra = rdiff = 0
    for d in days:
        sub = m5[m5.index < pd.Timestamp(d + pd.Timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)]
        sub = sub[sub.index > sub.index[-1] - pd.Timedelta(days=130)]
        now = I._at(d.date(), END[S["id"]], S["tz"])
        r = I.FN[S["kind"]](S, sub, now)
        if S["id"] == "NOISE_NAS":
            R = sum(t["R"] for t in r.get("trades", []) if t["state"] == "CLOSED") if r.get("trades") else None
            di = (1 if r["trades"][0]["dir"] == "LONG" else -1) if r.get("trades") else None
        else:
            t = r.get("trade")
            R = t.get("R") if t else None
            di = (0 if t.get("outcome") == "whipsaw" else (1 if t.get("dir") == "LONG" else -1)) if t else None
        rr = ref[ref.day == d]
        if len(rr) and R is not None:
            match += 1
            if abs(rr.R.iloc[0] - R) > 0.02 or int(rr.dir.iloc[0]) != di:
                rdiff += 1
                if rdiff <= 3:
                    print("   diff", S["id"], d.date(), "ref", round(rr.R.iloc[0], 3), int(rr.dir.iloc[0]), "live", R, di, r["status"])
        elif len(rr):
            miss += 1
            if miss <= 3:
                print("   missed", S["id"], d.date(), round(rr.R.iloc[0], 3), r["status"], r["status_text"])
        elif R is not None:
            extra += 1
            if extra <= 3:
                print("   extra", S["id"], d.date(), R, r["status"])
    # known, intended difference: the day after a market holiday the live engine uses the last real close
    # (research skipped those days because the holiday had no close)
    known = {"2025-04-21"}
    ok = rdiff == 0 and miss == 0 and extra <= sum(1 for d in known if S["kind"] == "bigmove")
    ok_all &= ok
    print(f"{S['id']:10s} research trades H1-2025={len(ref):3d}  matched={match:3d}  R/dir differ={rdiff}  missed={miss}  extra={extra}  {'OK' if ok else 'CHECK'}")
print("ALL OK" if ok_all else "DIFFERENCES FOUND")

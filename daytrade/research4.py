"""
Round 4 — "big-move day continuation into the close" (market intraday momentum).

Hypothesis (literature, not data-mined): Gao, Han, Li & Zhou (2018) and Baltussen, Da, Lammers &
Martens (2021): when a market has already moved a lot versus yesterday's close, the move tends to
continue into the close because option-dealer gamma hedging and leveraged-ETF rebalancing must
trade in the direction of the move late in the day.

Rule: at decision time T, z = (price_T / previous close - 1) / sigma, sigma = past-60-day std of the
same quantity. If |z| >= k -> enter in the direction of the move at the next 5-min bar open, exit at the
session close. Optional protective stop = half of the day's move retraced (that distance is 1R).
Costs: round-trip spread+slippage in bps per market. Fills: stop first if a bar hits it; gaps at open.
"""
from __future__ import annotations
import os, sys, math, itertools
import numpy as np
import pandas as pd
from scipy import stats as st

DATA = os.environ.get("M5_DIR", "m5")
US = dict(tz="America/New_York", close="16:00", T=("12:00", "13:30", "15:00", "15:30"))
EU = dict(tz="Europe/Berlin", close="17:30", T=("13:30", "15:00", "16:30", "17:00"))
FX = dict(tz="America/New_York", close="16:00", T=("10:00", "12:00", "14:00", "15:00"))
CFG = {
    "NAS100": ("usatechidxusd", 1.5, US), "SPX500": ("usa500idxusd", 1.5, US), "GER40": ("deuidxeur", 1.5, EU),
    "EURUSD": ("eurusd", 1.1, FX), "GBPUSD": ("gbpusd", 1.4, FX), "USDJPY": ("usdjpy", 1.2, FX),
    "AUDUSD": ("audusd", 1.6, FX), "USDCAD": ("usdcad", 1.5, FX), "USDCHF": ("usdchf", 1.8, FX),
    "XAUUSD": ("xauusd", 1.5, FX), "XAGUSD": ("xagusd", 4.0, FX), "WTI": ("lightcmdusd", 4.0, FX),
    "BTCUSD": ("btcusd", 8.0, FX), "ETHUSD": ("ethusd", 10.0, FX), "USTBOND": ("ustbondtrusd", 2.0, FX),
}
KS = (1.0, 1.5, 2.0)
PERIODS = (("2014-19", "2014", "2019"), ("2020-22", "2020", "2022"), ("2023-26", "2023", "2026"))


def mins(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def prepare(mid):
    f, cost, c = CFG[mid]
    df = pd.read_parquet(os.path.join(DATA, f"{f}_m5.parquet")).astype("float64")
    df.index = df.index.tz_convert(c["tz"])
    mod = df.index.hour * 60 + df.index.minute
    cm = mins(c["close"])
    # trading day = the 24h ending at the session close
    day = (df.index + pd.Timedelta(minutes=24 * 60 - cm)).normalize().tz_localize(None)
    df = df.assign(day=day, mod=mod)
    if mid not in ("BTCUSD", "ETHUSD"):
        df = df[pd.DatetimeIndex(df.day).dayofweek < 5]
    return df, cost * 1e-4, c, cm


def run(mid):
    df, cost, c, cm = prepare(mid)
    days = {d: g for d, g in df.groupby("day")}
    order = sorted(days)
    # previous close = close of the last bar before the session close time
    def close_px(g):
        x = g[(g["mod"] < cm) & (g["mod"] >= cm - 60)] if cm >= 60 else g
        return x.close.iloc[-1] if len(x) else np.nan
    closes = pd.Series({d: close_px(days[d]) for d in order})
    prevc = closes.shift(1)
    out = {}
    for T in c["T"]:
        tm = mins(T)
        # price at T = close of the bar ending at T
        pT = pd.Series({d: (lambda x: x.close.iloc[-1] if len(x) else np.nan)(days[d][(days[d]["mod"] < tm) & (days[d]["mod"] >= tm - 30)]) for d in order})
        move = pT / prevc - 1
        sig = move.rolling(60, min_periods=40).std().shift(1)
        z = move / sig
        for d in order:
            zz = z.get(d)
            if zz != zz or abs(zz) < min(KS):
                continue
            g = days[d]
            w = g[(g["mod"] >= tm) & (g["mod"] < cm)]
            if len(w) < 2:
                continue
            di = 1.0 if zz > 0 else -1.0
            o, h, l, cl = w.open.values, w.high.values, w.low.values, w.close.values
            entry = o[0]
            risk = 0.5 * abs(move[d]) * entry
            if risk <= 0:
                continue
            stop = entry - di * risk
            ex_ns = cl[-1]
            ex_s = cl[-1]
            for j in range(len(w)):
                if j > 0 and di * (o[j] - stop) <= 0:
                    ex_s = o[j]; break
                if (l[j] <= stop) if di > 0 else (h[j] >= stop):
                    ex_s = stop; break
            for stopmode, ex in (("none", ex_ns), ("half", ex_s)):
                ret = di * (ex / entry - 1) - cost
                out.setdefault((T, stopmode), []).append((d, abs(zz), di, ret, ret * entry / risk))
    return {k: pd.DataFrame(v, columns=["day", "z", "dir", "ret", "R"]) for k, v in out.items()}


def summarize(tr):
    n = len(tr)
    if n < 30:
        return None
    r = tr.ret.values
    t = r.mean() / (r.std(ddof=1) / math.sqrt(n))
    row = dict(n=n, per_year=round(n / max(1, (tr.day.max() - tr.day.min()).days / 365.25), 1), hit=round(float((r > 0).mean()), 3),
               net_bp=round(r.mean() * 1e4, 2), t=round(t, 2), p=float(st.norm.sf(t)), meanR=round(tr.R.mean(), 3),
               avg_win_R=round(tr.R[tr.R > 0].mean(), 2), avg_loss_R=round(tr.R[tr.R <= 0].mean(), 2))
    for name, a, b in PERIODS:
        x = tr[(tr.day >= a) & (tr.day <= b + "-12-31")].ret
        row[f"bp_{name}"] = round(x.mean() * 1e4, 1) if len(x) >= 10 else None
        row[f"n_{name}"] = len(x)
    yr = tr.groupby(tr.day.dt.year).ret.sum()
    row["pos_years"] = f"{int((yr > 0).sum())}/{len(yr)}"
    return row


def main(out="rout4"):
    os.makedirs(out, exist_ok=True)
    rows = []
    for mid in CFG:
        try:
            res = run(mid)
        except Exception as e:
            print("skip", mid, repr(e)[:100], flush=True)
            continue
        for (T, sm), tr in res.items():
            for k in KS:
                s = summarize(tr[tr.z >= k])
                if s:
                    rows.append(dict(market=mid, T=T, k=k, stop=sm, **s))
                    tr[tr.z >= k].to_csv(os.path.join(out, f"tr_{mid}_{T.replace(':','')}_{k}_{sm}.csv"), index=False)
        print("done", mid, flush=True)
    r = pd.DataFrame(rows)
    o = np.argsort(r.p.values); m = len(r)
    ok = r.p.values[o] <= 0.05 * np.arange(1, m + 1) / m
    kk = np.where(ok)[0].max() + 1 if ok.any() else 0
    r["fdr5"] = False
    r.iloc[o[:kk], r.columns.get_loc("fdr5")] = True
    r.to_csv(os.path.join(out, "results.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print("tests", m, "FDR5% pass", int(r.fdr5.sum()))
    print(r.sort_values("t", ascending=False).head(45).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

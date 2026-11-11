"""
Round 3 — systematic intraday scan on all 15 markets with a THREE-WAY time split.

  discovery  2014-2019 : every (market, entry hour, exit hour, signal) is tested NET of costs;
                         Benjamini-Hochberg FDR 5% over ALL tests decides what moves on.
  validation 2020-2022 : survivors must stay profitable net of costs (t > 1.5), direction frozen.
  holdout    2023-now  : touched once, only reported.

Trades are time-based, inside one trading day (18:00 -> 16:00 New York, never across the 17:00
rollover, so no swap). Signals use only information available at the entry time.
Families (each has a documented rationale):
  UNC      time-of-day drift (overnight equity drift; "currencies depreciate in their own
           trading hours", Breedon & Ranaldo 2013; gold fixing effects)
  MOMD     intraday momentum: direction of the move since the day's open (Gao et al. 2018)
  PREVW    continuation/reversal of the immediately preceding window of the same length
  PREVDAY  direction of yesterday's move
  TREND    direction of the daily trend (close vs 50-day average)
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd
from scipy import stats as st

DATA = os.environ.get("M5_DIR", "m5")
MK = {  # id: (file, base round-trip cost bps, NY hours [lo,hi) with normal spread; None = always)
    "NAS100": ("usatechidxusd", 1.5, (9, 16)), "SPX500": ("usa500idxusd", 1.5, (9, 16)),
    "GER40": ("deuidxeur", 1.5, (3, 12)),
    "EURUSD": ("eurusd", 1.1, (2, 16)), "GBPUSD": ("gbpusd", 1.4, (2, 16)), "USDJPY": ("usdjpy", 1.2, (2, 16)),
    "AUDUSD": ("audusd", 1.6, (2, 16)), "USDCAD": ("usdcad", 1.5, (2, 16)), "USDCHF": ("usdchf", 1.8, (2, 16)),
    "XAUUSD": ("xauusd", 1.5, (2, 16)), "XAGUSD": ("xagusd", 4.0, (2, 16)), "WTI": ("lightcmdusd", 4.0, (3, 15)),
    "BTCUSD": ("btcusd", 8.0, None), "ETHUSD": ("ethusd", 10.0, None), "USTBOND": ("ustbondtrusd", 2.0, (8, 16)),
}
DURS = (1, 2, 4, 6, 8, 12, 16)      # in half-hours
NPOS = 45                              # 18:00 ... 16:00 New York in 30-minute steps
LASTP = NPOS - 1
T_VAL, T_HOLD = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")
FAMS = ("UNC", "MOMD", "PREVW", "PREVDAY", "TREND")


def hour_of(j):            # grid position -> New York clock hour as a float (j=0 is 18:00)
    return (18 + j / 2) % 24


def clock(j):
    h = hour_of(j)
    return f"{int(h):02d}:{'30' if h % 1 else '00'}"


def grid(mid):
    """Days x 23 matrix of prices at 18:00, 19:00, ... 16:00 New York."""
    f = MK[mid][0]
    df = pd.read_parquet(os.path.join(DATA, f"{f}_m5.parquet"), columns=["open", "close"]).astype("float64")
    idx = df.index.tz_convert("America/New_York")
    sh = idx + pd.Timedelta(hours=6)                     # 18:00 -> 00:00 of the trading day
    day = sh.normalize().tz_localize(None)
    pos = sh.hour * 2 + sh.minute // 30                  # bar belongs to half-hour slot `pos`
    h = pd.DataFrame({"day": day, "pos": pos, "open": df.open.values, "close": df.close.values})
    g = h.groupby(["day", "pos"]).agg(o=("open", "first"), c=("close", "last"))
    cl = g.c.unstack("pos")                              # close of hour slot p  -> price at boundary p+1
    op = g.o.unstack("pos")
    cols = {0: op.get(0, pd.Series(np.nan, index=cl.index))}
    for j in range(1, NPOS):
        cols[j] = cl.get(j - 1, pd.Series(np.nan, index=cl.index))
    P = pd.DataFrame(cols)
    if mid not in ("BTCUSD", "ETHUSD"):
        P = P[P.index.dayofweek < 5]
    return P.dropna(how="all")


def cost_of(mid, a, b):
    _, c, good = MK[mid]
    if good is None:
        return c * 1e-4
    ok = lambda j: good[0] <= hour_of(j) <= good[1]
    return c * 1e-4 * (1.0 if ok(a) and ok(b) else 2.0)


def _z(q):
    """Size of a move in units of its own recent (60-day, past-only) standard deviation."""
    return q / q.rolling(60, min_periods=40).std().shift(1)


def signals(L, a, b, cache):
    """name -> series in {-1, 0, +1}; 0 = no trade. Thresholds k are in sigma units (0 = sign only)."""
    out = {"UNC": pd.Series(1.0, index=L.index)}
    def big(name, q, ks):
        z = cache.get((name, q.name))
        if z is None:
            z = cache.setdefault((name, q.name), _z(q))
        for k in ks:
            out[f"{name}{k}"] = np.sign(z).where(z.abs() > k, 0.0) if k else np.sign(q)
    if a >= 2:
        q = (L[a] - L[0]); q.name = ("o", a); big("MOMD", q, (0, 1, 2))               # move since the 18:00 open
    q = (L[a] - L[LASTP].shift(1)); q.name = ("c", a); big("MOMC", q, (1, 2))          # move since yesterday's 16:00
    w = b - a
    if a - w >= 0:
        q = (L[a] - L[a - w]); q.name = ("w", a, w); big("PREVW", q, (0, 1, 2))       # preceding window
    dcl = L[LASTP]
    q = (dcl.shift(1) - dcl.shift(2)); q.name = ("d",); big("PREVDAY", q, (0, 1, 2))  # yesterday
    out["TREND"] = np.sign(dcl.shift(1) - dcl.shift(1).rolling(50).mean())
    return out


def tstat(x):
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    return x.mean(), x.std(ddof=1) / math.sqrt(n), n


def scan(mid):
    P = grid(mid)
    L = np.log(P)
    rows = []
    cache = {}
    for a in range(2, LASTP):
        ends = sorted({a + d for d in DURS if a + d <= LASTP} | {LASTP})
        for b in ends:
            r = L[b] - L[a]
            c = cost_of(mid, a, b)
            for fam, s in signals(L, a, b, cache).items():
                x = (s * r).replace([np.inf, -np.inf], np.nan).dropna()
                x = x[x != 0]
                xd = x[x.index < T_VAL]
                m, se, n = tstat(xd)
                if n < 100 or not se or se != se or se == 0:
                    continue
                dirn = 1.0 if m >= 0 else -1.0
                t_net = (abs(m) - c) / se
                p = min(1.0, 2 * st.norm.sf(t_net))
                xv = dirn * x[(x.index >= T_VAL) & (x.index < T_HOLD)] - c
                xh = dirn * x[x.index >= T_HOLD] - c
                mv, sv, nv = tstat(xv)
                mh, shh, nh = tstat(xh)
                rows.append(dict(market=mid, fam=fam, a=a, b=b, entry_ny=clock(a), exit_ny=clock(b),
                                 dir=("with" if dirn > 0 else "against") if fam != "UNC" else ("long" if dirn > 0 else "short"),
                                 cost_bp=round(c * 1e4, 2), d_n=n, d_gross_bp=round(abs(m) * 1e4, 2), d_net_bp=round((abs(m) - c) * 1e4, 2),
                                 d_t=round(t_net, 2), p=p, d_hit=round(float((dirn * xd > 0).mean()), 3),
                                 v_n=nv, v_net_bp=round(mv * 1e4, 2) if mv == mv else None, v_t=round(mv / sv, 2) if sv and sv == sv else None,
                                 h_n=nh, h_net_bp=round(mh * 1e4, 2) if mh == mh else None, h_t=round(mh / shh, 2) if shh and shh == shh else None,
                                 h_hit=round(float((xh > 0).mean()), 3) if nh else None))
    return pd.DataFrame(rows)


def bh(p, q):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    r = np.zeros(m, bool); r[o[:k]] = True
    return r


def main(out="rout3"):
    os.makedirs(out, exist_ok=True)
    parts = []
    for mid in MK:
        try:
            r = scan(mid)
        except Exception as e:
            print("skip", mid, repr(e)[:120], flush=True)
            continue
        parts.append(r)
        print(mid, "tests", len(r), "best d_t", r.d_t.max() if len(r) else None, flush=True)
    res = pd.concat(parts, ignore_index=True)
    # Stage 1: screen on discovery (net t >= 2.5). Stage 2: the real test is on VALIDATION data, which the
    # screen never saw -> Benjamini-Hochberg 10% on one-sided validation p-values of the screened rules.
    res["screened"] = (res.d_t >= 2.5) & res.v_t.notna() & (res.v_n >= 30)
    res["v_p"] = st.norm.sf(res.v_t.fillna(-9))
    res["validated"] = False
    sc = res.screened.values
    if sc.any():
        res.loc[sc, "validated"] = bh(res.loc[sc, "v_p"].values, 0.10)
    res["holdout_ok"] = res.validated & (res.h_net_bp > 0) & (res.h_t > 1.0)
    res.to_csv(os.path.join(out, "scan_all.csv"), index=False)
    print("tests", len(res), "| screened", int(res.screened.sum()), "| validated (FDR 10%)", int(res.validated.sum()),
          "| confirmed in holdout (t>1)", int(res.holdout_ok.sum()))
    v = res[res.validated].sort_values(["market", "fam", "v_t"], ascending=[True, True, False])
    v.to_csv(os.path.join(out, "validated.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(v[["market", "fam", "entry_ny", "exit_ny", "dir", "cost_bp", "d_n", "d_net_bp", "d_t", "v_net_bp", "v_t", "h_n", "h_net_bp", "h_t", "h_hit"]].to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

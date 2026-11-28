"""
Round 5 — HIGH reward-to-risk setups (daily bars built from Dukascopy M5, NY 17:00 close, 2014 -> now).
Pre-declared rules, both directions, 15 markets. Entry next open; stops intrabar (stop wins ties,
gaps fill at the open); costs deducted. R = profit / initial risk.
  T55   55-day breakout, initial stop 2 ATR, exit on a close beyond the opposite 20-day channel (Turtle)
  T20   20-day breakout, initial stop 2 ATR, exit on the opposite 10-day channel
  B3R   55-day breakout, stop 2 ATR, target 3R, max 60 days
  B5R   55-day breakout, stop 1.5 ATR, target 5R, max 90 days
  SQZ   volatility-squeeze breakout, stop 1.5 ATR, chandelier trail 3 ATR from the best close
  PB3R  pullback in trend (RSI2 <10 / >90 with 200-day trend), stop 1 ATR, target 3R, max 10 days
"""
import os, sys, math
import numpy as np, pandas as pd
from scipy import stats as st
DATA = os.environ.get("M5_DIR", "m5")
MK = {"NAS100": ("usatechidxusd", 1.5), "SPX500": ("usa500idxusd", 1.5), "GER40": ("deuidxeur", 1.5),
      "EURUSD": ("eurusd", 1.1), "GBPUSD": ("gbpusd", 1.4), "USDJPY": ("usdjpy", 1.2), "AUDUSD": ("audusd", 1.6),
      "USDCAD": ("usdcad", 1.5), "USDCHF": ("usdchf", 1.8), "XAUUSD": ("xauusd", 1.5), "XAGUSD": ("xagusd", 4.0),
      "WTI": ("lightcmdusd", 4.0), "BTCUSD": ("btcusd", 8.0), "ETHUSD": ("ethusd", 10.0), "USTBOND": ("ustbondtrusd", 2.0)}
SPLIT = pd.Timestamp("2020-01-01")

def daily(mid):
    df = pd.read_parquet(os.path.join(DATA, MK[mid][0] + "_m5.parquet")).astype("float64")
    idx = df.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    d = df.groupby(idx.normalize().tz_localize(None)).agg(O=("open", "first"), H=("high", "max"), L=("low", "min"), C=("close", "last"), n=("close", "size"))
    d = d[d.n >= 60]
    if mid not in ("BTCUSD", "ETHUSD"):
        d = d[d.index.dayofweek < 5]
    pc = d.C.shift()
    d["atr"] = pd.concat([d.H - d.L, (d.H - pc).abs(), (d.L - pc).abs()], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()
    for n in (10, 20, 55):
        d[f"hh{n}"] = d.H.rolling(n).max().shift(1); d[f"ll{n}"] = d.L.rolling(n).min().shift(1)
    d["sma200"] = d.C.rolling(200).mean()
    dl = d.C.diff(); up = dl.clip(lower=0).ewm(alpha=.5, adjust=False).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=.5, adjust=False).mean()
    d["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    bw = 4 * d.C.rolling(20).std() / d.C.rolling(20).mean()
    d["sq"] = (bw.rolling(252, min_periods=120).rank(pct=True) < .15).rolling(5).max().fillna(0).astype(bool)
    return d.dropna(subset=["atr", "hh55", "sma200"])

def sig(d, s, di):
    if s in ("T55", "B3R", "B5R"): return (d.C > d.hh55) if di > 0 else (d.C < d.ll55)
    if s == "T20": return (d.C > d.hh20) if di > 0 else (d.C < d.ll20)
    if s == "SQZ": return d.sq & ((d.C > d.hh20) if di > 0 else (d.C < d.ll20))
    if s == "PB3R": return ((d.C > d.sma200) & (d.rsi2 < 10)) if di > 0 else ((d.C < d.sma200) & (d.rsi2 > 90))

P = {"T55": (2, None, 250, "ch20"), "T20": (2, None, 250, "ch10"), "B3R": (2, 3, 60, None), "B5R": (1.5, 5, 90, None),
     "SQZ": (1.5, None, 250, "chand"), "PB3R": (1, 3, 10, None)}

def run(d, s, di, cost):
    slm, tgt, mx, trail = P[s]
    O, H, L, C, A = d.O.values, d.H.values, d.L.values, d.C.values, d.atr.values
    ex_hi = d.hh20.values if trail == "ch20" else d.hh10.values
    ex_lo = d.ll20.values if trail == "ch20" else d.ll10.values
    S = sig(d, s, di).values; n = len(d); out = []; i = 0
    while i < n - 2:
        if not S[i]: i += 1; continue
        E = O[i + 1]; risk = slm * A[i]; stop = E - di * risk; T = E + di * tgt * risk if tgt else None
        best = C[i]; px = None; k = i + 1
        while k < n:
            if k > i + 1 and di * (O[k] - stop) <= 0: px = O[k]; break
            if (L[k] <= stop) if di > 0 else (H[k] >= stop): px = stop; break
            if T is not None and ((H[k] >= T) if di > 0 else (L[k] <= T)): px = T; break
            if trail in ("ch20", "ch10") and ((C[k] < ex_lo[k]) if di > 0 else (C[k] > ex_hi[k])):
                if k + 1 < n: px = O[k + 1]; k += 1
                else: px = C[k]
                break
            if trail == "chand":
                best = max(best, C[k]) if di > 0 else min(best, C[k])
                stop = max(stop, best - 3 * A[k]) if di > 0 else min(stop, best + 3 * A[k])
            if k - i >= mx: px = C[k]; break
            k += 1
        if px is None: break                      # still open at end of data
        out.append((d.index[i], di * (px - E) / risk - cost * 1e-4 * E / risk, k - i))
        i = k + 1
    return pd.DataFrame(out, columns=["day", "R", "bars"])

def stats(t):
    n = len(t)
    if n < 15: return None
    R = t.R.values; w = R[R > 0]; l = R[R <= 0]
    tt = R.mean() / (R.std(ddof=1) / math.sqrt(n))
    a, b = t[t.day < SPLIT].R, t[t.day >= SPLIT].R
    eq = np.cumsum(R)
    return dict(n=n, hit=round(len(w) / n, 3), avg_win=round(w.mean(), 2) if len(w) else 0, avg_loss=round(l.mean(), 2) if len(l) else 0,
                rr=round(w.mean() / -l.mean(), 2) if len(w) and len(l) else None, meanR=round(R.mean(), 3), t=round(tt, 2), p=float(st.norm.sf(tt)),
                r_2014_19=round(a.mean(), 2) if len(a) >= 5 else None, n1=len(a), r_2020_26=round(b.mean(), 2) if len(b) >= 5 else None, n2=len(b),
                best=round(R.max(), 1), maxdd=round(float((eq - np.maximum.accumulate(eq)).min()), 1), avg_days=round(t.bars.mean(), 1))

rows, allt = [], []
for m in MK:
    d = daily(m)
    for s in P:
        for di in (1, -1):
            t = run(d, s, di, MK[m][1]); t["market"], t["setup"], t["dir"] = m, s, di; allt.append(t)
            x = stats(t)
            if x: rows.append(dict(market=m, setup=s, dir="L" if di > 0 else "S", **x))
    print("done", m, d.index[0].date(), len(d), flush=True)
r = pd.DataFrame(rows); A = pd.concat(allt)
o = np.argsort(r.p.values); ok = r.p.values[o] <= 0.10 * np.arange(1, len(r) + 1) / len(r)
r["fdr10"] = False
if ok.any(): r.iloc[o[:np.where(ok)[0].max() + 1], r.columns.get_loc("fdr10")] = True
out = sys.argv[1] if len(sys.argv) > 1 else "rout5"
r.sort_values("t", ascending=False).to_csv(out + "/high_rr_results.csv", index=False); A.to_csv(out + "/high_rr_trades.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print("\nPOOLED over all 15 markets")
pool = [dict(setup=s, dir="L" if di > 0 else "S", **stats(A[(A.setup == s) & (A.dir == di)].sort_values("day"))) for s in P for di in (1, -1)]
print(pd.DataFrame(pool).drop(columns=["p"]).to_string(index=False))
print("\nPER MARKET, top 30 of", len(r), "tests | FDR10% pass:", int(r.fdr10.sum()))
print(r.sort_values("t", ascending=False).head(30).drop(columns=["p", "n1", "n2"]).to_string(index=False))

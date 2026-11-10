"""
Round 6 — market drivers: dealer gamma (GEX/DIX), volatility regime, and forced calendar flows.

Every hypothesis below is written BEFORE looking at results, with its expected sign (one-sided tests).
All trades are net of costs (spread + slippage; + 2 bp per night held on index CFDs as financing).
Periods: 2014-19 / 2020-22 / 2023-26 must all be reported. Benjamini-Hochberg FDR across every test.

A. Conditioning the existing US systems (factor values known the evening before):
  A1  big-move continuation (NAS/SPX at 13:30) is STRONGER when GEX is low (below its 1-year median)
  A2  ... STRONGER when VIX term structure is inverted or VIX is above its 1-year median
  A3  Nasdaq noise-band momentum is STRONGER when GEX is low
B. Forced flows (new standalone trades):
  B1  month-end pension rebalancing: last US trading day, trade the S&P/Nasdaq AGAINST the month-to-date
      stock-minus-bond performance (sign known at the previous close). Windows: prev close->close, 09:30->16:00
  B2  FX month-end fix (Melvin & Prins 2015): last business day, 14:00->16:00 London. If US stocks beat
      German stocks month-to-date, foreign holders sell USD into the 4pm fix -> long EUR/USD, GBP/USD, short USD/JPY
  B3  quarter-end only versions of B1 and B2 (bigger rebalancing)
  B4  option-expiration week: long S&P/Nasdaq from the Friday close before OPEX week to the OPEX Friday close
  B5  pre-holiday: long S&P/Nasdaq on the last trading day before a US market holiday (prev close -> close)
  B6  pre-FOMC drift (Lucca & Moench 2015): long S&P/Nasdaq 14:00 NY the day before -> 14:00 announcement day
  B7  overnight drift: long S&P/Nasdaq 16:00 -> 09:30 next session, every night
"""
from __future__ import annotations
import os, sys, re, glob, math, json
import numpy as np
import pandas as pd
from scipy import stats as st

M5 = os.environ.get("M5_DIR", "m5")
RD = os.environ.get("RD_DIR", "research_data")
R1 = os.environ.get("R1", "rout")
R4 = os.environ.get("R4", "rout4")
NY, LDN, FRA = "America/New_York", "Europe/London", "Europe/Berlin"
COST = {"SPX500": 1.5, "NAS100": 1.5, "EURUSD": 1.1, "GBPUSD": 1.4, "USDJPY": 1.2}
NIGHT = {"SPX500": 2.0, "NAS100": 2.0, "EURUSD": 0.5, "GBPUSD": 0.5, "USDJPY": 0.5}
FILE = {"SPX500": "usa500idxusd", "NAS100": "usatechidxusd", "GER40": "deuidxeur", "EURUSD": "eurusd",
        "GBPUSD": "gbpusd", "USDJPY": "usdjpy"}
PER = (("2014-19", "2014-01-01", "2019-12-31"), ("2020-22", "2020-01-01", "2022-12-31"), ("2023-26", "2023-01-01", "2026-12-31"))


# ------------------------------------------------------------------ data helpers
def m5(mid):
    df = pd.read_parquet(os.path.join(M5, FILE[mid] + "_m5.parquet")).astype(float)
    df.index = df.index.tz_convert("UTC")
    return df


def real_days(df, tz, s0, s1):
    """Dates (local) with a genuine session between s0 and s1 (not weekend, not flat filler)."""
    x = df.tz_convert(tz)
    t = x.index.strftime("%H:%M")
    w = x[(t >= s0) & (t < s1)]
    ok = w.groupby(w.index.date).apply(lambda g: len(g) >= 6 and (g.high.max() > g.low.min()) and ((g.high - g.low) > 0).mean() > 0.5)
    ok = ok[ok]
    return sorted(d for d in ok.index if d.weekday() < 5)


def px_at(df, tz, hhmm, window=30):
    """Per local date: close of the last 5-min bar ending in (T-window, T]."""
    x = df.tz_convert(tz)
    end = x.index + pd.Timedelta(minutes=5)
    m = end.hour * 60 + end.minute
    h, mm = map(int, hhmm.split(":"))
    T = h * 60 + mm
    sel = (m <= T) & (m > T - window) & (end.date == x.index.date)
    return x.close[sel].groupby(end[sel].date).last()


def yf_close(name):
    f = os.path.join(RD, f"YF_{name}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, index_col=0)
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[d.index.notna()]
    c = pd.to_numeric(d["Close"], errors="coerce").dropna()
    c.index = c.index.date
    return c


def summarize(r: pd.Series, name, expect=+1):
    r = r.dropna()
    n = len(r)
    if n < 20:
        return None
    x = expect * r.values
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n)) if x.std() > 0 else 0
    out = dict(test=name, n=n, hit=round(float((x > 0).mean()), 3), net_bp=round(x.mean() * 1e4, 2), t=round(t, 2), p=float(st.norm.sf(t)))
    idx = pd.to_datetime(pd.Series(r.index))
    for k, a, b in PER:
        sel = ((idx >= a) & (idx <= b)).values
        out[f"bp_{k}"] = round(x[sel].mean() * 1e4, 1) if sel.sum() >= 5 else None
        out[f"n_{k}"] = int(sel.sum())
    return out


# ------------------------------------------------------------------ factors
def factors():
    f = {}
    dix = os.path.join(RD, "DIX.csv")
    if os.path.exists(dix):
        d = pd.read_csv(dix)
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d = d.set_index("date")
        g = d["gex"].astype(float)
        gs = pd.Series(g.values, index=pd.to_datetime(g.index))
        med = gs.rolling(252, min_periods=120).median()
        f["gex_low"] = pd.Series((gs < med).values, index=g.index).where(med.notna().values)
        f["gex"] = g
        f["dix"] = d["dix"].astype(float)
    vix, v3 = yf_close("VIX"), yf_close("VIX3M")
    if vix is not None:
        vs = pd.Series(vix.values, index=pd.to_datetime(vix.index))
        f["vix_high"] = pd.Series((vs > vs.rolling(252, min_periods=120).median()).values, index=vix.index)
        if v3 is not None:
            r = (vix / v3).dropna()
            f["vix_inverted"] = r >= 1.0
    return f


def prev_day_factor(s, days):
    """Value known the evening before each trade day."""
    s = s.sort_index()
    idx = pd.Index(s.index)
    out = {}
    for d in days:
        pos = idx.searchsorted(d) - 1
        out[d] = s.iloc[pos] if pos >= 0 else np.nan
    return pd.Series(out)


def part_a(F):
    rows = []
    trades = {
        "BIGNAS_k1": f"{R4}/tr_NAS100_1330_1.0_half.csv", "BIGNAS_k1.5": f"{R4}/tr_NAS100_1330_1.5_half.csv", "BIGNAS_k2": f"{R4}/tr_NAS100_1330_2.0_half.csv",
        "BIGSPX_k1": f"{R4}/tr_SPX500_1330_1.0_half.csv", "BIGSPX_k1.5": f"{R4}/tr_SPX500_1330_1.5_half.csv", "BIGSPX_k2": f"{R4}/tr_SPX500_1330_2.0_half.csv",
        "NOISE_NAS": f"{R1}/trades_NOISE_CLEAN__NAS100.csv",
    }
    for name, path in trades.items():
        if not os.path.exists(path):
            continue
        t = pd.read_csv(path, parse_dates=["day"])
        if "ret" not in t:
            t["ret"] = t["R"] * 1e-4      # R-based file: test in R units (scaled so bp columns read as R x 1e4... see unit)
            unit = "R"
        else:
            unit = "ret"
        t = t.groupby("day").agg(ret=("ret", "sum"), R=("R", "sum"))
        days = [d.date() for d in t.index]
        for fac, expect_label in (("gex_low", "low GEX"), ("vix_high", "VIX above 1y median"), ("vix_inverted", "VIX curve inverted")):
            if fac not in F:
                continue
            if name == "NOISE_NAS" and fac != "gex_low":
                continue
            flag = prev_day_factor(F[fac].astype(float), days).values
            ok = ~np.isnan(flag)
            on, off = t[ok & (flag == 1)], t[ok & (flag == 0)]
            if len(on) < 20 or len(off) < 20:
                continue
            a, b = on.R.values, off.R.values
            tt = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            s_on = summarize(pd.Series(on.R.values * 1e-4, index=on.index), f"A|{name}|{expect_label}: ON")
            row = dict(test=f"A|{name}|{expect_label}", n_on=len(a), n_off=len(b), R_on=round(a.mean(), 3), R_off=round(b.mean(), 3),
                       hit_on=round((a > 0).mean(), 3), hit_off=round((b > 0).mean(), 3), t=round(tt, 2), p=float(st.norm.sf(tt)))
            for k in ("2014-19", "2020-22", "2023-26"):
                row[f"Ron_{k}"] = None if s_on is None or s_on[f"bp_{k}"] is None else round(s_on[f"bp_{k}"] / 1e4 * 1e4 / 1e4, 3)
            # per period difference
            idx = on.index
            for k, a0, b0 in PER:
                x = on[(on.index >= a0) & (on.index <= b0)].R
                y = off[(off.index >= a0) & (off.index <= b0)].R
                row[f"on_{k}"] = round(x.mean(), 3) if len(x) >= 5 else None
                row[f"off_{k}"] = round(y.mean(), 3) if len(y) >= 5 else None
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calendar flows
def last_days_of_month(days):
    s = pd.Series(days, index=pd.to_datetime(days))
    return set(s.groupby(s.index.to_period("M")).max().values)


def fomc_dates():
    """Announcement days parsed from federalreserve.gov pages (scheduled meetings; 2-day meetings -> second day)."""
    out = set()
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
    for f in sorted(glob.glob(os.path.join(RD, "fomc_*.html"))):
        html = open(f, encoding="utf-8", errors="ignore").read()
        if "current" in f:
            # panels per year: <h4><a id="...">2026 FOMC Meetings</a></h4> then month/date pairs
            for yb in re.split(r"(\d{4}) FOMC Meetings", html)[1:]:
                pass
            parts = re.split(r"(\d{4}) FOMC Meetings", html)
            for i in range(1, len(parts), 2):
                y = int(parts[i]); body = parts[i + 1]
                for mo, dd in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([A-Za-z/]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<', body, flags=re.S):
                    if "*" in dd and "notation" in dd.lower():
                        continue
                    d2 = re.findall(r"\d+", dd)
                    if not d2:
                        continue
                    mname = mo.split("/")[-1]
                    if mname[:3] not in [m[:3] for m in months]:
                        continue
                    m = [v for k, v in months.items() if k[:3] == mname[:3]][0]
                    try:
                        out.add(pd.Timestamp(y, m, int(d2[-1])).date())
                    except ValueError:
                        pass
        else:
            y = int(re.findall(r"fomc_(\d{4})", f)[0])
            for mname, d1, d2 in re.findall(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:-(\d{1,2}))?\s+Meeting", html):
                try:
                    out.add(pd.Timestamp(y, months[mname], int(d2 or d1)).date())
                except ValueError:
                    pass
            for m1, d1, m2, d2 in re.findall(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})-(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\s+Meeting", html):
                out.add(pd.Timestamp(y, months[m2], int(d2)).date())
    return sorted(out)


def part_b():
    rows, trades = [], {}
    us = {}
    for mid in ("SPX500", "NAS100"):
        df = m5(mid)
        days = real_days(df, NY, "09:30", "16:00")
        c16, o930, p14 = px_at(df, NY, "16:00", 60), px_at(df, NY, "09:35", 10), px_at(df, NY, "14:00")
        o930 = px_at(df.assign(close=df.open), NY, "09:35", 10)      # open of the 09:30 bar
        D = pd.DataFrame({"c": c16, "o": o930, "p14": p14}).reindex(days)
        us[mid] = D
    ger = m5("GER40")
    gdays = real_days(ger, FRA, "09:00", "17:30")
    gclose = px_at(ger, FRA, "17:30", 60).reindex(gdays)
    ief = yf_close("IEF")
    bond = None if ief is None else pd.Series(ief.values, index=ief.index)
    # US calendar from the S&P sessions
    spx_days = list(us["SPX500"].index)
    lastd = last_days_of_month(spx_days)
    yahoo_spx = yf_close("GSPC")
    holidays = set()
    if yahoo_spx is not None:
        trade_days = set(yahoo_spx.index)
        for d in pd.bdate_range("2014-01-01", max(trade_days)):
            if d.date() not in trade_days:
                holidays.add(d.date())
    def mtd(series, d):
        """Return from last close of previous month to the close BEFORE d."""
        s = series.dropna()
        prev = s[s.index < d]
        if len(prev) < 25:
            return np.nan
        start = s[[x < pd.Timestamp(d).replace(day=1).date() for x in s.index]]
        if not len(start):
            return np.nan
        return prev.iloc[-1] / start.iloc[-1] - 1
    for mid in ("SPX500", "NAS100"):
        D = us[mid]
        c = D.c
        prevc = c.shift(1)
        cost, night = COST[mid] * 1e-4, NIGHT[mid] * 1e-4
        # B1/B3 month-end rebalancing
        sig, r_cc, r_od, qe = {}, {}, {}, {}
        for d in D.index:
            if d not in lastd or bond is None:
                continue
            eq, bd = mtd(us["SPX500"].c, d), mtd(bond, d)
            if eq != eq or bd != bd:
                continue
            s_ = -np.sign(eq - bd)
            sig[d] = abs(eq - bd)
            r_cc[d] = s_ * (c[d] / prevc[d] - 1) - cost - night
            r_od[d] = s_ * (c[d] / D.o[d] - 1) - cost
            qe[d] = pd.Timestamp(d).month in (3, 6, 9, 12)
        r_cc, r_od, sig, qe = pd.Series(r_cc), pd.Series(r_od), pd.Series(sig), pd.Series(qe)
        for nm, r in (("prevclose->close", r_cc), ("open->close", r_od)):
            rows.append(summarize(r, f"B1|{mid}|month-end rebalance {nm}"))
            rows.append(summarize(r[sig > sig.median()], f"B1|{mid}|month-end rebalance {nm}, big MTD gap"))
            rows.append(summarize(r[qe], f"B3|{mid}|quarter-end rebalance {nm}"))
        trades[f"B1_{mid}_cc"] = r_cc
        # B4 OPEX week
        r = {}
        dl = list(D.index)
        for i, d in enumerate(dl):
            ts = pd.Timestamp(d)
            if ts.weekday() == 4 and 15 <= ts.day <= 21:          # third Friday
                j = i - 1
                while j >= 0 and (ts - pd.Timestamp(dl[j])).days < 7:
                    j -= 1
                if j >= 0:
                    nights = (ts - pd.Timestamp(dl[j])).days
                    r[d] = c[d] / c[dl[j]] - 1 - cost - night * nights
        rows.append(summarize(pd.Series(r), f"B4|{mid}|OPEX week long"))
        trades[f"B4_{mid}"] = pd.Series(r)
        # B5 pre-holiday
        r = {}
        for i, d in enumerate(dl[1:], 1):
            nxt = pd.Timestamp(d) + pd.offsets.BDay(1)
            if nxt.date() in holidays:
                r[d] = c[d] / c[dl[i - 1]] - 1 - cost - night
        rows.append(summarize(pd.Series(r), f"B5|{mid}|pre-holiday long"))
        trades[f"B5_{mid}"] = pd.Series(r)
        # B6 pre-FOMC drift
        fd = set(fomc_dates())
        r = {}
        for i, d in enumerate(dl[1:], 1):
            if d in fd and D.p14[dl[i - 1]] == D.p14[dl[i - 1]] and D.p14[d] == D.p14[d]:
                r[d] = D.p14[d] / D.p14[dl[i - 1]] - 1 - cost - night
        rows.append(summarize(pd.Series(r), f"B6|{mid}|pre-FOMC 14:00->14:00 long"))
        trades[f"B6_{mid}"] = pd.Series(r)
        # B7 overnight drift
        on = (D.o / prevc - 1 - cost - night).iloc[1:]
        rows.append(summarize(on, f"B7|{mid}|overnight 16:00->09:30 long"))
        trades[f"B7_{mid}"] = on
    # B2/B3 FX month-end fix
    for mid, usd_weak_dir in (("EURUSD", +1), ("GBPUSD", +1), ("USDJPY", -1)):
        df = m5(mid)
        p14, p16 = px_at(df, LDN, "14:00"), px_at(df, LDN, "16:00")
        fxdays = [d for d in sorted(set(p14.index) & set(p16.index)) if d.weekday() < 5]
        lastfx = last_days_of_month(fxdays)
        r, qe = {}, {}
        for d in fxdays:
            if d not in lastfx:
                continue
            us_m, de_m = mtd(us["SPX500"].c, d), mtd(gclose, d)
            if us_m != us_m or de_m != de_m:
                continue
            s_ = usd_weak_dir * np.sign(us_m - de_m)        # US outperformed -> USD sold into the fix
            r[d] = s_ * (p16[d] / p14[d] - 1) - COST[mid] * 1e-4
            qe[d] = pd.Timestamp(d).month in (3, 6, 9, 12)
        r, qe = pd.Series(r), pd.Series(qe)
        rows.append(summarize(r, f"B2|{mid}|month-end fix 14:00->16:00 London"))
        rows.append(summarize(r[qe], f"B3|{mid}|quarter-end fix 14:00->16:00 London"))
        trades[f"B2_{mid}"] = r
    return pd.DataFrame([x for x in rows if x]), trades


def bh(p, q=0.10):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    r = np.zeros(m, bool); r[o[:k]] = True
    return r


def main(out="rout6"):
    os.makedirs(out, exist_ok=True)
    F = factors()
    print("factors:", {k: len(v) for k, v in F.items()})
    print("FOMC dates parsed:", len(fomc_dates()), fomc_dates()[:3], fomc_dates()[-3:])
    A = part_a(F)
    B, trades = part_b()
    allp = pd.concat([A[["test", "p"]], B[["test", "p"]]], ignore_index=True)
    passed = set(allp.test[bh(allp.p.values)])
    A["fdr10"] = A.test.isin(passed)
    B["fdr10"] = B.test.isin(passed)
    A.to_csv(os.path.join(out, "A_conditioning.csv"), index=False)
    B.sort_values("t", ascending=False).to_csv(os.path.join(out, "B_flows.csv"), index=False)
    for k, v in trades.items():
        v.to_csv(os.path.join(out, f"tr_{k}.csv"))
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)
    print("\nA. Does the factor make the existing system better? (R per trade when factor ON vs OFF; t of the difference)")
    if len(A):
        print(A.drop(columns=[c for c in A.columns if c.startswith("Ron_")]).sort_values("t", ascending=False).to_string(index=False))
    print("\nB. Forced-flow trades (net bp per trade in the expected direction)")
    print(B.sort_values("t", ascending=False).to_string(index=False))
    print("\nTests:", len(allp), "| FDR 10% pass:", sorted(passed))


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

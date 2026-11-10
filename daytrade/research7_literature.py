"""
Round 7 — hypotheses from the literature review (pre-registered before running; one BH-FDR across all).
Costs: round-trip bps per market; index CFD financing 2 bp per night held (both directions); commodities/crypto 1 bp.

H1  Rebalancing THRESHOLD signal (Harvey-Mazzoleni-Melone 2026): simulate a 60/40 S&P/IEF portfolio; for each
    delta in 0..2.5% (0.1 steps) reset to 60% when |w-60%|>=delta; signal = mean drift across deltas, z-scored
    on the past 252 days. z>+1 -> short S&P at today's close, z<-1 -> long; hold 1 and 3 days.  Expect: profit.
H2  Rebalancing CALENDAR signal: last 4 trading days of the month position = -sign(month-to-date drift);
    first trading day of the next month = reverse. Also the high-VIX subset (VIX > 1y median).  Expect: profit.
H3  Positive-gamma intraday fade (Dim-Eraker-Vilkov): prior-day GEX in its top 1y quintile AND VIX/VIX3M<1.
    At 12:00 NY if |price-open| > 0.6 ATR -> fade at the next bar, target = open, stop 0.5 ATR beyond the
    day's extreme, exit 15:50.  S&P, Nasdaq.  Expect: profit (in R).
H4  Gamma split of the existing 13:30 big-move trades: prior-day GEX in bottom 1y quintile vs the rest.  Expect: low better.
H5  Slow trend on large-tick markets (Kurth-Eisler-Rej-Bouchaud 2026): EWMAC 32/128 + 64/256, forecast capped,
    vol-targeted, on T-bond, gold, silver, WTI, BTC; contrast group S&P, Nasdaq, DAX, 6 FX majors.
    Expect: large-tick Sharpe > 0, small-tick ~ 0. Reported futures-style and with 3%/yr CFD financing.
H6  Low-volatility 55-day breakouts (LeBaron effect): long 55d breakouts (trailing 20d exit) only if 20-day
    realised vol < its 1y 40th percentile at entry, vs the rest.  Expect: low-vol better.
H7  High-VIX index reversal (Nagel 2012): VIX > 1y 80th pct, VIX/VIX3M > 1, index down >=3 days, IBS < 0.2 ->
    buy close; exit first close above 5d MA or after 5 days; stop 2.5 ATR.  S&P, Nasdaq (Yahoo, 2008->).
H8  VIX term-structure normalisation: VIX/VIX3M crosses back below 1 after >=3 days above -> long index at
    close, hold 10 and 20 days, stop 2 ATR.  S&P, Nasdaq (Yahoo, 2008->).
H9  Post-FOMC drift in Treasuries (Indriawan-Jiao-Tse 2021): sign of the T-bond 14:00->16:00 NY move on FOMC
    day -> enter next day 08:30 NY in that direction, hold 5 and 15 days, stop 1.5 ATR.
H10 Dash for cash (Etula et al. 2020): long S&P from the close of T-1 (day before the last trading day) to the
    close of T+3; and short S&P from the close of T-8 to the close of T-4.  Expect: both profitable.
"""
from __future__ import annotations
import os, sys, math, re, glob
import numpy as np
import pandas as pd
from scipy import stats as st

M5 = os.environ.get("M5_DIR", "m5")
RD = os.environ.get("RD_DIR", "research_data")
R4 = os.environ.get("R4", "rout4")
R5 = os.environ.get("R5", "rout5")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research6_factors import fomc_dates, px_at, real_days  # noqa: E402

FILES = {"SPX500": ("usa500idxusd", 1.5, 2.0), "NAS100": ("usatechidxusd", 1.5, 2.0), "GER40": ("deuidxeur", 1.5, 2.0),
         "EURUSD": ("eurusd", 1.1, 0.5), "GBPUSD": ("gbpusd", 1.4, 0.5), "USDJPY": ("usdjpy", 1.2, 0.5),
         "AUDUSD": ("audusd", 1.6, 0.5), "USDCAD": ("usdcad", 1.5, 0.5), "USDCHF": ("usdchf", 1.8, 0.5),
         "XAUUSD": ("xauusd", 1.5, 1.0), "XAGUSD": ("xagusd", 4.0, 1.0), "WTI": ("lightcmdusd", 4.0, 1.0),
         "BTCUSD": ("btcusd", 8.0, 1.0), "USTBOND": ("ustbondtrusd", 2.0, 1.0)}
PER = (("2014-19", "2014-01-01", "2019-12-31"), ("2020-22", "2020-01-01", "2022-12-31"), ("2023-26", "2023-01-01", "2026-12-31"))
NY = "America/New_York"


def m5(mid):
    d = pd.read_parquet(os.path.join(M5, FILES[mid][0] + "_m5.parquet")).astype(float)
    d.index = d.index.tz_convert("UTC")
    return d


def daily(mid):
    """Daily OHLC on the New York 17:00 roll (date = trading day)."""
    df = m5(mid)
    idx = df.index.tz_convert(NY) + pd.Timedelta(hours=7)
    d = df.groupby(idx.normalize().tz_localize(None)).agg(O=("open", "first"), H=("high", "max"), L=("low", "min"),
                                                          C=("close", "last"), n=("close", "size"))
    d = d[(d.n >= 60) & ((d.H - d.L) > 0)]
    if mid not in ("BTCUSD",):
        d = d[d.index.dayofweek < 5]
    pc = d.C.shift()
    d["atr"] = pd.concat([d.H - d.L, (d.H - pc).abs(), (d.L - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
    return d


def yf(name, col="Close"):
    d = pd.read_csv(os.path.join(RD, f"YF_{name}.csv"), index_col=0)
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[d.index.notna()]
    out = d[["Open", "High", "Low", "Close", "Adj Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    out.index = out.index.normalize()
    return out


def summarize(x: pd.Series, name, unit="bp"):
    x = x.dropna()
    n = len(x)
    if n < 15:
        return dict(test=name, n=n, p=1.0)
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n)) if x.std() > 0 else 0.0
    k = 1e4 if unit == "bp" else 1.0
    row = dict(test=name, unit=unit, n=n, hit=round(float((x > 0).mean()), 3), mean=round(float(x.mean() * k), 3),
               t=round(float(t), 2), p=float(st.norm.sf(t)))
    idx = pd.to_datetime(pd.Series(x.index))
    for lab, a, b in PER:
        s = ((idx >= a) & (idx <= b)).values
        row[f"m_{lab}"] = round(float(x[s].mean() * k), 3) if s.sum() >= 5 else None
        row[f"n_{lab}"] = int(s.sum())
    return row


# =============================================================== H1/H2 rebalancing
def rebalancing_signals():
    spx, ief = yf("GSPC")["Close"], yf("IEF")["Adj Close"]
    df = pd.DataFrame({"s": spx.pct_change(), "b": ief.pct_change()}).dropna()
    df = df[df.index >= "2004-01-01"]
    deltas = np.round(np.arange(0.0, 0.0251, 0.001), 4)
    w = np.full(len(deltas), 0.6)
    thr = []
    rs, rb = df.s.values, df.b.values
    for i in range(len(df)):
        w = w * (1 + rs[i]) / (w * (1 + rs[i]) + (1 - w) * (1 + rb[i]))
        thr.append(float(np.mean(w - 0.6)))
        reset = np.abs(w - 0.6) >= deltas
        w = np.where(reset, 0.6, w)
    thr = pd.Series(thr, index=df.index)
    # calendar drift: since the last month-end reset
    cal, wc, prev_m = [], 0.6, None
    for i, d in enumerate(df.index):
        if prev_m is not None and d.month != prev_m:
            wc = 0.6
        wc = wc * (1 + rs[i]) / (wc * (1 + rs[i]) + (1 - wc) * (1 + rb[i]))
        cal.append(wc - 0.6)
        prev_m = d.month
    cal = pd.Series(cal, index=df.index)
    z = (thr - 0) / thr.rolling(252, min_periods=120).std().shift(1)
    return thr, z, cal


def position_pnl(pos: pd.Series, px: pd.Series, cost_bp, night_bp):
    """pos[t] = position held from close t to close t+1. Returns daily P&L series (decimal) on days with a position."""
    r = px.pct_change().shift(-1)
    change = pos.diff().abs().fillna(pos.abs())
    pnl = pos * r - change * cost_bp * 1e-4 / 2 - pos.abs() * night_bp * 1e-4
    # closing cost when the position ends
    return pnl[pos != 0].dropna()


def h1_h2(rows, trades):
    thr, z, cal = rebalancing_signals()
    vix = yf("VIX")["Close"]
    vix_high = vix > vix.rolling(252, min_periods=120).median()
    for mid in ("SPX500", "NAS100"):
        d = daily(mid)
        px = d.C
        zz = z.reindex(px.index, method="ffill")
        c, nb = FILES[mid][1], FILES[mid][2]
        for H in (1, 3):
            sig = pd.Series(np.where(zz > 1, -1, np.where(zz < -1, 1, 0)), index=px.index)
            pos = sig.replace(0, np.nan).ffill(limit=H - 1).fillna(0) if H > 1 else sig
            pnl = position_pnl(pos, px, c, nb)
            rows.append(summarize(pnl, f"H1|{mid}|threshold z>1 hold {H}d"))
            trades[f"H1_{mid}_{H}"] = pnl
        # calendar: hold through the last 4 trading days of the month (close T-4 -> close T), position decided each
        # close from the drift KNOWN at that close; then reverse from close T to close of the first day of next month.
        days = px.index
        per = days.to_period("M")
        rank_from_end = pd.Series(range(len(days)), index=days).groupby(per).rank(ascending=False).astype(int)
        cz = cal.reindex(days, method="ffill")
        pos = pd.Series(0.0, index=days)
        hold = rank_from_end.between(2, 5)            # held from this close into one of the last 4 days
        pos[hold.values] = -np.sign(cz[hold.values])
        last = rank_from_end == 1                     # held from the month's last close into the next month's first day
        pos[last.values] = np.sign(cz[last.values])
        pnl = position_pnl(pos, px, c, nb)
        rows.append(summarize(pnl, f"H2|{mid}|calendar last-4 + first-day reversal"))
        vh = vix_high.reindex(pnl.index, method="ffill").fillna(False).values
        rows.append(summarize(pnl[vh], f"H2|{mid}|calendar, VIX above 1y median"))
        trades[f"H2_{mid}"] = pnl


# =============================================================== H3/H4 gamma
def gex_quintiles():
    d = pd.read_csv(os.path.join(RD, "DIX.csv"))
    g = pd.Series(d["gex"].astype(float).values, index=pd.to_datetime(d["date"]))
    lo = g.rolling(252, min_periods=120).quantile(0.2)
    hi = g.rolling(252, min_periods=120).quantile(0.8)
    return g, lo, hi


def prev_val(s: pd.Series, day):
    s2 = s[s.index < pd.Timestamp(day)]
    return s2.iloc[-1] if len(s2) else np.nan


def h3(rows, trades):
    g, lo, hi = gex_quintiles()
    vix, v3 = yf("VIX")["Close"], yf("VIX3M")["Close"]
    contango = (vix / v3) < 1
    for mid in ("SPX500", "NAS100"):
        df = m5(mid)
        x = df.tz_convert(NY)
        days = real_days(df, NY, "09:30", "16:00")
        t = x.index.strftime("%H:%M")
        ses = x[(t >= "09:30") & (t < "16:00")]
        sg = {d: v for d, v in ses.groupby(ses.index.date)}
        H = pd.Series({d: sg[d].high.max() for d in days}); L = pd.Series({d: sg[d].low.min() for d in days})
        C = pd.Series({d: sg[d].close.iloc[-1] for d in days})
        atr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1).rolling(14).mean().shift(1)
        res = {}
        for d in days:
            ts = pd.Timestamp(d)
            if not (prev_val(g, ts) >= prev_val(hi, ts)) or not bool(prev_val(contango.astype(float), ts) == 1):
                continue
            a = atr.get(d)
            if not a == a:
                continue
            w = sg[d]
            tt = w.index.strftime("%H:%M")
            morn, aft = w[tt < "12:00"], w[(tt >= "12:00") & (tt < "15:50")]
            if len(morn) < 20 or len(aft) < 5:
                continue
            op, p12 = morn.open.iloc[0], morn.close.iloc[-1]
            if abs(p12 - op) <= 0.6 * a:
                continue
            di = -1 if p12 > op else 1
            ext = morn.high.max() if di < 0 else morn.low.min()
            E = aft.open.iloc[0]
            stop = ext - di * 0.5 * a
            risk = di * (E - stop)
            if risk <= 0:
                continue
            tgt = op
            o, h, l, c = aft.open.values, aft.high.values, aft.low.values, aft.close.values
            ex = c[-1]
            for j in range(len(c)):
                if (l[j] <= stop) if di > 0 else (h[j] >= stop):
                    ex = stop; break
                if (h[j] >= tgt) if di > 0 else (l[j] <= tgt):
                    ex = tgt; break
            res[ts] = di * (ex - E) / risk - FILES[mid][1] * 1e-4 * E / risk
        s = pd.Series(res)
        rows.append(summarize(s, f"H3|{mid}|positive-gamma fade (GEX top quintile, contango)", unit="R"))
        trades[f"H3_{mid}"] = s


def h4(rows, trades):
    g, lo, hi = gex_quintiles()
    for name, f in (("NAS100", "tr_NAS100_1330_1.0_half.csv"), ("SPX500", "tr_SPX500_1330_1.0_half.csv")):
        p = os.path.join(R4, f)
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p, parse_dates=["day"])
        flag = np.array([prev_val(g, d) <= prev_val(lo, d) for d in t.day])
        a, b = t.R[flag], t.R[~flag]
        tt = (a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b))
        row = dict(test=f"H4|{name}|big-move k1: GEX bottom quintile vs rest", unit="R diff", n=len(a), hit=round((a > 0).mean(), 3),
                   mean=round(a.mean(), 3), t=round(tt, 2), p=float(st.norm.sf(tt)), rest_mean=round(b.mean(), 3), rest_n=len(b))
        for lab, x0, x1 in PER:
            s = (t.day >= x0) & (t.day <= x1)
            row[f"m_{lab}"] = round(t.R[flag & s].mean(), 3) if (flag & s).sum() >= 5 else None
            row[f"n_{lab}"] = int((flag & s).sum())
        rows.append(row)


# =============================================================== H5 slow trend
def ewmac(c, f, s):
    vol = c.diff().ewm(span=36, min_periods=30).std()
    raw = (c.ewm(span=f, min_periods=f).mean() - c.ewm(span=s, min_periods=s).mean()) / vol
    scal = 10 / raw.abs().expanding(min_periods=250).mean().shift(1)    # past-only forecast scaling
    return (raw * scal).clip(-20, 20)


def h5(rows, trades):
    large, small = ["USTBOND", "XAUUSD", "XAGUSD", "WTI", "BTCUSD"], ["SPX500", "NAS100", "GER40", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
    group = {}
    for mid in large + small:
        d = daily(mid)
        c = d.C
        fc = (ewmac(c, 32, 128) + ewmac(c, 64, 256)) / 2
        ret = c.pct_change()
        vol = ret.ewm(span=36, min_periods=30).std() * math.sqrt(252)
        pos = (fc / 10) * (0.10 / vol)                     # 10% annualised vol target per market at forecast 10
        pos = pos.clip(-3, 3).shift(1)                      # decided at yesterday's close
        turn = pos.diff().abs()
        cost = turn * FILES[mid][1] * 1e-4 / 2
        fut = (pos * ret - cost).dropna()
        cfd = (fut - pos.abs().reindex(fut.index) * 0.03 / 252).dropna()
        fut, cfd = fut[fut.index >= "2015-01-01"], cfd[cfd.index >= "2015-01-01"]
        if len(fut) < 300:
            continue
        kind = "large-tick" if mid in large else "small-tick"
        for lab, x in (("futures-style", fut), ("CFD 3%/yr financing", cfd)):
            sr = x.mean() / x.std() * math.sqrt(252)
            rows.append(dict(test=f"H5|{mid}|slow trend {lab}", unit="Sharpe", n=len(x), mean=round(sr, 2),
                             t=round(sr * math.sqrt(len(x) / 252), 2), p=float(st.norm.sf(sr * math.sqrt(len(x) / 252))) if lab == "futures-style" else 1.0,
                             **{f"m_{a}": round(x[(x.index >= b) & (x.index <= e)].mean() / x[(x.index >= b) & (x.index <= e)].std() * math.sqrt(252), 2)
                                if ((x.index >= b) & (x.index <= e)).sum() > 100 else None for a, b, e in PER}, group=kind))
            group.setdefault((kind, lab), []).append(x.rename(mid))
    for (kind, lab), xs in group.items():
        P = pd.concat(xs, axis=1).fillna(0).mean(axis=1)
        sr = P.mean() / P.std() * math.sqrt(252)
        rows.append(dict(test=f"H5|POOL {kind}|slow trend {lab}", unit="Sharpe", n=len(P), mean=round(sr, 2), t=round(sr * math.sqrt(len(P) / 252), 2),
                         p=1.0, **{f"m_{a}": round(P[(P.index >= b) & (P.index <= e)].mean() / P[(P.index >= b) & (P.index <= e)].std() * math.sqrt(252), 2) for a, b, e in PER}))
        trades[f"H5_{kind}_{lab}"] = P


# =============================================================== H6 low-vol breakouts
def h6(rows, trades):
    A = pd.read_csv(os.path.join(R5, "high_rr_trades.csv"), parse_dates=["day"])
    T = A[(A.setup == "T55") & (A.dir == 1)].copy()
    flags = []
    cache = {}
    for _, r in T.iterrows():
        if r.market not in cache:
            try:
                d = daily(r.market) if r.market in FILES else None
            except Exception:
                d = None
            if d is None:
                cache[r.market] = None
            else:
                rv = d.C.pct_change().rolling(20).std()
                cache[r.market] = (rv, rv.rolling(252, min_periods=120).quantile(0.4))
        c = cache[r.market]
        if c is None:
            flags.append(np.nan); continue
        rv, q = c
        flags.append(float(rv.get(r.day, np.nan) < q.get(r.day, np.nan)) if r.day in rv.index else np.nan)
    T["low"] = flags
    T = T.dropna(subset=["low"])
    a, b = T[T.low == 1].R, T[T.low == 0].R
    tt = (a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b))
    row = dict(test="H6|POOL|55d breakout long: low-vol entry vs rest", unit="R diff", n=len(a), hit=round((a > 0).mean(), 3), mean=round(a.mean(), 3),
               t=round(tt, 2), p=float(st.norm.sf(tt)), rest_mean=round(b.mean(), 3), rest_n=len(b))
    for lab, x0, x1 in PER:
        s = (T.day >= x0) & (T.day <= x1)
        row[f"m_{lab}"] = round(T.R[(T.low == 1) & s].mean(), 3) if ((T.low == 1) & s).sum() >= 5 else None
        row[f"n_{lab}"] = int(((T.low == 1) & s).sum())
    rows.append(row)


# =============================================================== H7/H8 VIX events (Yahoo daily, 2008->)
def swing_trade(df, i0, d, max_days, stop_atr, exit_rule, cost_bp, night_bp):
    E = df.Close.iloc[i0]
    a = df.atr.iloc[i0]
    stop = E - d * stop_atr * a
    for k in range(i0 + 1, min(len(df), i0 + max_days + 1)):
        if (df.Low.iloc[k] <= stop) if d > 0 else (df.High.iloc[k] >= stop):
            px = min(stop, df.Open.iloc[k]) if d > 0 else max(stop, df.Open.iloc[k])
            return d * (px - E) / (stop_atr * a) - (cost_bp + night_bp * (k - i0)) * 1e-4 * E / (stop_atr * a), k
        if exit_rule is not None and exit_rule(k):
            px = df.Close.iloc[k]
            return d * (px - E) / (stop_atr * a) - (cost_bp + night_bp * (k - i0)) * 1e-4 * E / (stop_atr * a), k
    k = min(len(df) - 1, i0 + max_days)
    if k == i0:
        return None, k
    return d * (df.Close.iloc[k] - E) / (stop_atr * a) - (cost_bp + night_bp * (k - i0)) * 1e-4 * E / (stop_atr * a), k


def h7_h8(rows, trades):
    vix, v3 = yf("VIX")["Close"], yf("VIX3M")["Close"]
    ratio = (vix / v3).dropna()
    v80 = vix.rolling(252, min_periods=120).quantile(0.8)
    for mid, name in (("SPX500", "GSPC"), ("NAS100", "NDX")):
        df = yf(name)
        df = df[df.index >= "2008-01-01"].copy()
        pc = df.Close.shift()
        df["atr"] = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
        df["ibs"] = (df.Close - df.Low) / (df.High - df.Low)
        df["down3"] = (df.Close.diff() < 0).rolling(3).sum() == 3
        df["ma5"] = df.Close.rolling(5).mean()
        rr, v8, vx = ratio.reindex(df.index), v80.reindex(df.index), vix.reindex(df.index)
        c, nb = FILES[mid][1], FILES[mid][2]
        # H7
        out, i = {}, 0
        while i < len(df) - 1:
            if vx.iloc[i] > v8.iloc[i] and rr.iloc[i] > 1 and df.down3.iloc[i] and df.ibs.iloc[i] < 0.2 and df.atr.iloc[i] > 0:
                R, k = swing_trade(df, i, 1, 5, 2.5, lambda k: df.Close.iloc[k] > df.ma5.iloc[k], c, nb)
                if R is not None:
                    out[df.index[i]] = R
                i = k + 1
            else:
                i += 1
        s = pd.Series(out)
        rows.append(summarize(s, f"H7|{mid}|high-VIX reversal (IBS<0.2, down 3d)", unit="R"))
        trades[f"H7_{mid}"] = s
        # H8
        above = (rr > 1).astype(int)
        run = above.groupby((above != above.shift()).cumsum()).cumsum()
        cross = (rr < 1) & (run.shift(1) >= 3) & (above.shift(1) == 1)
        for hold in (10, 20):
            out, i = {}, 0
            while i < len(df) - 1:
                if cross.iloc[i] and df.atr.iloc[i] > 0:
                    R, k = swing_trade(df, i, 1, hold, 2.0, None, c, nb)
                    if R is not None:
                        out[df.index[i]] = R
                    i = k + 1
                else:
                    i += 1
            s = pd.Series(out)
            rows.append(summarize(s, f"H8|{mid}|VIX curve normalisation, hold {hold}d", unit="R"))
            trades[f"H8_{mid}_{hold}"] = s


# =============================================================== H9 post-FOMC Treasuries
def h9(rows, trades):
    df = m5("USTBOND")
    p14, p16, p0830 = px_at(df, NY, "14:00"), px_at(df, NY, "16:00", 60), px_at(df.assign(close=df.open), NY, "08:35", 10)
    d = daily("USTBOND")
    fd = [x for x in fomc_dates() if pd.Timestamp(x) >= d.index[0]]
    for hold in (5, 15):
        out = {}
        for f in fd:
            if f not in p14.index or f not in p16.index:
                continue
            di = 1 if p16[f] > p14[f] else -1
            nxt = [x for x in d.index if x.date() > f]
            if len(nxt) < hold + 1:
                continue
            e_day = nxt[0].date()
            E = p0830.get(e_day, np.nan)
            if not E == E:
                continue
            a = d.atr.loc[:pd.Timestamp(f)].iloc[-1]
            stop = E - di * 1.5 * a
            R = None
            for k in range(hold):
                row = d.loc[nxt[k]]
                if (row.L <= stop) if di > 0 else (row.H >= stop):
                    R = -1.0; break
            if R is None:
                R = di * (d.loc[nxt[hold - 1]].C - E) / (1.5 * a)
            out[pd.Timestamp(f)] = R - (FILES["USTBOND"][1] + FILES["USTBOND"][2] * hold) * 1e-4 * E / (1.5 * a)
        s = pd.Series(out)
        rows.append(summarize(s, f"H9|USTBOND|post-FOMC drift hold {hold}d", unit="R"))
        trades[f"H9_{hold}"] = s


# =============================================================== H10 dash for cash
def h10(rows, trades):
    for mid in ("SPX500", "NAS100"):
        d = daily(mid)
        days = d.index
        per = days.to_period("M")
        pos_in_m = pd.Series(range(len(days)), index=days).groupby(per).rank(ascending=False).astype(int)  # 1 = last day
        c, nb = FILES[mid][1], FILES[mid][2]
        tom, pre = {}, {}
        idx = list(days)
        for i, dd in enumerate(idx):
            if pos_in_m.iloc[i] == 2 and i + 4 < len(idx):          # T-1 close -> T+3 close  (4 nights)
                tom[dd] = d.C.iloc[i + 4] / d.C.iloc[i] - 1 - (c + 4 * nb) * 1e-4
            if pos_in_m.iloc[i] == 9 and i + 4 < len(idx):          # T-8 close -> T-4 close, short
                pre[dd] = -(d.C.iloc[i + 4] / d.C.iloc[i] - 1) - (c + 4 * nb) * 1e-4
        rows.append(summarize(pd.Series(tom), f"H10|{mid}|turn-of-month long T-1 -> T+3"))
        rows.append(summarize(pd.Series(pre), f"H10|{mid}|pre-month-end short T-8 -> T-4"))
        trades[f"H10tom_{mid}"], trades[f"H10pre_{mid}"] = pd.Series(tom), pd.Series(pre)


def bh(p, q=0.10):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    r = np.zeros(m, bool); r[o[:k]] = True
    return r


def main(out="rout7"):
    os.makedirs(out, exist_ok=True)
    rows, trades = [], {}
    for f in (h1_h2, h3, h4, h5, h6, h7_h8, h9, h10):
        try:
            f(rows, trades)
            print("done", f.__name__, flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("FAILED", f.__name__, repr(e), flush=True)
    R = pd.DataFrame([r for r in rows if r])
    tested = R.p < 1.0
    R["fdr10"] = False
    R.loc[tested, "fdr10"] = bh(R.loc[tested, "p"].values)
    R.to_csv(os.path.join(out, "results.csv"), index=False)
    for k, v in trades.items():
        v.to_csv(os.path.join(out, "tr_" + re.sub(r"[^A-Za-z0-9_.-]", "_", k) + ".csv"))
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 30)
    cols = [c for c in ["test", "unit", "n", "hit", "mean", "t", "m_2014-19", "m_2020-22", "m_2023-26", "n_2014-19", "n_2020-22", "n_2023-26", "rest_mean", "fdr10"] if c in R.columns]
    print(R[cols].to_string(index=False))
    print("\nFDR-tested:", int(tested.sum()), "| pass:", R[R.fdr10].test.tolist())


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

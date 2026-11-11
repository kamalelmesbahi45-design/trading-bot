"""
Day-trading edge research on 12+ years of Dukascopy M5 bars.

Every setup below was DECLARED BEFORE looking at results, with a small fixed set of
variants. All variants of all setups on all markets are counted in one Benjamini-Hochberg
false-discovery control, then checked out-of-sample (2022 onward, never used to choose anything).

Fill rules (pessimistic):
  * market entries fill at the bar open; stop-entries at the level, or at the open if price gapped past it
  * if a bar touches both stop and target -> STOP; if a bar triggers both OCO entry orders -> -1R
  * every trade pays a round-trip cost (spread + slippage) from COST_BPS
  * every position is flat by the session end (pure day trading)
"""
from __future__ import annotations
import os, sys, json, math, itertools
import numpy as np
import pandas as pd
from scipy import stats as st

DATA = os.environ.get("M5_DIR", "m5")
OOS_START = pd.Timestamp("2022-01-01")

MKT = {
    # id      file             tz                  session (local)     cost bps round trip
    "NAS100": ("usatechidxusd", "America/New_York", ("09:30", "16:00"), 1.5),
    "SPX500": ("usa500idxusd",  "America/New_York", ("09:30", "16:00"), 1.5),
    "GER40":  ("deuidxeur",     "Europe/Berlin",    ("09:00", "17:30"), 1.5),
    "BTCUSD": ("btcusd",        "America/New_York", ("09:30", "16:00"), 8.0),
    "XAUUSD": ("xauusd",        "Europe/London",    ("07:00", "17:00"), 1.5),
    "EURUSD": ("eurusd",        "Europe/London",    ("07:00", "17:00"), 1.1),
    "GBPUSD": ("gbpusd",        "Europe/London",    ("07:00", "17:00"), 1.4),
    "USDJPY": ("usdjpy",        "Europe/London",    ("07:00", "17:00"), 1.2),
}
INDEX_LIKE = ["NAS100", "SPX500", "GER40", "BTCUSD"]
FX_LIKE = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]


# ------------------------------------------------------------------ data
def load(mid: str) -> pd.DataFrame:
    f, tz, _, _ = MKT[mid]
    df = pd.read_parquet(os.path.join(DATA, f"{f}_m5.parquet")).astype("float64")
    df.index = df.index.tz_convert(tz)
    return df


def daily_context(df: pd.DataFrame, mid: str) -> pd.DataFrame:
    """Daily bars on the local calendar (FX/gold: NY 17:00 roll) -> ATR14, SMA50, prior session H/L."""
    _, tz, (s0, s1), _ = MKT[mid]
    if mid in FX_LIKE:
        x = df.copy()
        x.index = x.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)   # 17:00 NY -> midnight
        day = x.index.date
    else:
        day = df.index.date
    d = df.groupby(day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    d.index = pd.to_datetime(d.index)
    d = d[d.index.dayofweek < 5] if mid != "BTCUSD" else d
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["sma50"] = d.close.rolling(50).mean()
    # session high/low (cash session for indices) used as "previous day" levels
    t = df.index.strftime("%H:%M")
    ses = df[(t >= s0) & (t < s1)]
    sd = ses.groupby(ses.index.date).agg(sh=("high", "max"), sl=("low", "min"), sc=("close", "last"))
    sd.index = pd.to_datetime(sd.index)
    d = d.join(sd, how="left")
    # everything known BEFORE the day starts
    out = pd.DataFrame({
        "atr": d.atr.shift(), "trend": np.sign(d.close.shift() - d.sma50.shift()),
        "pdh": (d.sh if mid in INDEX_LIKE and mid != "BTCUSD" else d.high).shift(),
        "pdl": (d.sl if mid in INDEX_LIKE and mid != "BTCUSD" else d.low).shift(),
        "pdc": (d.sc if mid in INDEX_LIKE and mid != "BTCUSD" else d.close).shift(),
    })
    return out


def sessions(df: pd.DataFrame, s0: str, s1: str):
    """Yield (date, frame) for the local window [s0, s1)."""
    t = df.index.strftime("%H:%M")
    w = df[(t >= s0) & (t < s1)]
    for d, g in w.groupby(w.index.date):
        if len(g) >= 6:
            yield pd.Timestamp(d), g


# ------------------------------------------------------------------ trade engine
def run_trade(o, h, l, c, k, entry, d, stop, target):
    """Walk bars k.. ; return (exit_price, outcome, bars_held). Flat at the last bar's close."""
    n = len(c)
    for j in range(k, n):
        if j > k:   # gaps at bar open
            if d * (o[j] - stop) <= 0:
                return o[j], "stop", j - k
            if target is not None and d * (o[j] - target) >= 0:
                return o[j], "target", j - k
        hit_stop = (l[j] <= stop) if d > 0 else (h[j] >= stop)
        hit_tgt = target is not None and ((h[j] >= target) if d > 0 else (l[j] <= target))
        if hit_stop:
            return stop, "stop", j - k
        if hit_tgt:
            return target, "target", j - k
    return c[-1], "eod", n - 1 - k


def R_of(entry, exit_px, d, risk, cost_bps):
    return d * (exit_px - entry) / risk - cost_bps * 1e-4 * entry / risk


def stop_entry(o, h, l, start, end, up, dn):
    """First bar in [start,end) touching the buy-stop `up` or sell-stop `dn`. Returns (k, d, fill) or None.
    Both touched in one bar -> ('both')."""
    for j in range(start, end):
        bu = up is not None and h[j] >= up
        bd = dn is not None and l[j] <= dn
        if bu and bd:
            return j, 0, None
        if bu:
            return j, 1, max(up, o[j])
        if bd:
            return j, -1, min(dn, o[j])
    return None


# ------------------------------------------------------------------ setups
def setup_orb5(mid, df, ctx, target_mode, trend_filter):
    """Zarattini & Aziz (2023) QQQ ORB: direction of the first 5-min candle, stop at its low/high."""
    _, _, (s0, s1), cost = MKT[mid]
    out = []
    for day, g in sessions(df, s0, s1):
        if day not in ctx.index:
            continue
        cx = ctx.loc[day]
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        if c[0] == o[0]:
            continue
        d = 1 if c[0] > o[0] else -1
        if trend_filter and cx.trend * d <= 0:
            continue
        entry = o[1]
        stop = l[0] if d > 0 else h[0]
        risk = d * (entry - stop)
        if risk <= 0:
            continue
        tgt = entry + d * (10 if target_mode == "EOD" else 2) * risk
        ex, oc, b = run_trade(o, h, l, c, 1, entry, d, stop, tgt)
        out.append((day, d, R_of(entry, ex, d, risk, cost), oc, b))
    return out


def setup_orb_range(mid, df, ctx, W, target_mode, trend_filter):
    """Classic opening-range breakout with OCO stop orders at the range; stop = other side of the range."""
    _, _, (s0, s1), cost = MKT[mid]
    nb = W // 5
    out = []
    for day, g in sessions(df, s0, s1):
        if day not in ctx.index or len(g) < nb + 14:
            continue
        cx = ctx.loc[day]
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        hi, lo = h[:nb].max(), l[:nb].min()
        if hi <= lo:
            continue
        up = hi if (not trend_filter or cx.trend > 0) else None
        dn = lo if (not trend_filter or cx.trend < 0) else None
        last_entry = len(g) - 12                       # no new entries in the last hour
        e = stop_entry(o, h, l, nb, last_entry, up, dn)
        if e is None:
            continue
        k, d, fill = e
        if d == 0:
            out.append((day, 0, -1.0 - cost * 1e-4 * hi / (hi - lo), "whipsaw", 0))
            continue
        stop = lo if d > 0 else hi
        risk = d * (fill - stop)
        if risk <= 0:
            continue
        tgt = fill + d * 2 * risk if target_mode == "2R" else None
        ex, oc, b = run_trade(o, h, l, c, k, fill, d, stop, tgt)
        out.append((day, d, R_of(fill, ex, d, risk, cost), oc, b))
    return out


def setup_noise(mid, df, ctx):
    """Zarattini, Aziz & Barbon (2024) intraday momentum: trade breaks of the time-of-day 'noise area',
    checked every 30 min, trailing exit at the band, flat at the close. R unit = band half-width."""
    _, _, (s0, s1), cost = MKT[mid]
    days = list(sessions(df, s0, s1))
    # move-from-open profile per bar index, 14-day rolling mean (strictly past days)
    prof = {}
    hist = []
    out = []
    for day, g in days:
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        op = o[0]
        mv = np.abs(c / op - 1)
        if len(hist) >= 14 and day in ctx.index and not math.isnan(ctx.loc[day].pdc):
            L = min(len(mv), min(len(x) for x in hist[-14:]))
            sig = np.mean([x[:L] for x in hist[-14:]], axis=0)
            pc = ctx.loc[day].pdc
            ub = max(op, pc) * (1 + sig)
            lb = min(op, pc) * (1 - sig)
            pos, entry, risk, d = 0, 0.0, 0.0, 0
            for j in range(5, L, 6):                       # bars closing on :00 / :30
                if pos != 0:
                    if (pos > 0 and c[j] < ub[j]) or (pos < 0 and c[j] > lb[j]):
                        out.append((day, pos, R_of(entry, c[j], pos, risk, cost), "trail", j))
                        pos = 0
                if pos == 0 and j < L - 6:
                    if c[j] > ub[j]:
                        pos, entry, risk = 1, c[j], max(ub[j] - op, op * 1e-4)
                    elif c[j] < lb[j]:
                        pos, entry, risk = -1, c[j], max(op - lb[j], op * 1e-4)
            if pos != 0:
                out.append((day, pos, R_of(entry, c[L - 1], pos, risk, cost), "eod", L))
        hist.append(mv)
    return out


def setup_london(mid, df, ctx, stop_mode, target_mode, trend_filter):
    """Asian range (00:00-07:00 London) breakout with OCO stop orders valid 07:00-12:00, flat 17:00."""
    _, _, _, cost = MKT[mid]
    t = df.index.strftime("%H:%M")
    out = []
    asia = df[t < "07:00"]
    ldn = df[(t >= "07:00") & (t < "17:00")]
    ag = {pd.Timestamp(d): g for d, g in asia.groupby(asia.index.date)}
    for d0, g in ldn.groupby(ldn.index.date):
        day = pd.Timestamp(d0)
        if day.dayofweek >= 5 or day not in ag or day not in ctx.index or len(ag[day]) < 60 or len(g) < 60:
            continue
        a = ag[day]
        hi, lo = a.high.max(), a.low.min()
        if hi <= lo:
            continue
        cx = ctx.loc[day]
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        up = hi if (not trend_filter or cx.trend > 0) else None
        dn = lo if (not trend_filter or cx.trend < 0) else None
        e = stop_entry(o, h, l, 0, 60, up, dn)             # 07:00-12:00
        if e is None:
            continue
        k, d, fill = e
        if d == 0:
            out.append((day, 0, -1.0, "whipsaw", 0))
            continue
        stop = (lo if d > 0 else hi) if stop_mode == "opp" else (hi + lo) / 2
        risk = d * (fill - stop)
        if risk <= 0:
            continue
        tgt = fill + d * 2 * risk if target_mode == "2R" else None
        ex, oc, b = run_trade(o, h, l, c, k, fill, d, stop, tgt)
        out.append((day, d, R_of(fill, ex, d, risk, cost), oc, b))
    return out


def setup_sweep(mid, df, ctx, k_R, trend_filter):
    """Liquidity sweep: price trades beyond yesterday's high/low, then a 15-min close back inside ->
    fade it. Stop beyond the sweep extreme, target k_R, flat at session end."""
    _, _, (s0, s1), cost = MKT[mid]
    if mid == "BTCUSD":
        s0, s1 = "03:00", "16:00"
    out = []
    for day, g in sessions(df, s0, s1):
        if day not in ctx.index:
            continue
        cx = ctx.loc[day]
        if any(math.isnan(v) for v in (cx.pdh, cx.pdl, cx.atr)):
            continue
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        n = len(c)
        swept_hi = swept_lo = False
        ext_hi, ext_lo = -np.inf, np.inf
        for j in range(n - 3):
            ext_hi, ext_lo = max(ext_hi, h[j]), min(ext_lo, l[j])
            swept_hi |= h[j] > cx.pdh
            swept_lo |= l[j] < cx.pdl
            if (j + 1) % 3 != 0:                           # act only on 15-minute closes
                continue
            d = 0
            if swept_hi and c[j] < cx.pdh and (not trend_filter or cx.trend < 0):
                d, stop = -1, ext_hi + 0.05 * cx.atr
            elif swept_lo and c[j] > cx.pdl and (not trend_filter or cx.trend > 0):
                d, stop = 1, ext_lo - 0.05 * cx.atr
            if d == 0:
                continue
            entry = o[j + 1]
            risk = d * (entry - stop)
            if not (0.05 * cx.atr <= risk <= 0.6 * cx.atr):
                break                                       # sanity: skip silly stops, one decision per day
            tgt = entry + d * k_R * risk
            ex, oc, b = run_trade(o, h, l, c, j + 1, entry, d, stop, tgt)
            out.append((day, d, R_of(entry, ex, d, risk, cost), oc, b))
            break
    return out


TESTS = []
for m in INDEX_LIKE:
    for tm, tf in itertools.product(["EOD", "2R"], [False, True]):
        TESTS.append(("ORB5", m, dict(target_mode=tm, trend_filter=tf)))
    for W, tm, tf in itertools.product([15, 30], ["2R", "EOD"], [False, True]):
        TESTS.append(("ORB_RANGE", m, dict(W=W, target_mode=tm, trend_filter=tf)))
    TESTS.append(("NOISE", m, {}))
for m in FX_LIKE:
    for sm, tm, tf in itertools.product(["opp", "mid"], ["2R", "EOD"], [False, True]):
        TESTS.append(("LONDON", m, dict(stop_mode=sm, target_mode=tm, trend_filter=tf)))
for m in INDEX_LIKE + FX_LIKE:
    for kR, tf in itertools.product([2, 3], [False, True]):
        TESTS.append(("SWEEP", m, dict(k_R=kR, trend_filter=tf)))
FN = {"ORB5": setup_orb5, "ORB_RANGE": setup_orb_range, "NOISE": setup_noise, "LONDON": setup_london, "SWEEP": setup_sweep}


# ------------------------------------------------------------------ statistics
def stats_of(tr: pd.DataFrame) -> dict:
    R = tr.R.values
    n = len(R)
    if n < 30:
        return dict(n=n, valid=False)
    m, sd = R.mean(), R.std(ddof=1)
    t = m / (sd / math.sqrt(n)) if sd > 0 else 0
    p = 1 - st.t.cdf(t, n - 1)
    oos = tr[tr.day >= OOS_START].R
    ins = tr[tr.day < OOS_START].R
    years = tr.groupby(tr.day.dt.year).R.sum()
    eq = np.cumsum(R)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    losing = max((len(list(g)) for k, g in itertools.groupby(R <= 0) if k), default=0)
    wins = R[R > 0]
    return dict(valid=True, n=n, per_year=round(n / max(1, (tr.day.iloc[-1] - tr.day.iloc[0]).days / 365.25), 1),
                hit=round(float((R > 0).mean()), 3), meanR=round(float(m), 3), t=round(float(t), 2), p=float(p),
                is_n=int(len(ins)), is_meanR=round(float(ins.mean()), 3) if len(ins) else None,
                oos_n=int(len(oos)), oos_meanR=round(float(oos.mean()), 3) if len(oos) else None,
                oos_t=round(float(oos.mean() / (oos.std(ddof=1) / math.sqrt(len(oos)))), 2) if len(oos) > 2 and oos.std() > 0 else None,
                pos_years=f"{int((years > 0).sum())}/{len(years)}", totalR=round(float(R.sum()), 1), maxddR=round(dd, 1),
                max_losing_streak=int(losing), avg_win=round(float(wins.mean()), 2) if len(wins) else 0,
                pf=round(float(wins.sum() / -R[R <= 0].sum()), 2) if (R <= 0).any() else None)


def bh(p, q=0.10):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    r = np.zeros(m, bool)
    r[o[:k]] = True
    return r


def main(out_dir="research_out"):
    os.makedirs(out_dir, exist_ok=True)
    cache = {}
    rows, trades_all = [], {}
    for name, m, kw in TESTS:
        if m not in cache:
            try:
                df = load(m)
            except Exception as e:
                print("missing data", m, e)
                cache[m] = None
                continue
            cache[m] = (df, daily_context(df, m))
            print("loaded", m, len(df), df.index[0], df.index[-1], flush=True)
        if cache[m] is None:
            continue
        df, ctx = cache[m]
        tr = pd.DataFrame(FN[name](m, df, ctx, **kw), columns=["day", "dir", "R", "outcome", "bars"])
        key = f"{name}|{m}|" + ",".join(f"{k}={v}" for k, v in kw.items())
        s = stats_of(tr) if len(tr) else dict(n=0, valid=False)
        rows.append(dict(key=key, setup=name, market=m, **kw, **s))
        trades_all[key] = tr
        print(f"{key:60s} n={s.get('n')} meanR={s.get('meanR')} p={s.get('p', 1):.4f} oos={s.get('oos_meanR')}", flush=True)
    res = pd.DataFrame(rows)
    val = res.valid.fillna(False)
    res["fdr"] = False
    res.loc[val, "fdr"] = bh(res.loc[val, "p"].values)
    res["grade"] = np.where(res.fdr & (res.oos_meanR > 0) & (res.oos_t > 1.0) & (res.n >= 100), "A",
                   np.where((res.p < 0.05) & (res.oos_meanR > 0), "B", np.where(res.meanR > 0, "C", "X")))
    res.sort_values("p").to_csv(os.path.join(out_dir, "results.csv"), index=False)
    # equity curves (daily R) for every A/B test
    curves = {}
    for _, r in res[res.grade.isin(["A", "B"])].iterrows():
        tr = trades_all[r.key]
        cr = tr.groupby(tr.day.dt.to_period("M")).R.sum().cumsum()
        curves[r.key] = [[str(k), round(float(v), 2)] for k, v in cr.items()]
        tr.to_csv(os.path.join(out_dir, "trades_" + r.key.replace("|", "__").replace(",", "_").replace("=", "-") + ".csv"), index=False)
    json.dump(curves, open(os.path.join(out_dir, "curves.json"), "w"))
    print(res.grade.value_counts().to_dict(), "tests:", len(res))


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

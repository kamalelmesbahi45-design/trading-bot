"""
Live day-trading engine. Six systems that came out of the research in daytrade/ (12 years of
Dukascopy 5-minute data, net of costs). Each run rebuilds TODAY's plan for every system from the
latest 5-minute bars: the levels to place, whether the trade triggered, where it is now, and the
result once the session is over. Rules are byte-for-byte the research rules (see tests/test_intraday_replay.py).

Data live: Yahoo Finance via yfinance (5m for 60 days, 1h for 2 years, 1d). Replay/test: Dukascopy parquet.
"""
from __future__ import annotations
import datetime as dt
from zoneinfo import ZoneInfo
import math
import numpy as np
import pandas as pd

UTC = dt.timezone.utc
NY, LDN, FRA = ZoneInfo("America/New_York"), ZoneInfo("Europe/London"), ZoneInfo("Europe/Berlin")
B5 = pd.Timedelta(minutes=5)

# Backtest statistics (daytrade/results, 2014 -> Nov 2026, after costs). R = profit / initial risk.
SYSTEMS = [
    dict(id="BIG_NAS", name="Nasdaq big-move continuation", market="Nasdaq 100", yf="^NDX", kind="bigmove", tz=NY,
         T="13:30", close="16:00", k=2.0, close_from_daily=True, cost=1.5, evidence="Candidate",
         stats=dict(n=231, per_year=18.3, hit=0.619, meanR=0.13, avg_win=0.48, avg_loss=-0.44, pos_years="8/13", maxdd=-6.2, max_losing_streak=7, oos=0.026),
         why="On days the index has already moved > 2 sigma by 13:30 New York, option-dealer gamma hedging and leveraged-ETF "
             "rebalancing push the same way into the close (Baltussen, Da, Lammers & Martens 2021; Gao et al. 2018).",
         rules="At 13:30 NY: if price is beyond the trigger level, enter in that direction at once. Stop = half of the day's move "
               "given back. Exit at 16:00 NY. No trade if neither trigger is reached."),
    dict(id="BIG_SPX", name="S&P 500 big-move continuation", market="S&P 500", yf="^GSPC", kind="bigmove", tz=NY,
         T="13:30", close="16:00", k=2.0, close_from_daily=True, cost=1.5, evidence="Candidate",
         stats=dict(n=251, per_year=19.8, hit=0.534, meanR=0.112, avg_win=0.59, avg_loss=-0.44, pos_years="9/13", maxdd=-7.6, max_losing_streak=7, oos=0.034),
         why="Same mechanism as the Nasdaq version: hedging flows late in the session continue large moves.",
         rules="At 13:30 NY: if price is beyond the trigger level, enter in that direction. Stop = half the day's move given back. Exit 16:00 NY."),
    dict(id="BIG_XAU", name="Gold big-move continuation", market="Gold", yf="GC=F", kind="bigmove", tz=NY,
         T="10:00", close="16:00", k=2.0, close_from_daily=False, cost=1.5, evidence="Candidate",
         stats=dict(n=193, per_year=15.2, hit=0.549, meanR=0.114, avg_win=0.63, avg_loss=-0.51, pos_years="9/13", maxdd=-5.1, max_losing_streak=4, oos=0.11),
         why="A > 2 sigma move by 10:00 New York (after London and the US data releases) tends to extend through the NY session.",
         rules="At 10:00 NY: if gold is beyond the trigger level, enter in that direction. Stop = half the move since yesterday's 16:00 "
               "given back. Exit 16:00 NY."),
    dict(id="ORB_DAX", name="DAX 30-min opening range breakout (with trend)", market="DAX 40", yf="^GDAXI", kind="orb", tz=FRA,
         s0="09:00", s1="17:30", W=30, last_entry="16:30", evidence="B",
         stats=dict(n=2337, per_year=184.6, hit=0.414, meanR=0.065, avg_win=1.43, avg_loss=-0.90, pos_years="10/13", maxdd=-36.3, max_losing_streak=11, oos=0.072),
         why="The first 30 minutes of Xetra set the day's battle lines; a break in the direction of the daily trend tends to run "
             "(opening-range breakout, Crabel 1990; Zarattini & Aziz 2023).",
         rules="After 09:30 Frankfurt: one stop order at the 30-min range high (if daily trend up) or low (if down). Stop = other side of "
               "the range. Exit 17:30. Orders expire 16:30."),
    dict(id="NOISE_NAS", name="Nasdaq noise-band momentum", market="Nasdaq 100", yf="^NDX", kind="noise", tz=NY,
         s0="09:30", s1="16:00", evidence="B",
         stats=dict(n=2544, per_year=198.8, hit=0.394, meanR=0.062, avg_win=1.20, avg_loss=-0.68, pos_years="9/13", maxdd=-30.8, max_losing_streak=15, oos=0.084),
         why="Moves that leave the normal intraday 'noise area' (time-of-day average move of the last 14 days) tend to continue "
             "(Zarattini, Aziz & Barbon 2024, 'Beat the Market').",
         rules="Every :00 and :30 NY from 10:00 to 15:30: if the half-hour closes above the upper band -> long; below the lower band "
               "-> short. Exit when a half-hour closes back inside the band, or at 16:00. R = band distance from the open."),
    dict(id="LDN_JPY", name="London breakout of the Asian range", market="USD/JPY", yf="JPY=X", kind="london", tz=LDN,
         evidence="B",
         stats=dict(n=2677, per_year=208.3, hit=0.473, meanR=0.047, avg_win=0.93, avg_loss=-0.74, pos_years="11/13", maxdd=-22.3, max_losing_streak=12, oos=0.069),
         why="Liquidity arriving at the London open breaks the quiet Asian range; the first break tends to carry through the day.",
         rules="At 07:00 London: buy stop at the 00:00-07:00 high and sell stop at the low (one cancels the other), valid until 12:00. "
               "Stop = other side of the range. Exit 17:00 London."),
]


# ------------------------------------------------------------------ small helpers
def _utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df[["open", "high", "low", "close"]].astype(float).dropna()


def _at(day: dt.date, hhmm: str, tz) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return pd.Timestamp(dt.datetime.combine(day, dt.time(h, m), tzinfo=tz))


def _iso(t):
    return None if t is None else pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(x, nd=6):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def run_trade(o, h, l, c, k, d, stop, target):
    for j in range(k, len(c)):
        if j > k:
            if d * (o[j] - stop) <= 0:
                return o[j], "stop", j
            if target is not None and d * (o[j] - target) >= 0:
                return o[j], "target", j
        if (l[j] <= stop) if d > 0 else (h[j] >= stop):
            return stop, "stop", j
        if target is not None and ((h[j] >= target) if d > 0 else (l[j] <= target)):
            return target, "target", j
    return c[-1], "eod", len(c) - 1


def stop_entry(o, h, l, start, end, up, dn):
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


def _done(bars, now):
    """Only completed 5-minute bars."""
    return bars[bars.index + B5 <= now]


def _base(S, day, nd):
    return dict(id=S["id"], name=S["name"], market=S["market"], kind=S["kind"], evidence=S["evidence"], stats=S["stats"],
                why=S["why"], rules=S["rules"], day=str(day), nd=nd, levels=[], trade=None, times={}, chart=[])


def _nd(px):
    return 2 if px >= 100 else (3 if px >= 50 else 5)


def _chart(bars, tz):
    return [dict(t=int(i.timestamp()), o=_r(r.open), h=_r(r.high), l=_r(r.low), c=_r(r.close)) for i, r in bars.iterrows()]


def _trade_state(d, fill, stop, target, o, h, l, c, k, idx, session_over, cost_bps=0.0):
    ex, oc, j = run_trade(o, h, l, c, k, d, stop, target)
    risk = abs(fill - stop)
    if oc == "eod" and not session_over:
        return dict(dir="LONG" if d > 0 else "SHORT", entry=_r(fill), stop=_r(stop), target=_r(target), entry_time=_iso(idx[k]),
                    state="OPEN", last=_r(c[-1]), R_now=_r(d * (c[-1] - fill) / risk, 2))
    R = d * (ex - fill) / risk - cost_bps * 1e-4 * fill / risk
    return dict(dir="LONG" if d > 0 else "SHORT", entry=_r(fill), stop=_r(stop), target=_r(target), entry_time=_iso(idx[k]),
                state="CLOSED", exit=_r(ex), exit_time=_iso(idx[j] + B5), outcome=oc, R=_r(R, 3))


# ------------------------------------------------------------------ systems
def sys_london(S, m5, now, **_):
    tz = S["tz"]
    day = now.astimezone(tz).date()
    x = m5.tz_convert(tz)
    t7, t12, t17 = _at(day, "07:00", tz), _at(day, "12:00", tz), _at(day, "17:00", tz)
    today = _done(x[x.index.date == day], now)
    out = _base(S, day, 3)
    out["times"] = dict(orders=_iso(t7), expire=_iso(t12), exit=_iso(t17))
    asia = today[today.index < t7]
    out["chart"] = _chart(today, tz)
    if day.weekday() >= 5:
        return dict(out, status="CLOSED", status_text="Weekend: no session.")
    if now < t7:
        if len(asia):
            out["levels"] = [("Asian high so far", _r(asia.high.max())), ("Asian low so far", _r(asia.low.min()))]
        return dict(out, status="WAITING", status_text="Asian range still forming. Orders go in at 07:00 London.")
    if len(asia) < 60:
        return dict(out, status="NO DATA", status_text="Not enough Asian-session bars today.")
    hi, lo = asia.high.max(), asia.low.min()
    out["levels"] = [("Buy stop (range high)", _r(hi)), ("Sell stop (range low)", _r(lo)), ("Range / risk", _r(hi - lo))]
    out["plan"] = dict(long=dict(entry=_r(hi), stop=_r(lo)), short=dict(entry=_r(lo), stop=_r(hi)), risk=_r(hi - lo))
    win = today[(today.index >= t7) & (today.index < t17)]
    o, h, l, c = win.open.values, win.high.values, win.low.values, win.close.values
    n_entry = int((win.index < t12).sum())
    e = stop_entry(o, h, l, 0, n_entry, hi, lo)
    over = now >= t17
    if e is None:
        if now >= t12:
            return dict(out, status="NO TRADE", status_text="Range not broken before 12:00: no trade today.")
        return dict(out, status="ORDERS", status_text="Both stop orders working until 12:00 London.")
    k, d, fill = e
    if d == 0:
        out["trade"] = dict(state="CLOSED", outcome="whipsaw", R=-1.0, entry_time=_iso(win.index[k]))
        return dict(out, status="DONE", status_text="Both sides hit in one bar: counted as -1R.")
    stop = lo if d > 0 else hi
    out["trade"] = _trade_state(d, fill, stop, None, o, h, l, c, k, win.index, over, 1.2)
    st = "IN TRADE" if out["trade"]["state"] == "OPEN" else "DONE"
    return dict(out, status=st, status_text=f"{out['trade']['dir']} triggered at {fill:.3f}.")


def sys_orb(S, m5, now, d1=None, **_):
    tz = S["tz"]
    day = now.astimezone(tz).date()
    x = m5.tz_convert(tz)
    t0, tor, tle, t1 = _at(day, S["s0"], tz), _at(day, S["s0"], tz) + pd.Timedelta(minutes=S["W"]), _at(day, S["last_entry"], tz), _at(day, S["s1"], tz)
    out = _base(S, day, 1)
    out["times"] = dict(range_end=_iso(tor), expire=_iso(tle), exit=_iso(t1))
    if day.weekday() >= 5:
        return dict(out, status="CLOSED", status_text="Weekend: no session.")
    # daily trend known before today: last close vs 50-day average
    if d1 is None:
        daily = x.close.groupby(x.index.date).last()
        daily = daily[[d.weekday() < 5 for d in daily.index]]
    else:
        daily = d1.tz_convert(tz).close
        daily.index = daily.index.date
    daily = daily[daily.index < day]
    if len(daily) < 50:
        return dict(out, status="NO DATA", status_text="Not enough daily history for the trend filter.")
    trend = 1 if daily.iloc[-1] > daily.iloc[-50:].mean() else -1
    out["trend"] = "UP" if trend > 0 else "DOWN"
    ses = _done(x[(x.index >= t0) & (x.index < t1)], now)
    out["chart"] = _chart(ses, tz)
    if now < tor:
        return dict(out, status="WAITING", status_text=f"Daily trend {out['trend']}: only the {'buy' if trend > 0 else 'sell'} side "
                                                         f"will be used. Range is set at {S['s0']}+{S['W']}m Frankfurt.")
    rng = ses[ses.index < tor]
    if len(rng) < S["W"] // 5:
        return dict(out, status="NO DATA", status_text="No opening-range bars today (holiday?).")
    hi, lo = rng.high.max(), rng.low.min()
    lvl = hi if trend > 0 else lo
    out["levels"] = [("Opening-range high", _r(hi)), ("Opening-range low", _r(lo)),
                     (("Buy stop" if trend > 0 else "Sell stop"), _r(lvl)), ("Risk (range)", _r(hi - lo))]
    out["plan"] = (dict(long=dict(entry=_r(hi), stop=_r(lo))) if trend > 0 else dict(short=dict(entry=_r(lo), stop=_r(hi)))) | dict(risk=_r(hi - lo))
    win = ses[ses.index >= tor]
    o, h, l, c = win.open.values, win.high.values, win.low.values, win.close.values
    e = stop_entry(o, h, l, 0, int((win.index < tle).sum()), hi if trend > 0 else None, lo if trend < 0 else None)
    over = now >= t1
    if e is None:
        if now >= tle:
            return dict(out, status="NO TRADE", status_text="Range not broken on the trend side before 16:30: no trade.")
        return dict(out, status="ORDERS", status_text=f"{'Buy' if trend > 0 else 'Sell'} stop working until 16:30 Frankfurt.")
    k, d, fill = e
    stop = lo if d > 0 else hi
    out["trade"] = _trade_state(d, fill, stop, None, o, h, l, c, k, win.index, over, 1.5)
    st = "IN TRADE" if out["trade"]["state"] == "OPEN" else "DONE"
    return dict(out, status=st, status_text=f"{out['trade']['dir']} triggered at {fill:.1f}.")


def _sessions(x, s0, s1):
    t = x.index.strftime("%H:%M")
    w = x[(t >= s0) & (t < s1)]
    # real sessions only: weekdays with actual price movement (data vendors pad holidays/weekends with flat bars)
    return [(d, g) for d, g in w.groupby(w.index.date)
            if len(g) >= 6 and d.weekday() < 5 and (g.high.max() - g.low.min()) > 0 and ((g.high - g.low) > 0).mean() > 0.5]


def sys_noise(S, m5, now, **_):
    tz = S["tz"]
    day = now.astimezone(tz).date()
    x = m5.tz_convert(tz)
    t0, t1 = _at(day, S["s0"], tz), _at(day, S["s1"], tz)
    out = _base(S, day, 1)
    out["times"] = dict(open=_iso(t0), first_check=_iso(t0 + pd.Timedelta(minutes=30)), exit=_iso(t1))
    if day.weekday() >= 5:
        return dict(out, status="CLOSED", status_text="Weekend: no session.")
    past = [(d, g) for d, g in _sessions(x, S["s0"], S["s1"]) if d < day]
    if len(past) < 15:
        return dict(out, status="NO DATA", status_text="Not enough past sessions for the noise band.")
    hist = [np.abs(g.close.values / g.open.values[0] - 1) for _, g in past[-14:]]
    L = min(len(v) for v in hist)
    sig = np.mean([v[:L] for v in hist], axis=0)
    pc = past[-1][1].close.values[-1]
    ses = _done(x[(x.index >= t0) & (x.index < t1)], now)
    out["chart"] = _chart(ses, tz)
    checks = [(j, t0 + (j + 1) * B5) for j in range(5, L, 6) if j < L - 6]
    if now < t0 or len(ses) == 0:
        out["levels"] = [("Previous close", _r(pc))] + [(f"Band width {(t).tz_convert(tz).strftime('%H:%M')}", f"±{sig[j] * 100:.2f}%") for j, t in checks[::3]]
        return dict(out, status="WAITING", status_text="Bands are fixed at the 09:30 New York open.")
    o, h, l, c = ses.open.values, ses.high.values, ses.low.values, ses.close.values
    op = o[0]
    ub, lb = max(op, pc) * (1 + sig), min(op, pc) * (1 - sig)
    out["bands"] = [dict(t=_iso(t), up=_r(ub[j]), dn=_r(lb[j])) for j, t in checks]
    out["levels"] = [("Open", _r(op)), ("Previous close", _r(pc)),
                     ("Upper band 10:00", _r(ub[5])), ("Lower band 10:00", _r(lb[5])),
                     ("Upper band 15:30", _r(ub[checks[-1][0]])), ("Lower band 15:30", _r(lb[checks[-1][0]]))]
    pos, entry, risk, ent_j, trades = 0, 0.0, 0.0, None, []
    n = len(c)
    for j in range(5, min(L, n), 6):
        if pos != 0 and ((pos > 0 and c[j] < ub[j]) or (pos < 0 and c[j] > lb[j])):
            trades.append(dict(dir="LONG" if pos > 0 else "SHORT", entry=_r(entry), entry_time=_iso(ses.index[ent_j] + B5), state="CLOSED",
                               exit=_r(c[j]), exit_time=_iso(ses.index[j] + B5), outcome="back in band",
                               R=_r(pos * (c[j] - entry) / risk - 1.5e-4 * entry / risk, 3)))
            pos = 0
        if pos == 0 and j < L - 6:
            if c[j] > ub[j]:
                pos, entry, risk, ent_j = 1, c[j], max(ub[j] - op, op * 1e-4), j
            elif c[j] < lb[j]:
                pos, entry, risk, ent_j = -1, c[j], max(op - lb[j], op * 1e-4), j
    over = now >= t1
    if pos != 0:
        if over and n >= L:
            trades.append(dict(dir="LONG" if pos > 0 else "SHORT", entry=_r(entry), entry_time=_iso(ses.index[ent_j] + B5), state="CLOSED",
                               exit=_r(c[L - 1]), exit_time=_iso(t1), outcome="close", R=_r(pos * (c[L - 1] - entry) / risk - 1.5e-4 * entry / risk, 3)))
        else:
            band = ub if pos > 0 else lb
            trades.append(dict(dir="LONG" if pos > 0 else "SHORT", entry=_r(entry), entry_time=_iso(ses.index[ent_j] + B5), state="OPEN",
                               stop=_r(band[min(n - 1, L - 1)]), last=_r(c[-1]), R_now=_r(pos * (c[-1] - entry) / risk, 2)))
    out["trades"] = trades
    out["trade"] = trades[-1] if trades else None
    if trades and trades[-1]["state"] == "OPEN":
        return dict(out, status="IN TRADE", status_text=f"{trades[-1]['dir']} since {pd.Timestamp(trades[-1]['entry_time']).tz_convert(tz):%H:%M} NY. "
                                                       f"Exit on a half-hour close back inside the band.")
    if over:
        return dict(out, status="DONE" if trades else "NO TRADE", status_text=(f"{len(trades)} trade(s) today." if trades else "Price stayed inside the band all day."))
    return dict(out, status="WATCHING", status_text="Check every :00 / :30: a half-hour close outside a band is the entry.")


def _price_at(df, bar, tz, hhmm, window_min):
    """Per local date: last close of bars whose END time is in (T - window, T]."""
    if df is None or not len(df):
        return pd.Series(dtype=float)
    x = df.tz_convert(tz)
    end = x.index + bar
    m = end.hour * 60 + end.minute
    h, mm = map(int, hhmm.split(":"))
    T = h * 60 + mm
    sel = (m <= T) & (m > T - window_min) & (end.date == x.index.date)
    s = x.close[sel]
    return s.groupby(end[sel].date).last()


def sys_bigmove(S, m5, now, h1=None, d1=None, **_):
    tz = S["tz"]
    day = now.astimezone(tz).date()
    out = _base(S, day, 2)
    T, tc = _at(day, S["T"], tz), _at(day, S["close"], tz)
    out["times"] = dict(decision=_iso(T), exit=_iso(tc))
    if day.weekday() >= 5:
        return dict(out, status="CLOSED", status_text="Weekend: no session.")
    pT = _price_at(m5, B5, tz, S["T"], 30).combine_first(_price_at(h1, pd.Timedelta(hours=1), tz, S["T"], 30))
    if S["close_from_daily"] and d1 is not None:
        c16 = d1.tz_convert(tz).close
        c16.index = c16.index.date
        c16 = _price_at(m5, B5, tz, S["close"], 60).combine_first(c16)
    else:
        c16 = _price_at(m5, B5, tz, S["close"], 60).combine_first(_price_at(h1, pd.Timedelta(hours=1), tz, S["close"], 60))
    days = sorted(set(c16.index) | set(pT.index))
    df = pd.DataFrame({"pT": pT, "c": c16}).reindex(days)
    df = df[[pd.Timestamp(d).weekday() < 5 for d in df.index]]
    df["prevc"] = df.c.shift(1)
    df["move"] = df.pT / df.prevc - 1
    sig = df.move.rolling(60, min_periods=40).std().shift(1)
    if day not in df.index:
        df.loc[day] = np.nan
        df.loc[day, "prevc"] = df.c.dropna().iloc[-1] if df.c.notna().any() else np.nan
        sig.loc[day] = df.move.dropna().iloc[-60:].std()
    pc, sg = df.loc[day, "prevc"], sig.get(day)
    if not (pc == pc and sg == sg and sg > 0):
        return dict(out, status="NO DATA", status_text="Not enough history for the trigger levels.")
    up, dn = pc * (1 + S["k"] * sg), pc * (1 - S["k"] * sg)
    out["nd"] = _nd(pc)
    out["levels"] = [("Previous close", _r(pc)), ("Long trigger (at decision time)", _r(up)), ("Short trigger (at decision time)", _r(dn)),
                     ("Normal move to decision time (1 sigma)", f"{sg * 100:.2f}%")]
    out["plan"] = dict(long=dict(trigger=_r(up), stop_rule="entry - (entry - prev close)/2"),
                       short=dict(trigger=_r(dn), stop_rule="entry + (prev close - entry)/2"))
    x = _done(m5.tz_convert(tz), now)
    today = x[x.index.date == day]
    out["chart"] = _chart(today[(today.index >= T - pd.Timedelta(hours=4)) & (today.index < tc)], tz)
    if now < T + B5:
        last = today.close.iloc[-1] if len(today) else None
        if last is not None:
            mv = last / pc - 1
            out["levels"].append(("Now vs previous close", f"{mv * 100:+.2f}% ({mv / sg:+.1f} sigma)"))
        return dict(out, status="WAITING", status_text=f"Decision at {S['T']} New York: trade only if price is beyond a trigger then.")
    p = df.loc[day, "pT"] if df.loc[day, "pT"] == df.loc[day, "pT"] else None
    if p is None:
        return dict(out, status="NO DATA", status_text="No price at the decision time (holiday?).")
    z = (p / pc - 1) / sg
    out["levels"].append(("Price at decision", _r(p)))
    out["levels"].append(("Move at decision", f"{(p / pc - 1) * 100:+.2f}% ({z:+.1f} sigma)"))
    if abs(z) < S["k"]:
        return dict(out, status="NO TRADE", status_text=f"Move was {z:+.1f} sigma at {S['T']}: below the {S['k']:.0f} sigma trigger.")
    d = 1 if z > 0 else -1
    w = today[(today.index >= T) & (today.index < tc)]
    if not len(w):
        return dict(out, status="NO DATA", status_text="No bars after the decision time yet.")
    o, h, l, c = w.open.values, w.high.values, w.low.values, w.close.values
    entry = o[0]
    risk = 0.5 * abs(p / pc - 1) * entry
    stop = entry - d * risk
    out["trade"] = _trade_state(d, entry, stop, None, o, h, l, c, 0, w.index, now >= tc, S["cost"])
    st = "IN TRADE" if out["trade"]["state"] == "OPEN" else "DONE"
    return dict(out, status=st, status_text=f"{out['trade']['dir']}: move {z:+.1f} sigma at {S['T']}.")


FN = dict(london=sys_london, orb=sys_orb, noise=sys_noise, bigmove=sys_bigmove)


# ------------------------------------------------------------------ live data
def fetch_yahoo(ticker):
    import time
    import yfinance as yf
    out = {}
    for key, kw in (("m5", dict(period="60d", interval="5m")), ("h1", dict(period="730d", interval="1h")), ("d1", dict(period="2y", interval="1d"))):
        for k in range(3):
            try:
                df = yf.download(ticker, auto_adjust=False, progress=False, threads=False, **kw)
                if df is not None and len(df):
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [str(c).lower() for c in df.columns]
                    if key == "d1":
                        # daily bars: index is the session date; stamp it at noon local so .date stays right after tz moves
                        df.index = pd.to_datetime(df.index).tz_localize(None) + pd.Timedelta(hours=12)
                        df.index = df.index.tz_localize("UTC")
                    out[key] = _utc(df)
                    break
            except Exception as e:
                print("[intraday] yahoo", ticker, key, e)
            time.sleep(2 * (k + 1))
    return out


def build(now=None, data=None):
    now = pd.Timestamp(now or dt.datetime.now(UTC))
    cache, res, errors = dict(data or {}), [], []
    for S in SYSTEMS:
        if S["yf"] not in cache:
            cache[S["yf"]] = fetch_yahoo(S["yf"])
        dd = cache[S["yf"]]
        if "m5" not in dd:
            errors.append(f"{S['id']}: no 5-minute data")
            res.append(dict(_base(S, now.date(), 2), status="NO DATA", status_text="Live data unavailable this run."))
            continue
        try:
            r = FN[S["kind"]](S, dd["m5"], now, h1=dd.get("h1"), d1=dd.get("d1"))
            last = dd["m5"].index.max()
            r["data_last_bar"] = _iso(last)
            res.append(r)
        except Exception as e:
            errors.append(f"{S['id']}: {e!r}")
            res.append(dict(_base(S, now.date(), 2), status="ERROR", status_text=f"Engine error: {e!r}"[:200]))
    return dict(generated_at=now.strftime("%Y-%m-%d %H:%M UTC"), systems=res, errors=errors)

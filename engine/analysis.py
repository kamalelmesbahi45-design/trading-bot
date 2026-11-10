"""Per-market analysis + cross-market aggregation -> one JSON document for the dashboard."""
from __future__ import annotations
import datetime as dt
import math
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .features import add_features, vol_regime
from . import journal as J
from .edges import SETUPS, signal_series, triple_barrier, summarize, benjamini_hochberg, grade, kelly_risk


def _f(x, nd=6):
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return round(x, nd)


def _price_nd(p: float) -> int:
    return 2 if p >= 100 else (4 if p >= 1 else 5)


# ----------------------------------------------------------------------------- load
def load_market(m: dict, source: str, local_dir: str | None):
    df, used = None, None
    if source in ("yahoo", "auto"):
        df = D.load_yahoo(m["yf"], C.HISTORY_YEARS)
        used = "Yahoo Finance (yfinance)" if df is not None else None
    if df is None and source in ("mirror", "auto"):
        df = D.load_mirror(m["id"], local_dir)
        used = "GitHub mirror dataset" if df is not None else None
    if df is None:
        return None, None
    # never trade off an unfinished candle: drop any bar dated today (UTC)
    today = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    df = df[df.index < today]
    if m["cls"] == "FX major":
        # spot FX trades ~24h: Yahoo's daily open is a snapshot from another time stamp, which
        # creates fake gaps. Standard cleaning: open = previous close (except after weekends).
        prev = df["Close"].shift()
        wk = (df.index.to_series().diff().dt.days <= 1).values
        df = df.copy()
        df.loc[wk, "Open"] = prev[wk]
        df["High"] = df[["Open", "High", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    if len(df) == 0:
        return None, None
    start = df.index[-1] - pd.Timedelta(days=int(365.25 * C.HISTORY_YEARS))
    return df[df.index >= start], used


# ----------------------------------------------------------------------------- levels
def level_stats(f: pd.DataFrame, ohlc: bool) -> dict | None:
    """Next-session probabilities for day traders, conditioned on today's close location + trend."""
    if not ohlc or len(f) < 300:
        return None
    g = f.copy()
    g["nH"], g["nL"], g["nO"], g["nC"] = g.High.shift(-1), g.Low.shift(-1), g.Open.shift(-1), g.Close.shift(-1)
    g = g.dropna(subset=["nH", "atr", "sma50", "ibs"])
    g["brk_hi"] = g.nH > g.High
    g["brk_lo"] = g.nL < g.Low
    g["up"] = g.nC > g.Close
    g["ibs_b"] = pd.cut(g.ibs, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low", "mid", "high"])
    g["trend"] = np.where(g.Close > g.sma50, "up", "down")
    last = f.iloc[-1]
    cur_ibs = "low" if last.ibs <= 1 / 3 else ("mid" if last.ibs <= 2 / 3 else "high")
    cur_tr = "up" if last.Close > last.sma50 else "down"
    sub = g[(g.ibs_b == cur_ibs) & (g.trend == cur_tr)]
    rec = g[g.index >= g.index[-1] - pd.Timedelta(days=3 * 365)]
    touch = []
    for k in (0.25, 0.5, 1.0, 1.5):
        up = ((rec.nH - rec.nO) >= k * rec.atr).mean()
        dn = ((rec.nO - rec.nL) >= k * rec.atr).mean()
        touch.append(dict(k=k, up=_f(up, 3), dn=_f(dn, 3),
                          up_px=_f(last.Close + k * last.atr), dn_px=_f(last.Close - k * last.atr)))
    rng_ratio = ((rec.nH - rec.nL) / rec.atr)
    P = (last.High + last.Low + last.Close) / 3
    return dict(
        cond=f"close location {cur_ibs} (IBS {last.ibs:.2f}) & trend {cur_tr}",
        n=int(len(sub)),
        p_break_high=_f(sub.brk_hi.mean(), 3), p_break_low=_f(sub.brk_lo.mean(), 3),
        p_both=_f((sub.brk_hi & sub.brk_lo).mean(), 3), p_close_up=_f(sub.up.mean(), 3),
        base_break_high=_f(g.brk_hi.mean(), 3), base_break_low=_f(g.brk_lo.mean(), 3), base_close_up=_f(g.up.mean(), 3),
        pdh=_f(last.High), pdl=_f(last.Low), pdc=_f(last.Close), pivot=_f(P),
        r1=_f(2 * P - last.Low), s1=_f(2 * P - last.High),
        touch=touch, n_touch=int(len(rec)),
        range_median_atr=_f(rng_ratio.median(), 2), range_p80_atr=_f(rng_ratio.quantile(0.8), 2),
    )


def seasonality(f: pd.DataFrame) -> dict:
    mr = f["Close"].resample("ME").last().pct_change().dropna()
    out = []
    for mth in range(1, 13):
        x = mr[mr.index.month == mth]
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")
        out.append(dict(m=mth, mean=_f(x.mean() * 100, 2), hit=_f((x > 0).mean(), 2), n=int(len(x)), t=_f(t, 2)))
    return dict(months=out, base=_f(mr.mean() * 100, 2))


def hourly_profile(intra: pd.DataFrame | None) -> list | None:
    if intra is None or len(intra) < 500:
        return None
    h = intra["Close"].resample("1h").last().dropna()
    r = np.log(h).diff().abs().dropna()
    r = r[r.index >= r.index[-1] - pd.Timedelta(days=365)]
    prof = r.groupby(r.index.hour).mean()
    tot = prof.sum()
    return [dict(h=int(k), share=_f(v / tot, 4), absret_bp=_f(v * 1e4, 2)) for k, v in prof.items()]


def cot_summary(m: dict) -> dict | None:
    if not m.get("cftc") or getattr(D, "OFFLINE", False):
        return None
    cot = D.load_cot(m["cftc"])
    if cot is None or len(cot) < 20:
        return None
    x = cot["net_pct_oi"] * (-1 if m["cot_inv"] else 1)
    pct = (x.rank(pct=True)).iloc[-1]
    return dict(date=str(cot.index[-1].date()), net_pct_oi=_f(x.iloc[-1], 1), pct_3y=_f(pct, 2),
                chg_4w=_f(x.iloc[-1] - x.iloc[-5], 1) if len(x) > 5 else None,
                series=[dict(t=str(d.date()), v=_f(v, 1)) for d, v in x.tail(156).items()])


# ----------------------------------------------------------------------------- per market
def analyse_market(m: dict, df: pd.DataFrame, used: str, macro_scores: dict, intra=None) -> dict:
    crypto = m["cls"] == "Crypto"
    ppy = 365 if crypto else 252
    if not crypto:                      # futures feeds sometimes emit Sunday-evening stub bars
        df = df[df.index.dayofweek < 5]
    ohlc = D.has_real_ohlc(df)
    # data-quality check: on clean daily data the median session range is ~0.8-1.0 x ATR
    quality = []
    if ohlc:
        tr_ = df.tail(500)
        rr = ((tr_.High - tr_.Low) / (tr_.High - tr_.Low).rolling(14).mean()).median()
        gap = ((tr_.Open - tr_.Close.shift()).abs() / (tr_.High - tr_.Low).rolling(14).mean()).median()
        if gap > 0.35 and m["cls"] not in ("Equity index", "Rates"):   # index overnight gaps are real
            quality.append(f"Large open-vs-previous-close gaps (median {gap:.2f}x avg range): feed mixes sessions or "
                           "futures rolls -> backtest fills are less reliable")
    else:
        quality.append("Close-only data: no intraday high/low, stops simulated on closes")
    f = add_features(df, ohlc, ppy)
    last = f.iloc[-1]
    nd = _price_nd(last.Close)

    setups = []
    for s in SETUPS:
        if s.needs_ohlc and not ohlc:
            continue
        if s.only_classes and m["cls"] not in s.only_classes:
            continue
        for d in s.dirs:
            sig = signal_series(s, f, d, crypto)
            tr = triple_barrier(f, sig, d, s, ohlc, m["cost_bps"])
            stt = summarize(tr, s)
            eq = []
            if len(tr):
                cr = tr["R"].cumsum()
                step = max(1, len(tr) // 120)
                eq = [dict(t=str(tr["date"].iloc[i].date()), v=_f(cr.iloc[i], 2)) for i in range(0, len(tr), step)]
                eq.append(dict(t=str(tr["date"].iloc[-1].date()), v=_f(cr.iloc[-1], 2)))
            # regime-conditional performance (same volatility regime as today)
            reg = None
            if len(tr) and last.vol_pct == last.vol_pct:
                cur = vol_regime(last.vol_pct)
                same = tr[tr["vol_pct"].apply(vol_regime) == cur]
                if len(same) >= 10:
                    reg = dict(regime=cur, n=int(len(same)), meanR=_f(same["R"].mean(), 3),
                               hit=_f((same["R"] > 0).mean(), 3))
            setups.append(dict(key=s.key, label=s.label, style=s.style, dir=d, active=bool(sig.iloc[-1]),
                               sl_atr=s.sl_atr, tp_atr=s.tp_atr, max_hold=s.max_hold, why=s.why,
                               entry_txt=s.entry_txt, exit_txt=s.exit_txt, stats=stt, equity=eq, regime=reg,
                               _reasons=s.reasons(last, d) if bool(sig.iloc[-1]) else []))

    # directional bias score (context, NOT a probability)
    tsm = []
    for mm in (1, 3, 12):
        v = last.vol20 * math.sqrt(mm / 12)
        if v and v == v and last[f"mom{mm}"] == last[f"mom{mm}"]:
            tsm.append(max(-2, min(2, last[f"mom{mm}"] / v)) / 2)
    tsm_s = float(np.mean(tsm)) if tsm else 0.0
    struct = float(np.mean([np.sign(last.Close - last.sma50), np.sign(last.Close - last.sma200),
                            np.sign(last.sma50 - last.sma200)])) if last.sma200 == last.sma200 else 0.0
    macro_s = 0.0
    parts = []
    if macro_scores.get("risk") is not None and m["risk"]:
        parts.append(m["risk"] * macro_scores["risk"])
    if macro_scores.get("usd") is not None and m["usd"]:
        parts.append(m["usd"] * macro_scores["usd"])
    if parts:
        macro_s = float(np.mean(parts))
    cot = cot_summary(m)
    cot_s = 0.0
    if cot and cot["pct_3y"] is not None:
        cot_s = -0.5 if cot["pct_3y"] >= 0.9 else (0.5 if cot["pct_3y"] <= 0.1 else 0.0)
    bias = 100 * (0.45 * tsm_s + 0.30 * struct + 0.15 * macro_s + 0.10 * cot_s)

    def chg(n):
        return _f((f.Close.iloc[-1] / f.Close.iloc[-1 - n] - 1) * 100, 2) if len(f) > n else None

    tail = f.tail(300)
    candles = [dict(t=str(i.date()), o=_f(r.Open, nd + 1), h=_f(r.High, nd + 1), l=_f(r.Low, nd + 1),
                    c=_f(r.Close, nd + 1)) for i, r in tail.iterrows()]
    line = lambda col: [dict(t=str(i.date()), v=_f(v, nd + 1)) for i, v in tail[col].items() if v == v]

    return dict(
        id=m["id"], name=m["name"], cls=m["cls"], source=used, ohlc=ohlc, nd=nd, quality=quality,
        first_date=str(f.index[0].date()), last_date=str(f.index[-1].date()), bars=int(len(f)),
        price=_f(last.Close, nd + 1), chg1d=chg(1), chg1w=chg(5), chg1m=chg(21), chg3m=chg(63), chg1y=chg(ppy),
        atr=_f(last.atr, nd + 1), atr_pct=_f(100 * last.atr / last.Close, 2), vol20=_f(last.vol20 * 100, 1),
        vol_pct=_f(last.vol_pct, 2), vol_regime=vol_regime(last.vol_pct), rsi2=_f(last.rsi2, 1), rsi14=_f(last.rsi14, 1),
        sma50=_f(last.sma50, nd + 1), sma200=_f(last.sma200, nd + 1), hh20=_f(last.hh20, nd + 1), ll20=_f(last.ll20, nd + 1),
        hh55=_f(last.hh55, nd + 1), ll55=_f(last.ll55, nd + 1), mom1=_f(last.mom1 * 100, 1), mom3=_f(last.mom3 * 100, 1),
        mom12=_f(last.mom12 * 100, 1), bbw_pct=_f(last.bbw_pct, 2), ibs=_f(last.ibs, 2),
        trend="Up" if struct > 0.3 else ("Down" if struct < -0.3 else "Mixed"),
        bias=_f(max(-100, min(100, bias)), 0),
        bias_parts=dict(momentum=_f(tsm_s, 2), structure=_f(struct, 2), macro=_f(macro_s, 2), cot=_f(cot_s, 2)),
        candles=candles, sma50_line=line("sma50"), sma200_line=line("sma200"), ema20_line=line("ema20"),
        setups=setups, levels=level_stats(f, ohlc), season=seasonality(f), hourly=hourly_profile(intra), cot=cot,
        _rets=f["ret"],
    )


# ----------------------------------------------------------------------------- macro
def macro_panel(source: str, local_dir: str | None, spx: pd.DataFrame | None) -> dict:
    out = dict(items=[], risk=None, usd=None, notes=[])
    risk_votes = []

    def add(key, label, val, chg=None, unit="", read=None, date=None):
        out["items"].append(dict(key=key, label=label, value=_f(val, 2), chg=_f(chg, 2), unit=unit, read=read,
                                 date=date))

    # VIX
    vix = D.load_yahoo(C.MACRO_YF["VIX"], 3) if source != "mirror" else None
    if vix is None:
        vix = D.load_mirror("VIX", local_dir)
    if vix is not None and len(vix) > 260:
        v = vix.Close
        pct = (v.tail(252) <= v.iloc[-1]).mean()
        risk_votes.append(1 if v.iloc[-1] < v.tail(252).median() else -1)
        add("VIX", "VIX (equity fear gauge)", v.iloc[-1], v.iloc[-1] - v.iloc[-6], "",
            f"higher than {pct * 100:.0f}% of the last year -> {'calm' if pct < 0.5 else 'stressed'}", str(v.index[-1].date()))
        out["vix_series"] = [dict(t=str(i.date()), v=_f(x, 2)) for i, x in v.tail(260).items()]
    v3 = D.load_yahoo(C.MACRO_YF["VIX3M"], 2) if source != "mirror" else None
    if vix is not None and v3 is not None and len(v3):
        ratio = vix.Close.iloc[-1] / v3.Close.iloc[-1]
        risk_votes.append(1 if ratio < 1 else -1)
        add("VIXTS", "VIX / VIX3M (term structure)", ratio, None, "",
            "contango = normal" if ratio < 1 else "BACKWARDATION = acute stress", str(v3.index[-1].date()))
    # S&P trend
    if spx is not None and len(spx) > 200:
        s200 = spx.Close.rolling(200).mean().iloc[-1]
        risk_votes.append(1 if spx.Close.iloc[-1] > s200 else -1)
        add("SPXT", "S&P 500 vs 200-day avg", 100 * (spx.Close.iloc[-1] / s200 - 1), None, "%",
            "equity trend up" if spx.Close.iloc[-1] > s200 else "equity trend down", str(spx.index[-1].date()))
    # DXY
    dxy = D.load_yahoo(C.MACRO_YF["DXY"], 3) if source != "mirror" else None
    dlab = "US Dollar Index (DXY)"
    if dxy is None:
        dxy = D.load_mirror("DXY", local_dir)
        dlab = "US Dollar Index (rebuilt from ICE formula)"
    if dxy is not None and len(dxy) > 60:
        c = dxy.Close
        s50 = c.rolling(50).mean().iloc[-1]
        m1 = c.iloc[-1] / c.iloc[-22] - 1
        usd = float(np.clip((np.sign(c.iloc[-1] - s50) + np.sign(m1)) / 2, -1, 1))
        out["usd"] = usd
        add("DXY", dlab, c.iloc[-1], 100 * m1, "", f"1m {m1 * 100:+.1f}%, {'above' if c.iloc[-1] > s50 else 'below'} 50d avg -> USD "
            f"{'strong' if usd > 0 else ('weak' if usd < 0 else 'neutral')}", str(c.index[-1].date()))
        out["dxy_series"] = [dict(t=str(i.date()), v=_f(x, 3)) for i, x in c.tail(260).items()]
    # Rates & credit (FRED)
    if source != "mirror":
        for sid, lab in C.MACRO_FRED.items():
            s = D.load_fred(sid)
            if s is None or len(s) < 30:
                continue
            ch = s.iloc[-1] - s.iloc[-22]
            read = None
            if sid == "BAMLH0A0HYM2":
                risk_votes.append(1 if ch <= 0 else -1)
                read = "credit spreads tightening (risk-on)" if ch <= 0 else "credit spreads widening (risk-off)"
            if sid == "T10Y2Y":
                read = "curve inverted" if s.iloc[-1] < 0 else "curve positive"
            if sid == "DFII10":
                read = "rising real yields = headwind for gold" if ch > 0 else "falling real yields = tailwind for gold"
            add(sid, lab, s.iloc[-1], ch, "", read, str(s.index[-1].date()))
        tnx = D.load_yahoo(C.MACRO_YF["US10Y"], 2)
        if tnx is not None and len(tnx) > 22:
            add("US10Y", "US 10Y yield", tnx.Close.iloc[-1], tnx.Close.iloc[-1] - tnx.Close.iloc[-22], "%",
                "1-month change", str(tnx.index[-1].date()))
    else:
        out["notes"].append("Rates/credit (FRED) and VIX term structure are fetched in the live GitHub run only.")
    if risk_votes:
        out["risk"] = float(np.mean(risk_votes))
        out["risk_votes"] = len(risk_votes)
    r = out["risk"]
    out["regime"] = ("n/a" if r is None else "RISK-ON" if r >= 0.5 else "RISK-OFF" if r <= -0.5 else "MIXED")
    return out


# ----------------------------------------------------------------------------- aggregate
def build(source: str = "auto", local_dir: str | None = None, intraday: bool = True) -> dict:
    t0 = dt.datetime.now(dt.timezone.utc)
    D.OFFLINE = source == "mirror"
    loaded = {}
    for m in C.MARKETS:
        df, used = load_market(m, source, local_dir)
        if df is not None and len(df) > 300:
            loaded[m["id"]] = (m, df, used)
        else:
            D._err(f"{m['id']}: no data from source '{source}'")
    spx = loaded.get("SPX500", (None, None, None))[1]
    macro = macro_panel(source, local_dir, spx)

    results = []
    for mid, (m, df, used) in loaded.items():
        intra = None
        if intraday:
            if source != "mirror":
                intra = D.load_yahoo_intraday(m["yf"])
            elif mid == "BTCUSD":
                mm = D.btc_minutes(lambda u: (local_dir + "/btc1m.csv") if local_dir else u)
                intra = mm
        results.append(analyse_market(m, df, used, macro, intra))

    # ---- false discovery control across every market x setup x direction test
    tests = [(r, s) for r in results for s in r["setups"] if s["stats"].get("valid")]
    flags = benjamini_hochberg([s["stats"]["pval"] for _, s in tests], C.FDR_Q)
    for (r, s), fl in zip(tests, flags):
        s["stats"]["fdr_pass"] = bool(fl)
    for r in results:
        for s in r["setups"]:
            s["grade"] = grade(s["stats"])
    n_tests, n_pass = len(tests), int(sum(flags))

    # ---- correlation matrix (60 sessions, common dates)
    rets = pd.DataFrame({r["id"]: r["_rets"] for r in results}).dropna(how="all")
    rets = rets[rets.index >= rets.index[-1] - pd.Timedelta(days=180)]
    corr = rets.tail(90).corr(min_periods=30)
    ids = list(corr.columns)

    # ---- today's trade plans
    signals = []
    for r in results:
        for s in r["setups"]:
            if not s["active"]:
                continue
            st_ = s["stats"]
            d = s["dir"]
            px, a, nd = r["price"], r["atr"], r["nd"]
            sl = px - d * s["sl_atr"] * a
            tp = px + d * s["tp_atr"] * a
            g = s["grade"]
            p = st_.get("p_post") if st_.get("valid") else None
            b = (abs(st_.get("avg_win", 0)) / abs(st_.get("avg_loss", -1))) if st_.get("valid") and st_.get("avg_loss") else 0
            risk = 0.0
            if g in ("A", "B") and p:
                risk = min(kelly_risk(p, b), C.BASE_RISK_PCT * (1.0 if g == "A" else 0.6))
            stop_dist = abs(px - sl)
            units = (10000 * risk / 100) / stop_dist if stop_dist > 0 and risk > 0 else 0
            reasons = list(s.pop("_reasons"))
            if st_.get("valid"):
                reasons.append(f"History on {r['name']}: {st_['n']} trades, hit {st_['hit'] * 100:.0f}% "
                               f"(95% CI {st_['hit_lo'] * 100:.0f}-{st_['hit_hi'] * 100:.0f}%), avg {st_['meanR']:+.2f}R net of costs")
                reasons.append(f"Out-of-sample (last 30% of trades): {st_['oos_mean']:+.2f}R per trade")
            if s.get("regime"):
                rg = s["regime"]
                reasons.append(f"In the current {rg['regime'].lower()}-volatility regime: {rg['n']} trades, avg {rg['meanR']:+.2f}R")
            for q in r.get("quality", []):
                reasons.append("Data caveat: " + q)
            # time stop date
            if r["cls"] == "Crypto":
                tstop = (pd.Timestamp(r["last_date"]) + pd.Timedelta(days=s["max_hold"])).date()
            else:
                tstop = (pd.Timestamp(r["last_date"]) + pd.offsets.BDay(s["max_hold"])).date()
            aligned = (r["bias"] or 0) * d > 0
            signals.append(dict(
                market=r["id"], name=r["name"], cls=r["cls"], setup=s["key"], label=s["label"], style=s["style"],
                dir="LONG" if d > 0 else "SHORT", grade=g, asof=r["last_date"],
                entry=_f(px, nd + 1), entry_txt=s["entry_txt"], stop=_f(sl, nd + 1), target=_f(tp, nd + 1),
                rr=_f(s["tp_atr"] / s["sl_atr"], 2), time_stop=str(tstop), exit_txt=s["exit_txt"],
                prob=_f(p, 3), p_be=_f(st_.get("p_be"), 3), exp_R=_f(st_.get("e_post"), 3), n=st_.get("n"),
                risk_pct=_f(risk, 2), units_per_10k=_f(units, 4 if px < 10 else 2),
                bias=r["bias"], bias_aligned=aligned, why=s["why"], reasons=reasons,
                invalidation=(f"Thesis is wrong if price trades {'below' if d > 0 else 'above'} {sl:,.{nd}f} "
                              f"(=-1R) or nothing happens by {tstop}."),
            ))
    gorder = {"A": 0, "B": 1, "C": 2, "X": 3}
    signals.sort(key=lambda x: (gorder[x["grade"]], -(x["exp_R"] or -9)))

    # same-bet warnings among tradeable signals
    tradeable = [s for s in signals if s["grade"] in ("A", "B")]
    clusters = []
    for i in range(len(tradeable)):
        for j in range(i + 1, len(tradeable)):
            a, b = tradeable[i], tradeable[j]
            if a["market"] in ids and b["market"] in ids:
                c = corr.loc[a["market"], b["market"]]
                same = c * (1 if a["dir"] == b["dir"] else -1)
                if same == same and same > 0.6:
                    clusters.append(dict(a=a["market"], b=b["market"], corr=_f(c, 2)))

    for r in results:
        r.pop("_rets", None)

    return dict(
        generated_at=t0.strftime("%Y-%m-%d %H:%M UTC"), source=source,
        snapshot=any(v[2] != "Yahoo Finance (yfinance)" for v in loaded.values()),
        universe=[dict(id=m["id"], name=m["name"], cls=m["cls"]) for m in C.MARKETS],
        markets=results, macro=macro, signals=signals, clusters=clusters,
        corr=dict(ids=ids, m=[[_f(corr.loc[a, b], 2) for b in ids] for a in ids]),
        calendar=D.load_calendar() if source != "mirror" else [],
        journal=J.score({k: v[1] for k, v in loaded.items()}),
        tests=dict(n=n_tests, fdr_pass=n_pass, q=C.FDR_Q),
        settings=dict(history_years=C.HISTORY_YEARS, prior=C.PRIOR_STRENGTH, oos=C.OOS_FRACTION,
                      base_risk=C.BASE_RISK_PCT, max_risk=C.MAX_RISK_PCT, kelly=C.KELLY_FRACTION),
        missing=[m["id"] for m in C.MARKETS if m["id"] not in loaded],
        errors=D.ERRORS[-40:],
    )

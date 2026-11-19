"""
Round 2: (1) new pre-declared setups — Crabel NR4/inside-day opening-range breakout, sweep CONTINUATION;
(2) meta-labeling (Lopez de Prado): a walk-forward gradient-boosting filter that only keeps the
candidate trades whose context looked best in PAST years. Every prediction for year Y is made by a
model trained only on trades before Jan 1 of Y, so all reported filtered results are out-of-sample.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from research import (MKT, INDEX_LIKE, FX_LIKE, load, sessions, run_trade, R_of, stop_entry,
                      stats_of, bh, OOS_START)


def day_features(df, mid):
    """Daily context known before the session (all shifted by one day)."""
    _, tz, (s0, s1), _ = MKT[mid]
    if mid in FX_LIKE:
        x = df.copy()
        x.index = x.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
        day = x.index.date
    else:
        day = df.index.date
    d = df.groupby(day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    d.index = pd.to_datetime(d.index)
    if mid != "BTCUSD":
        d = d[d.index.dayofweek < 5]
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    rng = d.high - d.low
    t = df.index.strftime("%H:%M")
    ses = df[(t >= s0) & (t < s1)]
    sd = ses.groupby(ses.index.date).agg(sh=("high", "max"), sl=("low", "min"), sc=("close", "last"), so=("open", "first"))
    sd.index = pd.to_datetime(sd.index)
    d = d.join(sd, how="left")
    use_ses = mid in INDEX_LIKE and mid != "BTCUSD"
    H, L, Cc = (d.sh, d.sl, d.sc) if use_ses else (d.high, d.low, d.close)
    f = pd.DataFrame(index=d.index)
    f["atr"] = atr.shift()
    f["trend"] = np.sign(d.close - d.close.rolling(50).mean()).shift()
    f["dist50"] = ((d.close - d.close.rolling(50).mean()) / atr).shift()
    f["ret1"] = ((d.close - d.close.shift()) / atr).shift()
    f["ret5"] = ((d.close - d.close.shift(5)) / atr).shift()
    f["atr_ratio"] = (atr / atr.rolling(100).mean()).shift()
    f["prev_rng"] = (rng / atr).shift()
    f["nr4"] = (rng <= rng.rolling(4).min()).shift().astype(float)
    f["inside"] = ((d.high <= d.high.shift()) & (d.low >= d.low.shift())).shift().astype(float)
    f["pdh"], f["pdl"], f["pdc"] = H.shift(), L.shift(), Cc.shift()
    f["dow"] = f.index.dayofweek
    # gap of today's session open vs previous session close (known at the open)
    f["gap"] = (d.so - Cc.shift()) / atr.shift() if use_ses else 0.0
    return f


# ---------------------------------------------------------------- candidate generators (with features)
def cand_orb_range(mid, df, F, W, exit_mode, filt=None):
    _, _, (s0, s1), cost = MKT[mid]
    nb = W // 5
    out = []
    for day, g in sessions(df, s0, s1):
        if day not in F.index or len(g) < nb + 14:
            continue
        fx = F.loc[day]
        if filt == "nr4_inside" and not (fx.nr4 == 1 or fx.inside == 1):
            continue
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        hi, lo = h[:nb].max(), l[:nb].min()
        if hi <= lo or not fx.atr > 0:
            continue
        e = stop_entry(o, h, l, nb, len(g) - 12, hi, lo)
        if e is None:
            continue
        k, d, fill = e
        if d == 0:
            out.append(dict(day=day, dir=0, R=-1.0, outcome="whipsaw", k=k, risk_atr=(hi - lo) / fx.atr, rng_atr=(hi - lo) / fx.atr))
            continue
        stop = lo if d > 0 else hi
        risk = d * (fill - stop)
        tgt = fill + d * 2 * risk if exit_mode == "2R" else None
        ex, oc, b = run_trade(o, h, l, c, k, fill, d, stop, tgt)
        out.append(dict(day=day, dir=d, R=R_of(fill, ex, d, risk, cost), outcome=oc, k=k,
                        risk_atr=risk / fx.atr, rng_atr=(hi - lo) / fx.atr))
    return out


def cand_london(mid, df, F, stop_mode, exit_mode, filt=None):
    _, _, _, cost = MKT[mid]
    t = df.index.strftime("%H:%M")
    asia = df[t < "07:00"]
    ldn = df[(t >= "07:00") & (t < "17:00")]
    ag = {pd.Timestamp(d): g for d, g in asia.groupby(asia.index.date)}
    out = []
    for d0, g in ldn.groupby(ldn.index.date):
        day = pd.Timestamp(d0)
        if day.dayofweek >= 5 or day not in ag or day not in F.index or len(ag[day]) < 60 or len(g) < 60:
            continue
        fx = F.loc[day]
        if filt == "nr4_inside" and not (fx.nr4 == 1 or fx.inside == 1):
            continue
        a = ag[day]
        hi, lo = a.high.max(), a.low.min()
        if hi <= lo or not fx.atr > 0:
            continue
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        e = stop_entry(o, h, l, 0, 60, hi, lo)
        if e is None:
            continue
        k, d, fill = e
        if d == 0:
            out.append(dict(day=day, dir=0, R=-1.0, outcome="whipsaw", k=k, risk_atr=(hi - lo) / fx.atr, rng_atr=(hi - lo) / fx.atr))
            continue
        stop = (lo if d > 0 else hi) if stop_mode == "opp" else (hi + lo) / 2
        risk = d * (fill - stop)
        if risk <= 0:
            continue
        tgt = fill + d * 2 * risk if exit_mode == "2R" else None
        ex, oc, b = run_trade(o, h, l, c, k, fill, d, stop, tgt)
        out.append(dict(day=day, dir=d, R=R_of(fill, ex, d, risk, cost), outcome=oc, k=k,
                        risk_atr=risk / fx.atr, rng_atr=(hi - lo) / fx.atr))
    return out


def cand_sweep_cont(mid, df, F, k_R):
    """Continuation after a break of yesterday's high/low: first 15-min close beyond the level ->
    go WITH the break; stop on the other side of the level by 0.25 ATR... (risk capped)."""
    _, _, (s0, s1), cost = MKT[mid]
    if mid == "BTCUSD":
        s0, s1 = "03:00", "16:00"
    out = []
    for day, g in sessions(df, s0, s1):
        if day not in F.index:
            continue
        fx = F.loc[day]
        if not fx.atr > 0 or math.isnan(fx.pdh):
            continue
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        n = len(c)
        if c[0] > fx.pdh or c[0] < fx.pdl:            # opened outside: not a break during the session
            continue
        for j in range(2, n - 13, 3):
            d = 1 if c[j] > fx.pdh else (-1 if c[j] < fx.pdl else 0)
            if d == 0:
                continue
            level = fx.pdh if d > 0 else fx.pdl
            entry = o[j + 1]
            stop = level - d * 0.25 * fx.atr
            risk = d * (entry - stop)
            if not (0.05 * fx.atr <= risk <= 0.6 * fx.atr):
                break
            ex, oc, b = run_trade(o, h, l, c, j + 1, entry, d, stop, entry + d * k_R * risk)
            out.append(dict(day=day, dir=d, R=R_of(entry, ex, d, risk, cost), outcome=oc, k=j + 1,
                            risk_atr=risk / fx.atr, rng_atr=np.nan))
            break
    return out


def cand_noise(mid, df, F):
    from research import setup_noise
    rows = setup_noise(mid, df, F.rename(columns={}))
    # NOTE: setup_noise's last field is the EXIT bar -> must never be used as a feature (look-ahead)
    return [dict(day=d, dir=di, R=r, outcome=oc, k=np.nan, risk_atr=np.nan, rng_atr=np.nan) for d, di, r, oc, b in rows]


POOL = []   # (setup_name, market, fn, kwargs)
for m in INDEX_LIKE:
    for W in (15, 30):
        for ex in ("EOD", "2R"):
            POOL.append((f"ORB{W}_{ex}", m, cand_orb_range, dict(W=W, exit_mode=ex)))
    POOL.append(("NOISE", m, cand_noise, {}))
for m in FX_LIKE:
    for sm in ("opp", "mid"):
        for ex in ("EOD", "2R"):
            POOL.append((f"LDN_{sm}_{ex}", m, cand_london, dict(stop_mode=sm, exit_mode=ex)))
NEW_TESTS = []
for m in INDEX_LIKE:
    for ex in ("EOD", "2R"):
        NEW_TESTS.append((f"CRABEL30_{ex}", m, cand_orb_range, dict(W=30, exit_mode=ex, filt="nr4_inside")))
for m in FX_LIKE:
    for ex in ("EOD", "2R"):
        NEW_TESTS.append((f"CRABEL_LDN_{ex}", m, cand_london, dict(stop_mode="opp", exit_mode=ex, filt="nr4_inside")))
for m in INDEX_LIKE + FX_LIKE:
    for kR in (2, 3):
        NEW_TESTS.append((f"BREAK_CONT_{kR}R", m, cand_sweep_cont, dict(k_R=kR)))

FEATS = ["dir", "trend_al", "dist50_al", "ret1_al", "ret5_al", "gap_al", "atr_ratio", "prev_rng", "nr4", "inside",
         "dow", "k", "risk_atr", "rng_atr", "mkt_code", "setup_code"]


def add_feats(tr: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    x = tr.join(F, on="day")
    s = x["dir"].replace(0, np.nan)
    for c in ("trend", "dist50", "ret1", "ret5", "gap"):
        x[c + "_al"] = x[c] * s                       # aligned with the trade direction
    return x


def walk_forward(pool: pd.DataFrame, first_test_year=2018, keep_q=0.70):
    """Train on trades before year Y, predict expected R for year Y; keep trades whose prediction
    exceeds the keep_q quantile of the TRAINING predictions and is > 0."""
    pool = pool[pool.dir != 0].copy()
    pool["year"] = pool.day.dt.year
    preds = []
    for Y in range(first_test_year, pool.year.max() + 1):
        trn, tst = pool[pool.year < Y], pool[pool.year == Y]
        if len(tst) == 0 or len(trn) < 2000:
            continue
        mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                            l2_regularization=1.0, random_state=0)
        y = trn.R.clip(-1.5, 4)                       # winsorise the target
        mdl.fit(trn[FEATS], y)
        thr = max(0.0, float(np.quantile(mdl.predict(trn[FEATS]), keep_q)))
        p = mdl.predict(tst[FEATS])
        t = tst.assign(pred=p, keep=p > thr)
        preds.append(t)
    return pd.concat(preds) if preds else pd.DataFrame()


def main(out="rout2"):
    os.makedirs(out, exist_ok=True)
    data, feats = {}, {}
    mk = sorted(MKT)
    for m in mk:
        data[m] = load(m)
        feats[m] = day_features(data[m], m)
        print("loaded", m, flush=True)
    # ---- new pre-declared tests (count with round 1 for FDR)
    rows = []
    for name, m, fn, kw in NEW_TESTS:
        tr = pd.DataFrame(fn(m, data[m], feats[m], **kw))
        s = stats_of(tr) if len(tr) else dict(n=0, valid=False)
        rows.append(dict(key=f"{name}|{m}", setup=name, market=m, **s))
        print(f"{name}|{m}", s.get("n"), s.get("meanR"), s.get("oos_meanR"), round(s.get("p", 1), 4), flush=True)
    new = pd.DataFrame(rows)
    r1 = pd.read_csv(os.environ.get("R1", "rout/results.csv"))
    allp = pd.concat([r1[["key", "p", "valid"]], new[["key", "p", "valid"]]])
    v = allp.valid.fillna(False).astype(bool)
    passed = set(allp[v].key[bh(allp[v].p.values)])
    new["fdr_all"] = new.key.isin(passed)
    new.sort_values("p").to_csv(os.path.join(out, "new_tests.csv"), index=False)
    print("FDR passes across", int(v.sum()), "tests:", sorted(passed), flush=True)

    # ---- candidate pool + meta-labeling
    parts = []
    for name, m, fn, kw in POOL:
        tr = pd.DataFrame(fn(m, data[m], feats[m], **kw))
        if not len(tr):
            continue
        tr["setup"], tr["market"] = name, m
        parts.append(add_feats(tr, feats[m]))
        print("pool", name, m, len(tr), flush=True)
    pool = pd.concat(parts, ignore_index=True)
    pool["mkt_code"] = pool.market.map({m: i for i, m in enumerate(mk)})
    pool["setup_code"] = pool.setup.astype("category").cat.codes
    pool.to_parquet(os.path.join(out, "pool.parquet"))
    res = {}
    # one model per family (indices / FX) keeps the problem homogeneous
    for fam, mlist in (("INDEX", INDEX_LIKE), ("FX", FX_LIKE)):
        pf = pool[pool.market.isin(mlist)]
        # several setup variants share the same day/market -> keep the BEST-predicted one per day/market later
        wf = walk_forward(pf)
        wf.to_parquet(os.path.join(out, f"wf_{fam}.parquet"))
        base = wf
        kept = wf[wf.keep]
        # one trade per market-day: highest prediction
        kept1 = kept.sort_values("pred", ascending=False).drop_duplicates(["market", "day"]).sort_values("day")
        base = base.sort_values("day")
        kept = kept.sort_values("day")
        res[fam] = dict(all_candidates=stats_of(base.rename(columns={})), kept=stats_of(kept), kept_one_per_day=stats_of(kept1),
                        kept_by_market={m: stats_of(g) for m, g in kept1.groupby("market")},
                        kept_by_setup={s: stats_of(g) for s, g in kept1.groupby("setup")})
        print(fam, json.dumps({k: res[fam][k] for k in ("all_candidates", "kept_one_per_day")}, default=str)[:900], flush=True)
    json.dump(res, open(os.path.join(out, "meta.json"), "w"), default=str, indent=1)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

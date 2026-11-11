"""
Edge library + honest evaluation.

Rules of the game (to avoid fooling ourselves):
  * Parameters are textbook values fixed IN ADVANCE. Nothing is optimised on the data,
    so the backtest is not curve-fitted.
  * Entry is at the NEXT bar's open (no look-ahead). If stop and target are both hit
    inside the same daily bar we assume the STOP was hit first (pessimistic).
    Gaps through a stop are filled at the open (worse than the stop).
  * Every result is net of a conservative round-trip cost (config.cost_bps).
  * Overlapping trades of the same setup are not counted (one position at a time).
  * Probability = Bayesian-shrunk hit rate (Beta prior centred on the break-even
    hit rate), expectancy shrunk toward zero, plus an out-of-sample split and a
    Benjamini-Hochberg false-discovery control across ALL market x setup tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats as st

from . import config as C


@dataclass
class Setup:
    key: str
    label: str
    style: str
    needs_ohlc: bool
    sl_atr: float
    tp_atr: float
    max_hold: int
    dirs: tuple
    cond: Callable          # (f, d) -> bool Series
    why: str                # literature / mechanism
    entry_txt: str
    exit_txt: str
    reasons: Callable       # (row, d) -> list[str]
    only_classes: tuple = ()


def _tom_flag(f: pd.DataFrame, calendar_days: bool) -> pd.Series:
    """True on the 2nd-to-last trading day of the month (entry next open -> hold into new month)."""
    idx = f.index
    s = pd.Series(idx, index=idx)
    months = idx.to_period("M")
    pos_from_end = s.groupby(months).cumcount(ascending=False)  # 0 = last bar of month
    flag = pos_from_end == 1
    # the current (incomplete) month: estimate remaining sessions with a calendar
    last = idx[-1]
    month_end = (last + pd.offsets.MonthEnd(0)).normalize()
    if calendar_days:
        remaining = (month_end - last).days
    else:
        remaining = len(pd.bdate_range(last + pd.Timedelta(days=1), month_end))
    cur = months == last.to_period("M")
    flag[cur] = False
    if remaining == 1:
        flag.iloc[-1] = True
    return flag


def _fmt(x, nd=None):
    if nd is None:
        ax = abs(x)
        nd = 2 if ax >= 100 else (4 if ax >= 1 else 5)
    return f"{x:,.{nd}f}"


SETUPS: list[Setup] = [
    Setup("PULLBACK", "Trend pullback (RSI-2)", "Swing 2-7d", False, 1.5, 1.5, 7, (1, -1),
          cond=lambda f, d: ((f.Close > f.sma200) & (f.mom12 > 0) & (f.rsi2 < 10)) if d > 0
          else ((f.Close < f.sma200) & (f.mom12 < 0) & (f.rsi2 > 90)),
          why="Short-term mean reversion inside a long-term trend (Connors RSI-2; short-horizon reversal "
              "documented by Jegadeesh 1990 / Lehmann 1990). You buy fear inside an up-trend.",
          entry_txt="Next session open (or limit 0.25 ATR better).",
          exit_txt="Stop 1.5 ATR, target 1.5 ATR, time-stop 7 sessions.",
          reasons=lambda r, d: [
              f"Price {'above' if d > 0 else 'below'} 200-day average ({_fmt(r.Close)} vs {_fmt(r.sma200)}) -> long-term trend is {'up' if d > 0 else 'down'}",
              f"12-month momentum {r.mom12 * 100:+.1f}% confirms the trend",
              f"RSI(2) = {r.rsi2:.1f} -> short-term {'oversold' if d > 0 else 'overbought'} (trigger {'<10' if d > 0 else '>90'})",
          ]),
    Setup("BREAKOUT55", "55-day breakout + momentum", "Swing/position 1-8w", False, 2.0, 4.0, 40, (1, -1),
          cond=lambda f, d: ((f.Close > f.hh55) & (f.mom12 > 0) & (f.mom3 > 0)) if d > 0
          else ((f.Close < f.ll55) & (f.mom12 < 0) & (f.mom3 < 0)),
          why="Time-series momentum / trend following (Moskowitz-Ooi-Pedersen 2012; Hurst-Ooi-Pedersen 2017, "
              "a century of evidence). Low hit rate, big winners.",
          entry_txt="Next session open.",
          exit_txt="Stop 2 ATR, target 4 ATR, time-stop 40 sessions.",
          reasons=lambda r, d: [
              f"Close {_fmt(r.Close)} {'broke above the prior 55-day high' if d > 0 else 'broke below the prior 55-day low'} ({_fmt(r.hh55 if d > 0 else r.ll55)})",
              f"Momentum aligned: 3m {r.mom3 * 100:+.1f}%, 12m {r.mom12 * 100:+.1f}%",
          ]),
    Setup("SQUEEZE", "Volatility squeeze breakout", "Swing 1-3w", False, 1.5, 3.0, 15, (1, -1),
          cond=lambda f, d: (f.squeeze_recent & (f.Close > f.hh20)) if d > 0
          else (f.squeeze_recent & (f.Close < f.ll20)),
          why="Volatility clusters and mean-reverts (Engle ARCH 1982); compressed ranges tend to be followed "
              "by range expansion, so a breakout from a squeeze has a better payoff profile.",
          entry_txt="Next session open.",
          exit_txt="Stop 1.5 ATR, target 3 ATR, time-stop 15 sessions.",
          reasons=lambda r, d: [
              f"Bollinger width in the bottom 15% of the last year within the past 5 sessions (now {r.bbw_pct * 100:.0f}th pct)",
              f"Close {_fmt(r.Close)} {'above prior 20-day high ' + _fmt(r.hh20) if d > 0 else 'below prior 20-day low ' + _fmt(r.ll20)}",
          ]),
    Setup("IBS", "IBS reversal (close at extreme)", "Short swing 1-3d", True, 1.5, 1.0, 3, (1, -1),
          cond=lambda f, d: ((f.ibs < 0.15) & (f.Close > f.sma200)) if d > 0
          else ((f.ibs > 0.85) & (f.Close < f.sma200)),
          why="Internal Bar Strength: closes near the day's extreme tend to revert next days, strongest in "
              "equity indices (Pagonidis 2013; documented in index futures/ETFs).",
          entry_txt="Next session open.",
          exit_txt="Stop 1.5 ATR, target 1 ATR, time-stop 3 sessions.",
          reasons=lambda r, d: [
              f"Closed in the {'bottom' if d > 0 else 'top'} of today's range (IBS {r.ibs:.2f})",
              f"Trend filter: price {'above' if d > 0 else 'below'} 200-day average",
          ]),
    Setup("REJECTION", "Failed breakout (rejection)", "Day/short swing 1-5d", True, 1.25, 2.0, 5, (1, -1),
          cond=lambda f, d: ((f.Low < f.ll20) & (f.ibs > 0.7) & (f.Close > f.ll20)) if d > 0
          else ((f.High > f.hh20) & (f.ibs < 0.3) & (f.Close < f.hh20)),
          why="Liquidity grab / stop-run: price trades through a 20-day extreme, finds no follow-through and "
              "closes back inside the range -> trapped breakout traders fuel the reversal.",
          entry_txt="Next session open.",
          exit_txt="Stop 1.25 ATR (beyond the rejection wick), target 2 ATR, time-stop 5 sessions.",
          reasons=lambda r, d: [
              f"Swept the prior 20-day {'low ' + _fmt(r.ll20) if d > 0 else 'high ' + _fmt(r.hh20)} intraday",
              f"...but closed back inside the range with IBS {r.ibs:.2f} (strong rejection)",
          ]),
    Setup("TOM", "Turn-of-month flow", "Calendar 4d", False, 2.0, 10.0, 4, (1,),
          cond=None,
          why="Turn-of-the-month effect (Lakonishok-Smidt 1988; McConnell-Xu 2008): pension/payroll flows "
              "concentrate equity returns in the last day + first 3 days of the month.",
          entry_txt="Open of the last trading day of the month.",
          exit_txt="Exit at the close of the 3rd trading day of the new month; catastrophe stop 2 ATR.",
          reasons=lambda r, d: ["Tomorrow is the last trading day of the month (turn-of-month window opens)"],
          only_classes=("Equity index", "Crypto")),
]


# ----------------------------------------------------------------------------- engine
def signal_series(setup: Setup, f: pd.DataFrame, d: int, crypto: bool) -> pd.Series:
    if setup.key == "TOM":
        return _tom_flag(f, crypto)
    return setup.cond(f, d).fillna(False).astype(bool)


def triple_barrier(f: pd.DataFrame, sig: pd.Series, d: int, setup: Setup, ohlc: bool,
                   cost_bps: float) -> pd.DataFrame:
    O, H, L, Cl = (f[c].values for c in ["Open", "High", "Low", "Close"])
    A = f["atr"].values
    idx = f.index
    s = sig.values
    n = len(f)
    trades = []
    i = 0
    while i < n - 1:
        if not s[i] or not np.isfinite(A[i]) or A[i] <= 0:
            i += 1
            continue
        e_bar = i + 1 if ohlc else i
        E = O[e_bar] if ohlc else Cl[i]
        sld, tpd = setup.sl_atr * A[i], setup.tp_atr * A[i]
        stop, tgt = E - d * sld, E + d * tpd
        exit_px, exit_bar, outcome = None, None, None
        last = min(i + setup.max_hold, n - 1)
        if last <= i:
            break
        for k in range(e_bar, last + 1):
            if ohlc:
                if k > e_bar and (d * (O[k] - stop) <= 0):
                    exit_px, outcome = O[k], "stop"
                elif k > e_bar and (d * (O[k] - tgt) >= 0):
                    exit_px, outcome = O[k], "target"
                elif (d > 0 and L[k] <= stop) or (d < 0 and H[k] >= stop):
                    exit_px, outcome = stop, "stop"
                elif (d > 0 and H[k] >= tgt) or (d < 0 and L[k] <= tgt):
                    exit_px, outcome = tgt, "target"
            else:
                if k == e_bar:
                    continue
                if d * (Cl[k] - stop) <= 0:
                    exit_px, outcome = Cl[k], "stop"
                elif d * (Cl[k] - tgt) >= 0:
                    exit_px, outcome = Cl[k], "target"
            if exit_px is not None:
                exit_bar = k
                break
        if exit_px is None:
            if i + setup.max_hold > n - 1:      # trade still open at end of data -> not counted
                break
            exit_bar, exit_px, outcome = last, Cl[last], "time"
        cost_r = cost_bps * 1e-4 * E / sld
        R = d * (exit_px - E) / sld - cost_r
        trades.append(dict(date=idx[i], exit_date=idx[exit_bar], R=R, outcome=outcome,
                           bars=exit_bar - i, vol_pct=f["vol_pct"].iloc[i]))
        i = exit_bar + 1 if exit_bar > i else i + 1
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, setup: Setup, cost_bps_hint: float = 0.0) -> dict:
    out = dict(n=int(len(tr)))
    if len(tr) < 5:
        out.update(valid=False)
        return out
    R = tr["R"].values
    n = len(R)
    wins = int((R > 0).sum())
    mean, sd = R.mean(), R.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    p = 1 - st.t.cdf(t, df=n - 1)
    rng = np.random.default_rng(7)
    boot = rng.choice(R, size=(2000, n), replace=True).mean(axis=1)
    lo5 = float(np.percentile(boot, 5))
    # Wilson 95% CI on hit rate
    z = 1.96
    ph = wins / n
    den = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    # Bayesian shrinkage toward "no edge"
    p_be = setup.sl_atr / (setup.sl_atr + setup.tp_atr)       # break-even hit rate (pure barrier trade)
    k = C.PRIOR_STRENGTH
    p_post = (wins + k * p_be) / (n + k)
    e_post = mean * n / (n + k)
    # out-of-sample (chronological)
    cut = int(n * (1 - C.OOS_FRACTION))
    is_mean, oos_mean = float(R[:cut].mean()), float(R[cut:].mean()) if n - cut > 0 else float("nan")
    recent = tr[tr["date"] >= tr["date"].iloc[-1] - pd.Timedelta(days=5 * 365)]["R"]
    gp, gl = R[R > 0].sum(), -R[R < 0].sum()
    eq = np.cumsum(R)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    out.update(valid=True, hit=ph, hit_lo=centre - half, hit_hi=centre + half, p_post=p_post, p_be=p_be,
               meanR=float(mean), e_post=float(e_post), sdR=float(sd), t=float(t), pval=float(p), boot_lo5=lo5,
               is_mean=is_mean, oos_mean=oos_mean, recent_mean=float(recent.mean()) if len(recent) else float("nan"),
               recent_n=int(len(recent)), pf=float(gp / gl) if gl > 0 else None, maxdd=dd,
               avg_win=float(R[R > 0].mean()) if wins else 0.0, avg_loss=float(R[R <= 0].mean()) if wins < n else 0.0,
               avg_bars=float(tr["bars"].mean()),
               per_year=float(n / max((tr["date"].iloc[-1] - tr["date"].iloc[0]).days / 365.25, 1)))
    return out


def benjamini_hochberg(pvals: list[float], q: float) -> list[bool]:
    p = np.array(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    res = np.zeros(m, dtype=bool)
    res[order[:kmax]] = True
    return res.tolist()


def grade(s: dict) -> str:
    """A = statistically robust; B = promising; C = weak/unproven; X = no edge (skip)."""
    if not s.get("valid"):
        return "X"
    oos_ok = s["oos_mean"] > 0
    if s["n"] >= 40 and s.get("fdr_pass") and oos_ok and s["e_post"] >= 0.08 and s["boot_lo5"] > 0:
        return "A"
    if s["n"] >= 30 and s["pval"] < 0.05 and oos_ok and s["e_post"] >= 0.05:
        return "B"
    if s["e_post"] > 0 and s["n"] >= 20:
        return "C"
    return "X"


def kelly_risk(p: float, b: float) -> float:
    """Quarter-Kelly risk % (capped). b = average win / average loss in R."""
    if b <= 0:
        return 0.0
    f = p - (1 - p) / b
    return float(max(0.0, min(C.MAX_RISK_PCT, 100 * C.KELLY_FRACTION * f)))

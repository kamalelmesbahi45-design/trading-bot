"""Lightweight hypothesis tester for daily position-series strategies.

For each hypothesis:
    1. Run generator(features, bars) -> position series in {-1, 0, +1}.
    2. Compute trade-by-trade returns from close-to-close, lagged by 1 day so
       today's signal trades tomorrow's open (look-ahead-free).
    3. Subtract per-transaction cost: cost_pct = (spread_pts * 0.01 + slip_pts * 0.01) / price.
       Approximation: gold spread + slippage ~ 28c on a ~$2000 price = ~14 bps per side.
    4. Build equity curve, compute stats with deflated Sharpe (n_trials = number
       of hypotheses).
    5. Walk-forward by year: 4y IS, 1y OOS, step 1y. (With ~6.7y of data we get
       ~3 OOS folds.)

The point: a true edge clears the bar AFTER multi-test correction AND in OOS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.hypotheses import HYPOTHESES, Hypothesis

COST_PER_SIDE_BPS = 14.0   # ~28c gold spread + slippage on ~$2000 -> 14bps per trade side


@dataclass
class TestResult:
    id: str
    name: str
    n_trades: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    hit_rate: float
    profit_factor: float
    deflated_sharpe: float
    final_equity: float
    equity_curve: pd.Series
    oos_sharpe: float | None
    oos_curve: pd.Series | None


def _deflated_sharpe(sharpe_obs: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float:
    if n_obs < 3 or n_trials < 1:
        return float("nan")
    if n_trials <= 1:
        e_max = 0.0
    else:
        # Bailey & Lopez de Prado approx
        emc = 0.5772156649015329
        z1 = _z_inv(1.0 - 1.0 / n_trials)
        z2 = _z_inv(1.0 - 1.0 / (n_trials * math.e))
        inner = (1.0 - emc) * z1 + emc * z2
        if inner <= 0:
            return float("nan")
        e_max = math.sqrt(inner)
    var = (1.0 - skew * sharpe_obs + 0.25 * (kurt - 1.0) * sharpe_obs * sharpe_obs) / max(n_obs - 1, 1)
    if var <= 0:
        return float("nan")
    z = (sharpe_obs - e_max) / math.sqrt(var)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _z_inv(p: float) -> float:
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def _stats_from_returns(rets: pd.Series, periods_per_year: float = 252.0,
                         starting: float = 10_000.0):
    if rets.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    # Floor per-bar returns at -0.99 so blow-ups can't produce negative equity
    # (artificial -- in reality a stop or risk gate would intervene).
    rets = rets.clip(lower=-0.99)
    eq = starting * (1 + rets).cumprod()
    end_eq = float(eq.iloc[-1])
    if end_eq > 0:
        cagr = (end_eq / starting) ** (periods_per_year / max(len(rets), 1)) - 1.0
    else:
        cagr = -1.0   # zeroed out
    mu = rets.mean()
    sd = rets.std(ddof=0)
    sh = (mu / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    dsd = rets.clip(upper=0).std(ddof=0)
    so = (mu / dsd * np.sqrt(periods_per_year)) if dsd > 0 else 0.0
    peak = eq.cummax()
    mdd = float((eq / peak - 1.0).min())
    return float(cagr), float(sh), float(so), mdd, eq


def _run_one(h: Hypothesis, feats: pd.DataFrame, bars: pd.DataFrame,
              n_trials: int, oos_start: pd.Timestamp | None = None) -> TestResult:
    pos = h.generator(feats, bars).fillna(0.0)
    # Trade at next close: today's signal earns tomorrow's close-to-close return.
    fwd_ret = bars["close"].pct_change().shift(-1).fillna(0.0)
    trade_ret = pos * fwd_ret
    # Cost on position changes: |pos[i] - pos[i-1]| * (cost_per_side)
    pos_change = pos.diff().abs().fillna(pos.abs())
    cost = pos_change * (COST_PER_SIDE_BPS / 10_000.0)
    net = (trade_ret - cost).dropna()

    cagr, sharpe, sortino, mdd, eq = _stats_from_returns(net)
    pos_runs = (pos.diff() != 0).sum()    # rough trade count
    wins = (net > 0).sum()
    losses = (net < 0).sum()
    hit = wins / max(wins + losses, 1)
    gross_win = net[net > 0].sum()
    gross_loss = abs(net[net < 0].sum())
    pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    sk = float(net.skew()) if len(net) > 3 else 0.0
    ku = float(net.kurt()) if len(net) > 3 else 3.0
    dsh = _deflated_sharpe(sharpe, n_trials, len(net), sk, ku)

    # Walk-forward OOS by year
    oos_curve = None
    oos_sharpe = None
    if oos_start is not None:
        oos_mask = net.index >= oos_start
        oos_rets = net[oos_mask]
        if len(oos_rets) > 30:
            _, oos_sharpe, _, _, oos_eq = _stats_from_returns(oos_rets)
            oos_curve = oos_eq

    calmar = float(cagr / abs(mdd)) if mdd != 0 else 0.0
    return TestResult(
        id=h.id, name=h.name, n_trades=int(pos_runs),
        cagr=cagr, sharpe=sharpe, sortino=sortino, max_dd=mdd, calmar=calmar,
        hit_rate=float(hit), profit_factor=pf, deflated_sharpe=dsh,
        final_equity=float(eq.iloc[-1] if len(eq) else 10_000.0),
        equity_curve=eq if isinstance(eq, pd.Series) else pd.Series(dtype=float),
        oos_sharpe=oos_sharpe, oos_curve=oos_curve,
    )


def run_all(feats: pd.DataFrame, bars: pd.DataFrame, oos_start: pd.Timestamp | None = None,
            hypotheses: list[Hypothesis] | None = None) -> list[TestResult]:
    hs = hypotheses or HYPOTHESES
    out: list[TestResult] = []
    for h in hs:
        try:
            res = _run_one(h, feats, bars, n_trials=len(hs), oos_start=oos_start)
        except Exception as e:
            print(f"  WARN {h.id} failed: {e}")
            continue
        out.append(res)
    return out


def summarise(results: list[TestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "id": r.id, "name": r.name, "n_trades": r.n_trades,
            "final$": round(r.final_equity, 0),
            "CAGR": round(r.cagr, 4), "Sharpe": round(r.sharpe, 3),
            "Sortino": round(r.sortino, 3), "MaxDD": round(r.max_dd, 4),
            "Calmar": round(r.calmar, 3), "HitRate": round(r.hit_rate, 3),
            "PF": round(r.profit_factor, 3) if r.profit_factor != float("inf") else 999.0,
            "DefSh": round(r.deflated_sharpe, 3),
            "OOS_Sh": round(r.oos_sharpe, 3) if r.oos_sharpe is not None else None,
        })
    return pd.DataFrame(rows).sort_values("DefSh", ascending=False)

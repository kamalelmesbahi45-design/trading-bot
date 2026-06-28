"""Performance metrics. All produced from an equity curve + trade log.

Conventions:
    * equity_curve: pd.Series indexed by ts (any frequency), values in USD.
    * Returns are computed as simple pct_change on the curve.
    * Annualisation factor is inferred from the median bar spacing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceStats:
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    avg_R: float
    expectancy_R: float
    profit_factor: float
    n_trades: int
    deflated_sharpe: float | None = None


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_seconds = float(np.median(np.diff(index.asi8)) / 1e9)
    if median_seconds <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median_seconds


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())  # type: ignore[no-any-return]


def cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty or equity_curve.iloc[0] <= 0:
        return 0.0
    start_eq = float(equity_curve.iloc[0])
    end_eq = float(equity_curve.iloc[-1])
    total_seconds = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
    if total_seconds <= 0:
        return 0.0
    years = total_seconds / (365.25 * 24 * 3600)
    if years <= 0 or end_eq <= 0:
        return 0.0
    return float((end_eq / start_eq) ** (1.0 / years) - 1.0)


def _daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Resample equity to daily closes before computing returns.

    Mark-to-market equity on intraday bars has many near-flat bars between
    trades; computing Sharpe directly on those crushes the return std and
    produces nonsensical inflated ratios. Industry standard is daily.
    """
    if equity_curve.empty:
        return equity_curve
    daily = equity_curve.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def sharpe(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    rets = _daily_returns(equity_curve)
    if len(rets) < 2 or rets.std(ddof=0) == 0:
        return 0.0
    excess = rets - risk_free_rate / 252.0
    return float(excess.mean() / excess.std(ddof=0) * math.sqrt(252.0))


def sortino(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    rets = _daily_returns(equity_curve)
    if len(rets) < 2:
        return 0.0
    downside = rets.clip(upper=0.0)
    ds_std = downside.std(ddof=0)
    if ds_std == 0:
        return 0.0
    excess = rets.mean() - risk_free_rate / 252.0
    return float(excess / ds_std * math.sqrt(252.0))


def deflated_sharpe(sharpe_obs: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float:
    """Bailey & Lopez de Prado deflated Sharpe ratio.

    Corrects observed Sharpe for multiple-testing inflation and non-normality.
    Returns the probability that the true Sharpe exceeds zero (a p-value-style
    score in [0, 1]). Returns NaN if inputs are degenerate.
    """
    if n_obs < 3 or n_trials < 1:
        return float("nan")
    # With a single observed strategy there's no multiple-testing inflation; e_max=0.
    if n_trials == 1:
        e_max = 0.0
    else:
        emc = 0.5772156649015329  # Euler-Mascheroni
        n_inv = 1.0 / n_trials
        a = _z_inv(1.0 - n_inv)
        b = _z_inv(1.0 - n_inv * math.e ** -1)
        inner = (1.0 - emc) * a + emc * b
        if inner <= 0:
            return float("nan")
        e_max = math.sqrt(inner)
    # Variance of the Sharpe estimator (non-normal)
    var = (1.0 - skew * sharpe_obs + 0.25 * (kurt - 1.0) * sharpe_obs * sharpe_obs) / max(n_obs - 1, 1)
    if var <= 0:
        return float("nan")
    z = (sharpe_obs - e_max) / math.sqrt(var)
    return float(_norm_cdf(z))


def _z_inv(p: float) -> float:
    """Inverse standard-normal CDF via Beasley-Springer-Moro (good enough for DSR)."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    # Rational approximation
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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_stats(equity_curve: pd.Series, trades: pd.DataFrame, n_trials: int = 1) -> PerformanceStats:
    """Build the full PerformanceStats summary."""
    if equity_curve.empty:
        return PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    cg = cagr(equity_curve)
    sh = sharpe(equity_curve)
    so = sortino(equity_curve)
    mdd = max_drawdown(equity_curve)
    cal = (cg / abs(mdd)) if mdd != 0 else 0.0

    if trades is None or trades.empty:
        win_rate = 0.0
        avg_r = 0.0
        expectancy = 0.0
        profit_factor = 0.0
        n_trades = 0
    else:
        pnl = trades["pnl"].astype(float)
        r = trades["r_multiple"].astype(float) if "r_multiple" in trades.columns else pd.Series([0.0] * len(trades))
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0
        avg_r = float(r.mean())
        expectancy = avg_r
        gross_loss = abs(float(losses.sum()))
        profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf") if wins.sum() > 0 else 0.0
        n_trades = len(pnl)

    # Deflated Sharpe (rough): use observed Sharpe + n_obs from curve
    rets = equity_curve.pct_change().dropna()
    if len(rets) >= 5:
        sk = float(rets.skew()) if not pd.isna(rets.skew()) else 0.0
        ku = float(rets.kurt()) if not pd.isna(rets.kurt()) else 3.0
        dsr = deflated_sharpe(sh, n_trials, len(rets), sk, ku)
    else:
        dsr = None

    return PerformanceStats(
        cagr=cg, sharpe=sh, sortino=so,
        max_drawdown=mdd, calmar=cal,
        win_rate=win_rate, avg_R=avg_r, expectancy_R=expectancy,
        profit_factor=profit_factor, n_trades=n_trades,
        deflated_sharpe=dsr,
    )

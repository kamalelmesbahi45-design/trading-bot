"""Risk, sizing and portfolio construction for the macro agent.

Every trade idea passes through:

    1. ATR-based stop & R-multiple target (day vs swing have different ATR
       multiples and time horizons).
    2. Position sizing -- the per-trade ``risk_dollars = equity * risk_pct``
       is divided by the per-share stop distance to get the unit count, then
       capped by a max-leverage rule (gross < max_gross) and a max-weight
       rule (no single name above max_weight_pct of equity).
    3. Cluster exposure cap -- total |risk| in any one correlation cluster
       (e.g. "metals" = GLD+SLV+GDX) is capped, so we don't unknowingly take
       three near-identical bets.
    4. Portfolio correlation check -- if the average pairwise correlation of
       the resulting longs (and separately the shorts) is above a threshold,
       the lower-conviction trades are dropped to restore diversification.

All sizing is expressed in *risk dollars* (the dollar amount you lose if the
stop is hit). That makes the math identical across asset classes; converting
to share / contract count is a per-asset trivial step at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xauusd_bot.agent.signals import atr, realised_vol


@dataclass
class TradeTicket:
    ticker: str
    label: str
    asset_class: str
    cluster: str
    horizon: str               # "day" | "swing"
    strategy: str              # day_momentum / day_breakout / day_mean_revert / swing_trend
    side: str                  # LONG / SHORT
    entry: float
    stop: float
    target: float              # 1R, 2R targets are derived; this is the take-profit (usually 2R)
    atr_value: float
    risk_dollars: float        # $ amount risked if stop is hit
    units: float               # shares / units of the proxy ETF (or contracts for crypto/futures)
    notional: float            # |units * entry|
    weight_pct: float          # signed notional as % of equity
    r_multiple: float          # target / risk ratio
    conviction: float          # in [0, 1]
    rationale: str

    def to_line(self) -> str:
        return (
            f"{self.ticker:4} {self.label:12} {self.horizon:5} {self.strategy:18} "
            f"{self.side:6} {self.entry:8.2f} {self.stop:8.2f} {self.target:8.2f} "
            f"{self.atr_value:6.2f} {self.units:9.2f} ${self.risk_dollars:7.0f} "
            f"{self.weight_pct:+5.1%} {self.r_multiple:3.1f} {self.conviction:5.2f}"
        )


@dataclass
class RiskSettings:
    equity: float = 100_000.0
    risk_pct_per_trade: float = 0.005       # 0.5% per trade
    max_gross_leverage: float = 2.0
    max_weight_pct: float = 0.25            # max 25% notional in any single name
    max_cluster_risk_pct: float = 0.015     # cap cluster risk to 3x single-trade risk
    max_avg_correlation: float = 0.65       # within longs / within shorts
    atr_stop_mult_day: float = 1.5
    atr_stop_mult_swing: float = 2.5
    rr_target_day: float = 2.0              # 2R take-profit for day
    rr_target_swing: float = 3.0            # 3R for swing


def _round(x: float, q: float = 0.01) -> float:
    if not np.isfinite(x):
        return float("nan")
    return round(x / q) * q


def build_ticket(
    ticker: str,
    label: str,
    asset_class: str,
    cluster: str,
    ohlc: pd.DataFrame,
    side: str,
    horizon: str,
    strategy: str,
    conviction: float,
    settings: RiskSettings,
    rationale: str = "",
) -> TradeTicket | None:
    """Construct a TradeTicket from OHLC + side + horizon. Returns None if the
    setup can't be sized cleanly (e.g. ATR not available, stop too tight)."""
    close = ohlc["close"].dropna()
    if len(close) < 30:
        return None
    entry = float(close.iloc[-1])
    atr_series = atr(ohlc)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    mult = settings.atr_stop_mult_day if horizon == "day" else settings.atr_stop_mult_swing
    rr = settings.rr_target_day if horizon == "day" else settings.rr_target_swing

    if side == "LONG":
        stop = entry - mult * atr_value
        target = entry + mult * atr_value * rr
    elif side == "SHORT":
        stop = entry + mult * atr_value
        target = entry - mult * atr_value * rr
    else:
        return None

    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return None

    risk_dollars = settings.equity * settings.risk_pct_per_trade
    units = risk_dollars / risk_per_unit
    if side == "SHORT":
        units = -units
    notional = abs(units) * entry
    max_notional = settings.equity * settings.max_weight_pct
    if notional > max_notional:
        scale = max_notional / notional
        units *= scale
        notional *= scale
        risk_dollars *= scale

    weight_pct = (units * entry) / settings.equity
    r_multiple = rr

    return TradeTicket(
        ticker=ticker,
        label=label,
        asset_class=asset_class,
        cluster=cluster,
        horizon=horizon,
        strategy=strategy,
        side=side,
        entry=_round(entry),
        stop=_round(stop),
        target=_round(target),
        atr_value=_round(atr_value, 0.001),
        risk_dollars=_round(risk_dollars, 1.0),
        units=_round(units, 0.0001),
        notional=_round(notional, 1.0),
        weight_pct=round(weight_pct, 4),
        r_multiple=round(r_multiple, 2),
        conviction=round(float(np.clip(conviction, 0.0, 1.0)), 3),
        rationale=rationale,
    )


@dataclass
class PortfolioRisk:
    gross: float
    net: float
    by_cluster_risk: dict[str, float]
    by_class_notional: dict[str, float]
    avg_long_corr: float
    avg_short_corr: float
    n_long: int
    n_short: int
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _avg_pairwise_corr(corr: pd.DataFrame, tickers: list[str]) -> float:
    tickers = [t for t in tickers if t in corr.columns]
    if len(tickers) < 2:
        return 0.0
    sub = corr.loc[tickers, tickers]
    n = len(tickers)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = sub.to_numpy()[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def apply_portfolio_rules(
    tickets: list[TradeTicket],
    returns_panel: pd.DataFrame,
    settings: RiskSettings,
) -> tuple[list[TradeTicket], PortfolioRisk]:
    """Apply cluster caps and correlation pruning to a list of candidate tickets.

    Tickets are sorted by conviction descending; we admit them one by one as
    long as each new ticket (a) keeps that cluster's total risk under the cap
    and (b) keeps the average pairwise correlation within its side under the
    cap. Cheap, greedy, and easy to reason about.
    """
    if not tickets:
        return [], PortfolioRisk(0, 0, {}, {}, 0, 0, 0, 0)

    corr = pd.DataFrame()
    if not returns_panel.empty:
        rec = returns_panel.tail(126).dropna(how="all", axis=1)
        if rec.shape[1] >= 2:
            corr = rec.corr().fillna(0.0)

    sorted_tk = sorted(tickets, key=lambda t: -t.conviction)
    admitted: list[TradeTicket] = []
    dropped: list[str] = []
    cluster_risk: dict[str, float] = {}
    cluster_cap = settings.equity * settings.max_cluster_risk_pct

    for t in sorted_tk:
        # cluster cap
        next_cluster_risk = cluster_risk.get(t.cluster, 0.0) + t.risk_dollars
        if next_cluster_risk > cluster_cap * 1.0001:
            dropped.append(f"{t.ticker}:cluster_cap[{t.cluster}]")
            continue
        # correlation gate within side
        same_side = [a for a in admitted if a.side == t.side]
        tickers = [a.ticker for a in same_side] + [t.ticker]
        avg_c = _avg_pairwise_corr(corr, tickers) if not corr.empty else 0.0
        if len(same_side) >= 2 and avg_c > settings.max_avg_correlation:
            dropped.append(f"{t.ticker}:corr[{avg_c:.2f}]")
            continue
        admitted.append(t)
        cluster_risk[t.cluster] = next_cluster_risk

    # gross leverage cap (notional / equity)
    total_notional = sum(a.notional for a in admitted)
    if total_notional > settings.equity * settings.max_gross_leverage:
        scale = (settings.equity * settings.max_gross_leverage) / total_notional
        for a in admitted:
            a.units = round(a.units * scale, 4)
            a.notional = round(a.notional * scale, 2)
            a.risk_dollars = round(a.risk_dollars * scale, 2)
            a.weight_pct = round(a.weight_pct * scale, 4)

    gross = sum(a.notional for a in admitted) / settings.equity
    net = sum((a.units * a.entry) for a in admitted) / settings.equity
    by_class: dict[str, float] = {}
    for a in admitted:
        by_class[a.asset_class] = by_class.get(a.asset_class, 0.0) + a.notional / settings.equity

    longs = [a.ticker for a in admitted if a.side == "LONG"]
    shorts = [a.ticker for a in admitted if a.side == "SHORT"]
    avg_long_corr = _avg_pairwise_corr(corr, longs) if not corr.empty else 0.0
    avg_short_corr = _avg_pairwise_corr(corr, shorts) if not corr.empty else 0.0

    warnings: list[str] = []
    if gross > settings.max_gross_leverage * 0.95:
        warnings.append(f"gross leverage at cap ({gross:.2f}x)")
    if any(v > settings.max_cluster_risk_pct * 0.95 * settings.equity / settings.equity * 1
           for v in [r / settings.equity for r in cluster_risk.values()]):
        warnings.append("cluster risk at cap (see by_cluster_risk)")

    return admitted, PortfolioRisk(
        gross=round(gross, 3),
        net=round(net, 3),
        by_cluster_risk={k: round(v, 2) for k, v in cluster_risk.items()},
        by_class_notional={k: round(v, 3) for k, v in by_class.items()},
        avg_long_corr=round(avg_long_corr, 3),
        avg_short_corr=round(avg_short_corr, 3),
        n_long=len(longs),
        n_short=len(shorts),
        dropped=dropped,
        warnings=warnings,
    )


def asset_vol_summary(close: pd.Series) -> dict[str, float]:
    """Quick per-asset vol & range read for the report header."""
    if close.dropna().empty:
        return {"ann_vol": float("nan"), "drawdown_3m": float("nan")}
    v = realised_vol(close)
    recent = close.dropna().iloc[-63:]
    dd = float("nan") if recent.empty else float((recent / recent.cummax() - 1.0).min())
    return {"ann_vol": v, "drawdown_3m": dd}

"""Daily multi-asset briefing: "what to trade today".

Given a close panel up to the latest available date, produce per-asset:
    * direction  : LONG / SHORT / FLAT (blended trend sign)
    * strength   : |blended signal| in [0,1]
    * trend_3/6/12m : the three component returns
    * vol_annual : current EWMA annualised vol
    * target_weight : inverse-vol, vol-targeted book weight (what the system holds)
    * notional_pct  : |target_weight| as a share of gross book

Plus a portfolio summary: gross/net exposure, number of longs/shorts, the
top conviction names, and a plain-language regime read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xauusd_bot.research.panel import UNIVERSE_BY_TICKER, daily_returns
from xauusd_bot.research.signals import (
    ewma_vol,
    scale_to_portfolio_vol,
    tsmom_signal,
    vol_target_weights,
)


@dataclass
class AssetSignal:
    ticker: str
    label: str
    asset_class: str
    direction: str
    strength: float
    trend_3m: float
    trend_6m: float
    trend_12m: float
    vol_annual: float
    target_weight: float


@dataclass
class Briefing:
    as_of: pd.Timestamp
    signals: list[AssetSignal]
    gross_exposure: float
    net_exposure: float
    n_long: int
    n_short: int

    def to_text(self, top_n: int = 8) -> str:
        lines = [f"=== Daily briefing  {self.as_of.date()} ==="]
        lines.append(f"gross {self.gross_exposure:.0%}  net {self.net_exposure:+.0%}  "
                     f"longs {self.n_long}  shorts {self.n_short}\n")
        ranked = sorted(self.signals, key=lambda s: abs(s.target_weight), reverse=True)
        lines.append(f"{'asset':14}{'dir':6}{'str':>5}{'3m':>8}{'6m':>8}{'12m':>8}{'vol':>7}{'wgt':>8}")
        for s in ranked[:top_n]:
            if abs(s.target_weight) < 1e-4:
                continue
            lines.append(
                f"{s.label:14}{s.direction:6}{s.strength:4.0%} "
                f"{s.trend_3m:+7.1%}{s.trend_6m:+7.1%}{s.trend_12m:+7.1%}"
                f"{s.vol_annual:6.0%}{s.target_weight:+8.2f}"
            )
        flat = [s.label for s in self.signals if s.direction == "FLAT"]
        if flat:
            lines.append(f"\nflat / no-trade: {', '.join(flat)}")
        return "\n".join(lines)


def _dir(x: float) -> str:
    if x > 0.15:
        return "LONG"
    if x < -0.15:
        return "SHORT"
    return "FLAT"


def generate_briefing(panel: pd.DataFrame, target_vol: float = 0.10) -> Briefing:
    """Build the briefing as of the last row of `panel`."""
    returns = daily_returns(panel)
    sig = tsmom_signal(panel, lookbacks=(63, 126, 252))
    vol = ewma_vol(returns)
    raw_w = vol_target_weights(sig, vol, per_asset_vol_target=target_vol / 4.0)
    weights = scale_to_portfolio_vol(raw_w, returns, target_annual_vol=target_vol)

    as_of = panel.index[-1]
    last_sig = sig.iloc[-1]
    last_w = weights.iloc[-1]
    last_vol = vol.iloc[-1]

    def trail(lb: int, tk: str) -> float:
        if len(panel) <= lb or pd.isna(panel[tk].iloc[-1 - lb]):
            return float("nan")
        return float(panel[tk].iloc[-1] / panel[tk].iloc[-1 - lb] - 1.0)

    signals: list[AssetSignal] = []
    for tk in panel.columns:
        spec = UNIVERSE_BY_TICKER.get(tk)
        s_val = float(last_sig.get(tk, 0.0)) if not pd.isna(last_sig.get(tk, np.nan)) else 0.0
        w_val = float(last_w.get(tk, 0.0))
        signals.append(AssetSignal(
            ticker=tk,
            label=spec.label if spec else tk,
            asset_class=spec.asset_class if spec else "?",
            direction=_dir(s_val),
            strength=abs(s_val),
            trend_3m=trail(63, tk),
            trend_6m=trail(126, tk),
            trend_12m=trail(252, tk),
            vol_annual=float(last_vol.get(tk, float("nan"))),
            target_weight=w_val,
        ))

    gross = float(last_w.abs().sum())
    net = float(last_w.sum())
    n_long = int((last_w > 1e-4).sum())
    n_short = int((last_w < -1e-4).sum())
    return Briefing(as_of=as_of, signals=signals, gross_exposure=gross,
                    net_exposure=net, n_long=n_long, n_short=n_short)

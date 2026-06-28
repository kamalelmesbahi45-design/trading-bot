"""Walk-forward validation.

v1 scope: rolling out-of-sample VALIDATION (not optimisation). Splits bars into
overlapping (IS, OOS) windows and runs the configured engine on each OOS chunk
with the same parameters. Aggregates an OOS equity curve and per-fold stats.

A parameter-search variant (grid- or Bayes-optimise IS, evaluate OOS) will land
in a follow-up: needs a search-space spec separated from BotConfig.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from xauusd_bot.backtest.engine import BacktestEngine
from xauusd_bot.config import BotConfig


@dataclass
class WFOResult:
    oos_equity: pd.Series
    fold_results: list[dict[str, float | str]]
    param_history: pd.DataFrame
    stability_score: float


def _slice_bars(bars: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    mask = (bars.index >= pd.Timestamp(start)) & (bars.index < pd.Timestamp(end))
    return bars.loc[mask]


class WalkForward:
    def __init__(self, cfg: BotConfig, is_years: float = 2.0,
                 oos_months: int = 6, step_months: int = 6) -> None:
        self.cfg = cfg
        self.is_years = is_years
        self.oos_months = oos_months
        self.step_months = step_months

    def run(self, bars: pd.DataFrame, start: datetime, end: datetime) -> WFOResult:
        if bars.empty:
            return WFOResult(pd.Series(dtype=float), [], pd.DataFrame(), 0.0)

        oos_pieces: list[pd.Series] = []
        fold_results: list[dict[str, float | str]] = []

        is_span = timedelta(days=int(self.is_years * 365.25))
        oos_span = timedelta(days=int(self.oos_months * 30.5))
        step = timedelta(days=int(self.step_months * 30.5))

        cur_is_start = pd.Timestamp(start)
        i = 0
        while cur_is_start + is_span + oos_span <= pd.Timestamp(end):
            is_end = cur_is_start + is_span
            oos_end = is_end + oos_span
            oos_bars = _slice_bars(bars, is_end.to_pydatetime(), oos_end.to_pydatetime())
            if not oos_bars.empty:
                res = BacktestEngine(self.cfg).run(oos_bars)
                fold_results.append({
                    "fold": i,
                    "is_start": str(cur_is_start),
                    "oos_start": str(is_end),
                    "oos_end": str(oos_end),
                    "final_equity": float(res.metadata.get("final_equity", 0.0)),
                    "n_trades": float(res.metadata.get("n_trades", 0)),
                })
                if not res.equity_curve.empty:
                    oos_pieces.append(res.equity_curve)
            cur_is_start = cur_is_start + step
            i += 1

        if not oos_pieces:
            return WFOResult(pd.Series(dtype=float), fold_results, pd.DataFrame(), 0.0)

        oos_curve = pd.concat(oos_pieces).sort_index()
        oos_curve = oos_curve[~oos_curve.index.duplicated(keep="last")]

        # Stability: 1 - (std of fold final equities / mean) clipped to [0, 1].
        finals = [float(f["final_equity"]) for f in fold_results if isinstance(f["final_equity"], (int, float))]
        if len(finals) >= 2 and sum(finals) > 0:
            cv = float(pd.Series(finals).std(ddof=0) / max(pd.Series(finals).mean(), 1e-9))
            stability = max(0.0, min(1.0, 1.0 - cv))
        else:
            stability = 0.0

        # v1: params don't vary across folds, so param_history is empty.
        return WFOResult(oos_curve, fold_results, pd.DataFrame(), stability)

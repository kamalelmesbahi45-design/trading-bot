"""Walk-forward optimisation.

Default: IS=2y, OOS=6m, step=6m -> ~17 folds over 10y.
For each fold:
    1. Grid- or Bayes-optimise strategy params on IS to maximise OOS-robust objective
       (we use a penalised Sharpe that punishes parameter sensitivity).
    2. Lock params, run on OOS, record stats + trades.
Aggregate: concatenate OOS equity, compute stats, plus parameter stability map.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from xauusd_bot.config import BotConfig


@dataclass
class WFOResult:
    oos_equity: pd.Series
    fold_results: list[dict[str, float]]
    param_history: pd.DataFrame   # rows=folds, cols=params, values=chosen
    stability_score: float        # 0..1, 1 = perfectly stable


class WalkForward:
    def __init__(self, cfg: BotConfig, is_years: float = 2.0, oos_months: int = 6, step_months: int = 6) -> None:
        self.cfg = cfg
        self.is_years = is_years
        self.oos_months = oos_months
        self.step_months = step_months

    def run(self, bars: pd.DataFrame, start: datetime, end: datetime) -> WFOResult:
        raise NotImplementedError

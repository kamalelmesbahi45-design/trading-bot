"""Event-driven backtest loop.

Pipeline per bar:
    1. emit BarEvent
    2. update open positions: trailing stops, intrabar SL/TP via fill_sim
    3. run filters (session, news, weekend, regime) -> skip if any blocks
    4. run macro overlay -> sizing multiplier
    5. run strategies in parallel -> signals
    6. allocator reconciles -> sized intents
    7. risk gates (prop-firm) -> VETO / ALLOW / KILL
    8. sizer assigns lots -> orders
    9. fill_sim executes -> fills
   10. update equity, log trade

State persisted at end of run: equity_curve, trades, fills, daily_pnl.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xauusd_bot.config import BotConfig


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    daily_pnl: pd.Series
    metadata: dict[str, str | float]


class BacktestEngine:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        bars: pd.DataFrame,
        cross_asset: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
    ) -> BacktestResult:
        raise NotImplementedError

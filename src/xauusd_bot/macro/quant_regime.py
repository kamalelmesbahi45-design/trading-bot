"""Quantitative regime score from observable variables only. Fully backtestable.

Inputs (all daily): DXY, US10Y, VIX, gold/DXY rolling beta.
Score is a sigmoid blend of:
    -DXY trend, -real yield direction, +VIX regime, +abs(gold/DXY beta) trend break.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from xauusd_bot.macro.base import MacroOverlay


class QuantRegime(MacroOverlay):
    name = "quant_regime"

    def __init__(self, features: pd.DataFrame) -> None:
        """features: daily DataFrame with columns dxy, us10y, vix, gold_dxy_beta."""
        self.features = features

    def score(self, ts: datetime) -> float:
        raise NotImplementedError

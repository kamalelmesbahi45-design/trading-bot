"""Regime-aware enable/disable of strategies.

If quant_regime says high-vol trending: boost trend, kill mean-reversion.
If low-vol ranging: opposite.
If panic/stressed (VIX > threshold): everything OFF except short trend.
"""
from __future__ import annotations

import pandas as pd


class RegimeRouter:
    def __init__(self, regime_series: pd.Series) -> None:
        self.regime_series = regime_series  # values: 'trend_up', 'trend_dn', 'range', 'panic', 'chop'

    def enabled_strategies(self, ts: pd.Timestamp) -> set[str]:
        raise NotImplementedError

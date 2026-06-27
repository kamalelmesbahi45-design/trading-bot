"""Daily PnL summary, sent at configured UTC time."""
from __future__ import annotations

from datetime import date

import pandas as pd


def build_summary(day: date, fills: pd.DataFrame, equity_curve: pd.Series) -> str:
    """Format a short text summary suitable for Telegram."""
    raise NotImplementedError

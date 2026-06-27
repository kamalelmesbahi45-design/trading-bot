"""XAUUSD-internal regime features: vol-pct, ADX trend strength, session label."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

# Default session boundaries in UTC (approximate)
LONDON_OPEN = time(7, 0)
LONDON_CLOSE = time(16, 0)
NY_OPEN = time(12, 0)
NY_CLOSE = time(21, 0)


def atr_pct(close: pd.Series, atr_series: pd.Series) -> pd.Series:
    """ATR as percent of price. Filter for dead chop / parabolic vol."""
    safe_close = close.replace(0.0, np.nan)
    return (atr_series / safe_close) * 100.0


def session_label(index: pd.DatetimeIndex) -> pd.Series:
    """Label each timestamp: 'asia' | 'london' | 'ny' | 'overlap' | 'off'.

    Overlap = both London and NY open (12:00-16:00 UTC).
    """
    utc_index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")

    times = utc_index.time
    labels: list[str] = []
    for t in times:
        london = LONDON_OPEN <= t < LONDON_CLOSE
        ny = NY_OPEN <= t < NY_CLOSE
        if london and ny:
            labels.append("overlap")
        elif london:
            labels.append("london")
        elif ny:
            labels.append("ny")
        elif t < LONDON_OPEN:
            labels.append("asia")
        else:
            labels.append("off")
    return pd.Series(labels, index=index, name="session")

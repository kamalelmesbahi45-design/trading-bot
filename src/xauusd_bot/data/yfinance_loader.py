"""Cross-asset feed via yfinance for the macro/regime layer.

Tickers we care about:
    DXY:   ^DXY (US dollar index)
    VIX:   ^VIX (equity vol regime)
    US10Y: ^TNX (10y note yield, /10 to get %)
    SPX:   ^GSPC
    OIL:   CL=F
    BTC:   BTC-USD
    GOLD:  GC=F   (sanity-check against our Dukascopy XAUUSD)
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

TICKERS: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "US10Y": "^TNX",
    "SPX": "^GSPC",
    "OIL": "CL=F",
    "BTC": "BTC-USD",
    "GOLD_FUT": "GC=F",
}


def fetch_cross_asset(
    start: datetime,
    end: datetime,
    interval: str = "1h",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch cross-asset OHLC. Returns wide DataFrame keyed by (ticker, field)."""
    raise NotImplementedError

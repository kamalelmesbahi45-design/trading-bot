"""Cross-asset features for the macro/regime layer.

Inputs: aligned multi-asset OHLC frame from data/yfinance_loader.
Outputs: feature matrix indexed by XAUUSD bar timestamp, forward-fillable.
"""
from __future__ import annotations

import pandas as pd


def dxy_trend(dxy: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """Sign of (EMA_fast - EMA_slow) on DXY. Gold is structurally inverse-DXY."""
    raise NotImplementedError


def real_yield_proxy(us10y: pd.Series, breakeven_proxy: pd.Series | None = None) -> pd.Series:
    """Real-yield proxy. If no TIPS breakeven supplied, fall back to nominal trend."""
    raise NotImplementedError


def vix_regime(vix: pd.Series, low: float = 15.0, high: float = 25.0) -> pd.Series:
    """0=calm, 1=normal, 2=stressed."""
    raise NotImplementedError


def gold_dxy_beta(gold: pd.Series, dxy: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta of gold returns to DXY returns. Flags regime breaks."""
    raise NotImplementedError

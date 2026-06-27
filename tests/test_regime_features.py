"""Regime feature tests."""
from __future__ import annotations

import pandas as pd

from xauusd_bot.features.indicators import atr
from xauusd_bot.features.regime_features import atr_pct, session_label


def test_atr_pct_basic() -> None:
    close = pd.Series([2000.0, 2010.0, 2005.0])
    atr_s = pd.Series([10.0, 10.0, 10.0])
    pct = atr_pct(close, atr_s)
    # 10/2000 * 100 = 0.5%
    assert abs(pct.iloc[0] - 0.5) < 1e-9


def test_atr_pct_handles_zero_close() -> None:
    close = pd.Series([0.0, 2000.0])
    atr_s = pd.Series([10.0, 10.0])
    pct = atr_pct(close, atr_s)
    assert pd.isna(pct.iloc[0])
    assert abs(pct.iloc[1] - 0.5) < 1e-9


def test_atr_pct_aligns_with_atr_module() -> None:
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "high": [100.5 + i for i in range(n)],
        "low":  [ 99.5 + i for i in range(n)],
        "close":[100.0 + i for i in range(n)],
    }, index=idx)
    a = atr(df["high"], df["low"], df["close"], 14)
    pct = atr_pct(df["close"], a).dropna()
    # ATR ~ 1.0, close ~100..130 -> pct ~ 0.7..1.0 percent
    assert (pct > 0).all()
    assert (pct < 5).all()


def test_session_label_classifies_known_hours() -> None:
    # 02:00 UTC = asia, 08:00 = london, 13:00 = overlap (London+NY), 17:00 = ny, 22:00 = off
    idx = pd.DatetimeIndex([
        "2024-01-02 02:00",
        "2024-01-02 08:00",
        "2024-01-02 13:00",
        "2024-01-02 17:00",
        "2024-01-02 22:00",
    ], tz="UTC")
    labels = session_label(idx)
    assert labels.iloc[0] == "asia"
    assert labels.iloc[1] == "london"
    assert labels.iloc[2] == "overlap"
    assert labels.iloc[3] == "ny"
    assert labels.iloc[4] == "off"


def test_session_label_handles_naive_index() -> None:
    idx = pd.DatetimeIndex(["2024-01-02 08:00", "2024-01-02 13:00"])
    labels = session_label(idx)
    assert labels.iloc[0] == "london"
    assert labels.iloc[1] == "overlap"

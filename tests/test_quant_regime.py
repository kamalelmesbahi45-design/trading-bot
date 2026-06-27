"""QuantRegime overlay tests."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from xauusd_bot.macro.quant_regime import QuantRegime


def _features(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.005, n)
    return pd.DataFrame({
        "dxy": 100 * np.cumprod(1 + rets),
        "us10y": pd.Series(4.0 + np.cumsum(rng.normal(0, 0.002, n)), index=idx),
        "vix": pd.Series(15 + 5 * np.abs(rng.normal(0, 1, n)), index=idx),
        "gold": 2000 * np.cumprod(1 - rets),  # inverse to DXY
    }, index=idx)


def test_quant_regime_score_in_bounds() -> None:
    qr = QuantRegime(_features(200))
    for ts in [datetime(2024, 3, 1), datetime(2024, 5, 15), datetime(2024, 7, 1)]:
        s = qr.score(ts)
        assert -1.0 <= s <= 1.0


def test_quant_regime_missing_columns_raises() -> None:
    df = pd.DataFrame({"dxy": [100.0]})
    with pytest.raises(ValueError):
        QuantRegime(df)


def test_quant_regime_before_history_returns_zero() -> None:
    qr = QuantRegime(_features(200))
    # Before any data exists
    assert qr.score(datetime(1990, 1, 1)) == 0.0

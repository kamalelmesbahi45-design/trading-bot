"""MLFilter pass-through and threshold behaviour."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.strategies.ml_filter import MLFilter, _build_features
from xauusd_bot.types import Side, Signal


class _AlwaysLong(Strategy):
    name = "always_long"

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        last = bars.iloc[-1]
        return [Signal(ts=last.name, strategy=self.name, side=Side.LONG)]


class _ConstantProbaModel:
    """Deterministic stand-in: always returns a fixed class-1 probability."""
    def __init__(self, proba_class_1: float) -> None:
        self.p = proba_class_1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _toy_bars(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, n), index=idx)
    return pd.DataFrame({"high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)


def test_pass_through_when_model_missing(tmp_path: Path) -> None:
    inner = _AlwaysLong()
    flt = MLFilter(inner=inner, model_path=tmp_path / "nope.joblib", threshold=0.6)
    sigs = flt.generate(_toy_bars())
    assert len(sigs) == 1
    assert sigs[0].side is Side.LONG


def test_drops_signal_when_proba_below_threshold(tmp_path: Path) -> None:
    model_path = tmp_path / "m.joblib"
    joblib.dump(_ConstantProbaModel(0.10), model_path)
    flt = MLFilter(inner=_AlwaysLong(), model_path=model_path, threshold=0.6)
    sigs = flt.generate(_toy_bars())
    assert sigs == []


def test_keeps_signal_when_proba_above_threshold(tmp_path: Path) -> None:
    model_path = tmp_path / "m.joblib"
    joblib.dump(_ConstantProbaModel(0.95), model_path)
    flt = MLFilter(inner=_AlwaysLong(), model_path=model_path, threshold=0.6)
    sigs = flt.generate(_toy_bars())
    assert len(sigs) == 1
    assert sigs[0].meta["ml_proba"] == 0.95


def test_feature_row_has_expected_keys() -> None:
    bars = _toy_bars()
    sig = Signal(ts=datetime(2024, 1, 2, 10), strategy="t", side=Side.LONG)
    feats = _build_features(bars, sig)
    for k in ("atr", "rsi", "ema_fast_minus_slow", "ret_1", "ret_5", "side_long", "close"):
        assert k in feats.index

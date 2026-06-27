"""ML signal-quality filter.

Wraps any upstream Strategy. For each signal the wrapped strategy produces, the
filter builds a feature vector (price-action snapshot + regime features) and
queries an XGBoost/LightGBM model predicting P(trade meets target R). Signals
below `threshold` are dropped.

v1 contract:
    * The model is trained OFFLINE from prior backtest fills + features. The
      training script lives in scripts/ and produces `models/ml_filter_*.joblib`.
    * If `model_path` doesn't exist the filter passes signals through unchanged
      (warned once). This keeps the engine usable without a trained model.
    * Featurisation uses bars-up-to-signal-bar only; never look-ahead.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

from xauusd_bot.features.indicators import atr, ema, rsi
from xauusd_bot.strategies.base import Strategy, StrategyContext
from xauusd_bot.types import Signal


def _build_features(bars: pd.DataFrame, sig: Signal) -> pd.Series:
    """Build the feature row for the current signal bar."""
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    a = atr(high, low, close, 14).iloc[-1]
    ema_fast = ema(close, 20).iloc[-1]
    ema_slow = ema(close, 100).iloc[-1]
    r = rsi(close, 14).iloc[-1]
    last_close = close.iloc[-1]
    return pd.Series({
        "atr": float(a) if not pd.isna(a) else 0.0,
        "rsi": float(r) if not pd.isna(r) else 50.0,
        "ema_fast_minus_slow": float(ema_fast - ema_slow) if not (pd.isna(ema_fast) or pd.isna(ema_slow)) else 0.0,
        "ret_1": float(close.pct_change().iloc[-1]) if len(close) > 1 else 0.0,
        "ret_5": float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0,
        "side_long": 1.0 if sig.side.value == "long" else 0.0,
        "close": float(last_close),
    })


class MLFilter(Strategy):
    name = "ml_filter"
    _warned_no_model = False

    def __init__(self, inner: Strategy, model_path: Path, threshold: float = 0.55) -> None:
        self.inner = inner
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model: object | None = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_path.exists():
            if not MLFilter._warned_no_model:
                logger.warning(f"MLFilter: model not found at {self.model_path}; passing signals through")
                MLFilter._warned_no_model = True
            return False
        self._model = joblib.load(self.model_path)
        return True

    def generate(self, bars: pd.DataFrame, ctx: StrategyContext | None = None) -> list[Signal]:
        sigs = self.inner.generate(bars, ctx)
        if not sigs:
            return []
        if not self._load() or self._model is None:
            return sigs   # no model -> pass through

        model = self._model
        kept: list[Signal] = []
        for s in sigs:
            feats = _build_features(bars, s)
            x = feats.to_numpy().reshape(1, -1)
            try:
                proba = float(model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"MLFilter: model predict_proba failed ({e}); passing signal")
                kept.append(s)
                continue
            if proba >= self.threshold:
                kept.append(Signal(
                    ts=s.ts, strategy=s.strategy, side=s.side,
                    confidence=float(proba),
                    sl_price=s.sl_price, tp_price=s.tp_price,
                    meta={**s.meta, "ml_proba": float(proba)},
                ))
        return kept



"""Quantitative regime score from observable variables only. Fully backtestable.

Blend (sign convention: positive = gold-positive):
    -dxy_trend            (DXY up    -> gold negative)
    -sign(real_yield_change) (rising real yields -> gold negative)
    +vix_regime_normalised   (stress  -> gold positive)
    +max(0, -gold_dxy_beta)  (more negative beta = stronger gold/DXY inverse)
All terms clipped to [-1, +1] then averaged. Cached by timestamp lookup.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from xauusd_bot.features.cross_asset import dxy_trend, gold_dxy_beta, real_yield_proxy, vix_regime
from xauusd_bot.macro.base import MacroOverlay


class QuantRegime(MacroOverlay):
    name = "quant_regime"

    def __init__(self, features: pd.DataFrame) -> None:
        """features: daily DataFrame with columns dxy, us10y, vix, gold."""
        required = {"dxy", "us10y", "vix", "gold"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"QuantRegime features missing columns: {missing}")
        self.features = features.sort_index()
        self._score_series = self._build_score()

    def _build_score(self) -> pd.Series:
        f = self.features
        dxy_t = dxy_trend(f["dxy"], 20, 100)            # +1 / -1
        ry = real_yield_proxy(f["us10y"])
        ry_sign = ry.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        vix_r = vix_regime(f["vix"])                    # 0 / 1 / 2
        vix_norm = (vix_r - 1.0)                        # -1 / 0 / +1
        beta = gold_dxy_beta(f["gold"], f["dxy"])
        beta_term = (-beta).clip(lower=-1.0, upper=1.0).fillna(0.0)

        score = (-dxy_t.fillna(0.0) - ry_sign.fillna(0.0) + vix_norm.fillna(0.0) + beta_term) / 4.0
        return score.clip(lower=-1.0, upper=1.0).rename("quant_regime_score")

    def score(self, ts: datetime) -> float:
        """As-of lookup; returns 0.0 if no value available."""
        if self._score_series.empty:
            return 0.0
        target = pd.Timestamp(ts)
        if target.tz is None and self._score_series.index.tz is not None:
            target = target.tz_localize(self._score_series.index.tz)
        elif target.tz is not None and self._score_series.index.tz is None:
            target = target.tz_convert(None)
        # As-of: latest score whose index <= target
        idx = self._score_series.index.searchsorted(target, side="right") - 1
        if idx < 0:
            return 0.0
        val = float(self._score_series.iloc[idx])
        return val if val == val else 0.0  # NaN guard

"""Macro regime model.

Translates the cross-asset context series (DXY, VIX, US10Y, SPX, GOLD, OIL,
HYG vs TLT) into:
    * a ``risk_score`` in [-1, +1]  (+1 = full risk-on, -1 = full risk-off)
    * a ``trend_score`` in [-1, +1] (broad equity / commodity trend strength)
    * per-asset-class biases (multipliers applied to raw signal strength)

The model is intentionally simple and transparent -- five hand-picked features
each contribute one normalised z-score, blended with equal weight. No fitting,
no over-parameterisation; just the textbook macro plumbing.

References used:
    - VIX inverted as risk appetite proxy.
    - SPX above 200-day MA as the trend-following industry's regime filter
      (e.g. Faber's GTAA, Hurst-Pedersen-Pedersen Century-of-Evidence).
    - HYG/TLT ratio as a credit-risk-on/off gauge.
    - Steepening / flattening of long yields (US10Y trend) as duration risk.
    - DXY trend as the global liquidity tide (rising USD = risk-off bias for
      EM and commodities ex-gold).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RegimeRead:
    as_of: pd.Timestamp
    risk_score: float          # [-1, 1]
    trend_score: float         # [-1, 1]
    label: str                 # human label
    features: dict[str, float] = field(default_factory=dict)
    class_bias: dict[str, float] = field(default_factory=dict)

    def to_text(self) -> str:
        lines = [
            f"Regime  as_of={self.as_of.date()}  label={self.label}",
            f"  risk_score  {self.risk_score:+.2f}    trend_score {self.trend_score:+.2f}",
            "  features:",
        ]
        for k, v in self.features.items():
            lines.append(f"    {k:18} {v:+.2f}")
        lines.append("  class bias:")
        for k, v in self.class_bias.items():
            lines.append(f"    {k:14} x{v:.2f}")
        return "\n".join(lines)


def _zclip(x: float, lo: float = -2.0, hi: float = 2.0) -> float:
    if x != x:  # NaN
        return 0.0
    return max(min(x / hi, 1.0), -1.0) if x >= 0 else max(min(x / abs(lo), 1.0), -1.0)


def _trend_z(s: pd.Series, fast: int = 50, slow: int = 200) -> float:
    s = s.dropna()
    if len(s) < slow + 5:
        return 0.0
    ma_fast = s.rolling(fast).mean().iloc[-1]
    ma_slow = s.rolling(slow).mean().iloc[-1]
    if not np.isfinite(ma_slow) or ma_slow == 0:
        return 0.0
    # express in std units of recent moves
    vol = s.pct_change().rolling(slow).std().iloc[-1] or 1e-4
    return float(((ma_fast - ma_slow) / ma_slow) / (vol * np.sqrt(slow)))


def _level_z(s: pd.Series, lookback: int = 252) -> float:
    s = s.dropna()
    if len(s) < lookback // 2:
        return 0.0
    window = s.iloc[-lookback:]
    mu = window.mean()
    sd = window.std(ddof=0) or 1e-9
    return float((s.iloc[-1] - mu) / sd)


def assess_regime(macro: dict[str, pd.DataFrame]) -> RegimeRead:
    """Build a RegimeRead from the macro context dictionary.

    ``macro`` is the dict returned by ``data.load_macro_context``; each value
    is an OHLCV frame. Missing tickers degrade gracefully (their feature is
    treated as 0).
    """

    def close(name: str) -> pd.Series:
        df = macro.get(name)
        if df is None or df.empty:
            return pd.Series(dtype="float64")
        return df["close"].dropna()

    spx = close("SPX")
    vix = close("VIX")
    dxy = close("DXY")
    us10y = close("US10Y")
    gold = close("GOLD")
    hyg = close("HYG")

    features: dict[str, float] = {}

    # SPX trend (risk-on if positive)
    f_spx = _trend_z(spx)
    features["spx_trend"] = f_spx

    # VIX level (risk-off if elevated -> flip sign so risk-on = positive)
    f_vix = -_level_z(vix)
    features["vix_inv_level"] = f_vix

    # DXY trend (rising USD = risk-off pressure on EM/commodities)
    f_dxy = -_trend_z(dxy)
    features["dxy_inv_trend"] = f_dxy

    # US10Y trend (rising long yields = duration risk, risk-off in late cycle)
    f_yld = -_trend_z(us10y)
    features["us10y_inv_trend"] = f_yld

    # Credit appetite: HYG/Gold ratio trend (when HY out-performs gold -> risk-on)
    if not hyg.empty and not gold.empty:
        idx = hyg.index.intersection(gold.index)
        if len(idx) > 60:
            ratio = (hyg.reindex(idx) / gold.reindex(idx)).dropna()
            f_credit = _trend_z(ratio, fast=30, slow=120)
        else:
            f_credit = 0.0
    else:
        f_credit = 0.0
    features["credit_appetite"] = f_credit

    # Blend
    risk_score = float(np.mean([_zclip(v, -2, 2) for v in features.values()]))
    risk_score = max(min(risk_score, 1.0), -1.0)

    trend_score = float(np.mean([_zclip(features["spx_trend"], -2, 2),
                                 _zclip(features["credit_appetite"], -2, 2)]))

    if risk_score > 0.4:
        label = "risk_on_trending" if trend_score > 0.2 else "risk_on_choppy"
    elif risk_score < -0.4:
        label = "risk_off_trending" if trend_score < -0.2 else "risk_off_choppy"
    else:
        label = "mixed"

    # Per-asset-class bias: scales how strongly we lean into raw signals.
    # In risk-off, prefer USD-long / gold / rates; cut equity & risk-on FX.
    r = risk_score
    class_bias = {
        "equity":      1.0 + 0.8 * r,         # 0.2 .. 1.8
        "crypto":      1.0 + 1.0 * r,         # 0.0 .. 2.0
        "fx":          1.0,                    # neutral; handled per-cluster
        "commodity":   1.0 + 0.3 * r,
        "real_assets": 1.0 + 0.5 * r,
        "rates":       1.0 - 0.3 * r,         # rates a partial hedge
    }
    # USD-strength assets get the inverse bias automatically by sign of risk.
    class_bias = {k: round(max(0.1, min(2.0, v)), 3) for k, v in class_bias.items()}

    as_of = pd.Timestamp.now('UTC') if not spx.empty and spx.index[-1] is pd.NaT else (
        spx.index[-1] if not spx.empty else pd.Timestamp.now('UTC')
    )
    return RegimeRead(
        as_of=as_of, risk_score=risk_score, trend_score=trend_score, label=label,
        features={k: round(v, 3) for k, v in features.items()},
        class_bias=class_bias,
    )

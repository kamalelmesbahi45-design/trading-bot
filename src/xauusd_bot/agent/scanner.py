"""Daily opportunity scanner -- the public entry point of the agent.

Pipeline:

    universe -> OHLCV frames -> regime read
                                 |
                                 v
    per-asset (day + swing) signals -> conviction blend
                                            |
                                            v
                          ATR-sized trade tickets per setup
                                            |
                                            v
                  cluster caps + correlation pruning + leverage cap
                                            |
                                            v
                                    AgentReport (text + json)

Every step is pluggable: pass your own ``DataConfig`` to swap data sources,
your own ``RiskSettings`` to dial sizing aggressiveness, your own ``tickers``
list to restrict the universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_bot.agent.data import (
    DataConfig,
    close_panel,
    load_macro_context,
    load_universe,
)
from xauusd_bot.agent.regime import RegimeRead, assess_regime
from xauusd_bot.agent.risk import (
    PortfolioRisk,
    RiskSettings,
    TradeTicket,
    apply_portfolio_rules,
    asset_vol_summary,
    build_ticket,
)
from xauusd_bot.agent.signals import (
    SignalRead,
    cross_sectional_rank,
    day_signal,
    swing_signal,
)
from xauusd_bot.agent.universe import UNIVERSE_BY_TICKER


@dataclass
class AgentSettings:
    mode: str = "both"           # "day" | "swing" | "both"
    min_conviction: float = 0.35
    max_ideas_day: int = 6
    max_ideas_swing: int = 8
    risk: RiskSettings = field(default_factory=RiskSettings)
    data: DataConfig = field(default_factory=DataConfig)
    tickers: list[str] | None = None


@dataclass
class AgentReport:
    as_of: pd.Timestamp
    regime: RegimeRead
    tickets: list[TradeTicket]
    portfolio: PortfolioRisk
    skipped: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


def _conviction(signal_score: float, regime_bias: float, vol_penalty: float) -> float:
    """Map raw signal -> [0, 1] conviction.

    signal_score:  per-asset signal in [-1, 1]
    regime_bias:   class multiplier (0.1 .. 2.0)
    vol_penalty:   1.0 normal, < 1 if asset is unusually volatile (we trust it less)
    """
    raw = abs(signal_score) * regime_bias * vol_penalty
    return float(np.clip(raw, 0.0, 1.0))


def _vol_penalty(ann_vol: float) -> float:
    if not np.isfinite(ann_vol):
        return 1.0
    if ann_vol < 0.10:
        return 0.8       # very low vol -> noisy / mean-reverting -> trust trend less
    if ann_vol > 1.0:
        return 0.5       # extreme vol (crypto blow-off) -> cut conviction in half
    if ann_vol > 0.5:
        return 0.75
    return 1.0


def run_agent(settings: AgentSettings | None = None) -> AgentReport:
    settings = settings or AgentSettings()

    # 1. Data
    frames = load_universe(settings.data, tickers=settings.tickers)
    macro = load_macro_context(settings.data)
    if not frames:
        return AgentReport(
            as_of=pd.Timestamp.now('UTC'),
            regime=RegimeRead(pd.Timestamp.now('UTC'), 0.0, 0.0, "no_data"),
            tickets=[],
            portfolio=PortfolioRisk(0, 0, {}, {}, 0, 0, 0, 0),
            skipped={"_universe": "no usable data; run `xauusd data fetch` or check network"},
        )

    panel = close_panel(frames)
    returns_panel = panel.pct_change()

    # 2. Macro regime
    regime = assess_regime(macro)

    # 3. Per-asset signals
    swing_scores: dict[str, float] = {}
    swing_components: dict[str, SignalRead] = {}
    for tk, df in frames.items():
        s = swing_signal(df["close"])
        swing_scores[tk] = s.score
        swing_components[tk] = s
    # add cross-sectional rank kicker
    xs = cross_sectional_rank(swing_scores)
    for tk in frames:
        swing_components[tk] = swing_signal(frames[tk]["close"], xs_rank=xs.get(tk, 0.0))

    day_components: dict[str, SignalRead] = {}
    for tk, df in frames.items():
        day_components[tk] = day_signal(df)

    # 4. Build tickets
    candidates: list[TradeTicket] = []
    skipped: dict[str, str] = {}
    diag: dict[str, object] = {"per_asset": {}}

    for tk, df in frames.items():
        asset = UNIVERSE_BY_TICKER[tk]
        regime_bias = regime.class_bias.get(asset.asset_class, 1.0)
        vol_stats = asset_vol_summary(df["close"])
        vp = _vol_penalty(vol_stats["ann_vol"])

        diag_entry = {
            "ann_vol": round(vol_stats["ann_vol"], 3) if np.isfinite(vol_stats["ann_vol"]) else None,
            "drawdown_3m": round(vol_stats["drawdown_3m"], 3)
            if np.isfinite(vol_stats["drawdown_3m"]) else None,
            "swing_score": round(swing_components[tk].score, 3),
            "day_score": round(day_components[tk].score, 3),
            "day_strategy": day_components[tk].strategy,
        }
        diag["per_asset"][tk] = diag_entry  # type: ignore[index]

        # Swing
        if settings.mode in ("swing", "both"):
            s = swing_components[tk]
            conv = _conviction(s.score, regime_bias, vp)
            if conv >= settings.min_conviction and s.side != "FLAT":
                rationale = (
                    f"swing TSMOM {s.score:+.2f} (3m {s.components.get('ret_3m',0):+.1%} "
                    f"6m {s.components.get('ret_6m',0):+.1%} 12m {s.components.get('ret_12m',0):+.1%}) "
                    f"xs {s.components.get('xs_rank',0):+.2f}; regime bias x{regime_bias:.2f}"
                )
                t = build_ticket(
                    ticker=tk, label=asset.label, asset_class=asset.asset_class,
                    cluster=asset.cluster, ohlc=df, side=s.side, horizon="swing",
                    strategy=s.strategy, conviction=conv, settings=settings.risk,
                    rationale=rationale,
                )
                if t is not None:
                    candidates.append(t)
                else:
                    skipped[f"{tk}:swing"] = "ATR/stop calc failed"

        # Day
        if settings.mode in ("day", "both"):
            d = day_components[tk]
            conv = _conviction(d.score, regime_bias, vp)
            if conv >= settings.min_conviction and d.side != "FLAT":
                comps = ", ".join(f"{k}={v:+.2f}" for k, v in d.components.items())
                rationale = f"day {d.strategy} ({comps}); regime bias x{regime_bias:.2f}"
                t = build_ticket(
                    ticker=tk, label=asset.label, asset_class=asset.asset_class,
                    cluster=asset.cluster, ohlc=df, side=d.side, horizon="day",
                    strategy=d.strategy, conviction=conv, settings=settings.risk,
                    rationale=rationale,
                )
                if t is not None:
                    candidates.append(t)
                else:
                    skipped[f"{tk}:day"] = "ATR/stop calc failed"

    # 5. Cap by per-horizon idea count *before* portfolio rules
    day_c = sorted([c for c in candidates if c.horizon == "day"],
                   key=lambda c: -c.conviction)[: settings.max_ideas_day]
    swing_c = sorted([c for c in candidates if c.horizon == "swing"],
                     key=lambda c: -c.conviction)[: settings.max_ideas_swing]
    capped = day_c + swing_c

    # 6. Portfolio rules
    admitted, port = apply_portfolio_rules(capped, returns_panel, settings.risk)
    admitted = sorted(admitted, key=lambda t: -t.conviction)

    return AgentReport(
        as_of=panel.index[-1] if not panel.empty else pd.Timestamp.now('UTC'),
        regime=regime,
        tickets=admitted,
        portfolio=port,
        skipped=skipped,
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# convenience for ad-hoc use from a notebook / REPL
# ---------------------------------------------------------------------------
def run_from_cache(cache_dir: Path = Path("data/cache/multi")) -> AgentReport:
    """Run against the on-disk multi-asset CSV panel without network."""
    settings = AgentSettings(data=DataConfig(csv_panel_dir=cache_dir, allow_network=False))
    return run_agent(settings)

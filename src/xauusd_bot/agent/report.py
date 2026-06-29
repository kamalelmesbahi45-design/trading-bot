"""Pretty-printing for AgentReport.

Two outputs:
    * format_text  -- monospace ASCII summary suitable for terminals / Telegram
    * to_dict      -- JSON-serialisable view of the same data, for dashboards

Nothing in here makes trading decisions; this is purely presentation.
"""
from __future__ import annotations

from typing import Any

from xauusd_bot.agent.scanner import AgentReport


def to_dict(report: AgentReport) -> dict[str, Any]:
    return {
        "as_of": str(report.as_of),
        "regime": {
            "label": report.regime.label,
            "risk_score": report.regime.risk_score,
            "trend_score": report.regime.trend_score,
            "features": report.regime.features,
            "class_bias": report.regime.class_bias,
        },
        "portfolio": {
            "gross": report.portfolio.gross,
            "net": report.portfolio.net,
            "n_long": report.portfolio.n_long,
            "n_short": report.portfolio.n_short,
            "by_class_notional": report.portfolio.by_class_notional,
            "by_cluster_risk": report.portfolio.by_cluster_risk,
            "avg_long_corr": report.portfolio.avg_long_corr,
            "avg_short_corr": report.portfolio.avg_short_corr,
            "dropped": report.portfolio.dropped,
            "warnings": report.portfolio.warnings,
        },
        "tickets": [
            {
                "ticker": t.ticker, "label": t.label, "asset_class": t.asset_class,
                "cluster": t.cluster, "horizon": t.horizon, "strategy": t.strategy,
                "side": t.side, "entry": t.entry, "stop": t.stop, "target": t.target,
                "atr": t.atr_value, "units": t.units, "risk_usd": t.risk_dollars,
                "notional": t.notional, "weight_pct": t.weight_pct,
                "r_multiple": t.r_multiple, "conviction": t.conviction,
                "rationale": t.rationale,
            }
            for t in report.tickets
        ],
        "skipped": report.skipped,
        "diagnostics": report.diagnostics,
    }


def format_text(report: AgentReport, equity: float | None = None) -> str:
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append(f"MULTI-ASSET MACRO AGENT  as_of={report.as_of}  ")
    lines.append("=" * 96)
    lines.append("")

    # Regime block
    lines.append(report.regime.to_text())
    lines.append("")

    # Portfolio summary
    p = report.portfolio
    lines.append("Portfolio")
    lines.append(f"  gross {p.gross:.2f}x   net {p.net:+.2f}x   "
                 f"longs {p.n_long}  shorts {p.n_short}")
    if p.by_class_notional:
        lines.append("  notional by class: " + ", ".join(
            f"{k} {v:.0%}" for k, v in p.by_class_notional.items()
        ))
    if p.by_cluster_risk:
        lines.append("  risk $ by cluster: " + ", ".join(
            f"{k} ${v:.0f}" for k, v in p.by_cluster_risk.items()
        ))
    lines.append(f"  avg pairwise corr: long {p.avg_long_corr:+.2f}  "
                 f"short {p.avg_short_corr:+.2f}")
    if p.dropped:
        lines.append(f"  dropped by risk rules: {', '.join(p.dropped)}")
    if p.warnings:
        for w in p.warnings:
            lines.append(f"  WARN: {w}")
    lines.append("")

    # Trade tickets
    if report.tickets:
        lines.append(f"Trade tickets ({len(report.tickets)})")
        lines.append("-" * 96)
        header = (
            f"{'tkr':4} {'name':12} {'horz':5} {'strategy':18} {'side':6} "
            f"{'entry':>8} {'stop':>8} {'tgt':>8} {'atr':>6} {'units':>9} "
            f"{'risk$':>8} {'wgt':>5} {'R':>3} {'conv':>5}"
        )
        lines.append(header)
        for t in report.tickets:
            lines.append(t.to_line())
        lines.append("")
        lines.append("Rationale")
        for t in report.tickets:
            lines.append(f"  {t.ticker:4} {t.horizon:5} -> {t.rationale}")
    else:
        lines.append("No qualifying trades today. "
                     "Lower min_conviction or check that the data feed is fresh.")

    if report.skipped:
        lines.append("")
        lines.append("Skipped setups")
        for k, v in report.skipped.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=" * 96)
    lines.append(
        "Reminder: these are model outputs, not advice. Validate against your own "
        "execution venue, slippage, and risk budget before placing any order."
    )
    return "\n".join(lines)

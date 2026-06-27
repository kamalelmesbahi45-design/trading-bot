"""Render an HTML report from a backtest / WFO / MC result bundle.

Single self-contained HTML file; embeds Plotly figures via plotly.io.to_html
with include_plotlyjs='cdn'. No template engine needed -- a compact Python
f-string keeps the surface minimal.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd
import plotly.io as pio

from xauusd_bot.optimize.metrics import PerformanceStats, compute_stats
from xauusd_bot.optimize.monte_carlo import MonteCarloResult
from xauusd_bot.reports.plots import (
    drawdown_fig,
    equity_curve_fig,
    mc_distribution_fig,
    monthly_returns_heatmap,
    per_strategy_contribution,
)


def _fig_html(fig: object) -> str:
    html: str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, default_width="100%")
    return html


def _stats_table(stats: PerformanceStats) -> str:
    rows = ""
    items: dict[str, object] = asdict(stats) if is_dataclass(stats) else stats.__dict__
    for k, v in items.items():
        s = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows += f"<tr><td>{k}</td><td>{s}</td></tr>"
    return f"<table class='stats'><tbody>{rows}</tbody></table>"


def render_report(
    out_dir: Path,
    equity: pd.Series,
    trades: pd.DataFrame,
    stats: PerformanceStats | None = None,
    mc_result: MonteCarloResult | None = None,
    wfo_oos: pd.Series | None = None,
    title: str = "XAUUSD bot report",
) -> Path:
    """Render to <out_dir>/report.html and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = stats or compute_stats(equity, trades)

    figs: list[str] = []
    figs.append(_fig_html(equity_curve_fig(equity)))
    figs.append(_fig_html(drawdown_fig(equity)))
    figs.append(_fig_html(monthly_returns_heatmap(equity)))
    figs.append(_fig_html(per_strategy_contribution(trades)))
    if wfo_oos is not None and not wfo_oos.empty:
        figs.append(_fig_html(equity_curve_fig(wfo_oos, title="Walk-forward OOS equity")))
    if mc_result is not None:
        figs.append(_fig_html(mc_distribution_fig(mc_result.dd_distribution,
                                                  title="MC Max Drawdown")))
        figs.append(_fig_html(mc_distribution_fig(mc_result.cagr_distribution,
                                                  title="MC CAGR")))
        figs.append(_fig_html(mc_distribution_fig(mc_result.sharpe_distribution,
                                                  title="MC Sharpe")))

    body = "\n<hr>\n".join(figs)
    extra = ""
    if mc_result is not None:
        extra = (f"<p><b>Probability of ruin:</b> {mc_result.p_ruin:.2%} &nbsp;"
                 f"<b>P95 drawdown:</b> {mc_result.p95_drawdown:.2%}</p>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ border-bottom: 1px solid #ccc; padding-bottom: .3em; }}
    table.stats {{ border-collapse: collapse; margin: 1em 0; }}
    table.stats td {{ padding: 4px 12px; border-bottom: 1px solid #eee; }}
    table.stats td:first-child {{ font-weight: 600; color: #444; }}
    hr {{ border: 0; border-top: 1px solid #eee; margin: 1.5em 0; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <h2>Summary</h2>
  {_stats_table(stats)}
  {extra}
  <h2>Charts</h2>
  {body}
</body>
</html>
"""
    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out

"""Run the full hypothesis battery on real XAUUSD daily + macro context.

Prints a ranked table + writes an HTML dashboard with per-hypothesis equity curves.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from research.loaders import daily_join, load_all
from research.macro_features import build_macro_features
from research.tester import run_all, summarise


def main() -> None:
    parts = load_all()
    daily = daily_join(parts)
    bars = daily[["gold_open", "gold_high", "gold_low", "gold_close", "gold_volume"]].rename(
        columns=lambda c: c.replace("gold_", "")
    )
    feats = build_macro_features(daily)

    print(f"data span: {daily.index.min()} -> {daily.index.max()}  ({len(daily)} days)")
    print(f"features  : {list(feats.columns)}")

    # Use the last 20% of the series as OOS hold-out
    cut = int(len(daily) * 0.80)
    oos_start = daily.index[cut]
    print(f"OOS start : {oos_start} (last 20%)")

    results = run_all(feats, bars, oos_start=oos_start)
    table = summarise(results)
    print("\n=== Hypothesis ranking (sorted by deflated Sharpe) ===")
    print(table.to_string(index=False))

    # Save table + HTML dashboard
    out_dir = Path("reports/output/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "hypothesis_summary.csv", index=False)

    fig = go.Figure()
    for r in results:
        if r.equity_curve.empty:
            continue
        fig.add_trace(go.Scatter(x=r.equity_curve.index, y=r.equity_curve.values,
                                  mode="lines", name=f"{r.id} {r.name[:35]}"))
    fig.update_layout(title="Per-hypothesis equity curves (real XAUUSD daily)",
                      template="plotly_white", height=600,
                      xaxis_title="date", yaxis_title="equity (USD)")
    chart_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                              default_width="100%")

    rows = "".join(
        f"<tr><td>{r.id}</td><td>{r.name}</td>"
        f"<td>{r.n_trades}</td><td>${r.final_equity:,.0f}</td>"
        f"<td>{r.cagr:+.2%}</td><td>{r.sharpe:.2f}</td>"
        f"<td>{r.sortino:.2f}</td><td>{r.max_dd:.2%}</td>"
        f"<td>{r.calmar:.2f}</td><td>{r.hit_rate:.2%}</td>"
        f"<td>{r.profit_factor:.2f}</td><td>{r.deflated_sharpe:.3f}</td>"
        f"<td>{(r.oos_sharpe or 0):.2f}</td></tr>"
        for r in sorted(results, key=lambda x: x.deflated_sharpe, reverse=True)
    )

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>XAUUSD edge research</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 1px solid #ccc; padding-bottom: .3em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f5f5f7; }}
.note {{ background: #fff8c5; padding: 8px 12px; border-radius: 6px; margin: 1em 0; }}
</style></head><body>
<h1>XAUUSD edge research -- 13 grounded hypotheses</h1>
<p>Data: {daily.index.min().date()} to {daily.index.max().date()} ({len(daily)} daily bars).
   Cost: 14 bps per side (gold spread ~28c + slippage on ~$2000 price).
   Position trades next close. Walk-forward OOS hold-out = last 20% of the series.</p>
<p class='note'>
   <b>Multi-test correction:</b> Sharpes are tested across {len(results)} hypotheses
   simultaneously. Deflated Sharpe (Bailey & Lopez de Prado) accounts for this.
   A hypothesis is "real edge" if Sharpe &gt; 1 AND DefSh &gt; 0.95 AND OOS Sharpe &gt; 0.5.
</p>
<table>
<thead><tr>
<th>ID</th><th>Name</th><th>n_changes</th><th>final$</th><th>CAGR</th>
<th>Sharpe</th><th>Sortino</th><th>MaxDD</th><th>Calmar</th>
<th>HitRate</th><th>PF</th><th>DefSh</th><th>OOS Sh</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
{chart_html}
</body></html>
"""
    out_path = out_dir / "research_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nresearch dashboard: {out_path}")


if __name__ == "__main__":
    main()

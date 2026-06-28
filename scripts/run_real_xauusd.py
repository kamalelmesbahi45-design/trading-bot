"""Honest validation run on real XAUUSD H1 history.

Data: ForexBot GitHub repo (Mehrshad-Sorena/ForexBot/.../1H/XAUUSD_i.csv).
Span: 2020-11-23 -> 2022-04-22 (~17 months, 8,324 H1 bars).

Pipeline (per strategy):
    1. Load real H1 bars.
    2. Run engine on full series.
    3. Walk-forward validate (IS=6m, OOS=3m, step=3m).
    4. Monte Carlo: 5,000 shuffles of the realised trades.
    5. Compute stats with deflated Sharpe (n_trials=4 to penalise multi-test).
    6. Render a per-strategy HTML report.

Then a combined dashboard summarises all four side-by-side.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xauusd_bot.backtest.engine import BacktestEngine
from xauusd_bot.config import load_config
from xauusd_bot.optimize.metrics import compute_stats
from xauusd_bot.optimize.monte_carlo import shuffle_trades
from xauusd_bot.optimize.walk_forward import WalkForward
from xauusd_bot.reports.html_report import render_report


def load_h1(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    out = df[["open", "high", "low", "close", "volume"]].astype("float64").copy()
    return out


def _base_cfg():
    cfg = load_config("configs/personal_aggressive.yaml")
    cfg.filters.session.enabled = False
    cfg.filters.news.enabled = False
    # Use a slightly tighter regime filter -- skip dead chop only.
    cfg.filters.regime.enabled = True
    cfg.filters.regime.atr_pct_min = 0.05
    cfg.filters.regime.atr_pct_max = 5.0
    cfg.risk.sizer = "fixed_fractional"
    cfg.risk.per_trade_pct = 0.005
    cfg.risk.hard_cap_per_trade_pct = 0.01
    cfg.risk.max_concurrent_positions = 1
    cfg.risk.trailing_stop = True
    cfg.stops.atr_mult_sl = 2.0
    cfg.stops.atr_mult_tp = 3.0
    cfg.stops.breakeven_at_r = 1.0
    cfg.execution.spread_model = "static"
    cfg.execution.spread_static_points = 25     # 25c gold spread
    cfg.execution.slippage_points = 3
    cfg.execution.commission_per_lot = 7.0
    # disable all strategies; per-run enables one
    for k in cfg.strategies:
        cfg.strategies[k].enabled = False
    return cfg


@dataclass
class StratRun:
    name: str
    stats: object
    mc: object
    final_equity: float
    n_trades: int
    wfo_oos_curve: pd.Series
    wfo_folds: list[dict[str, float | str]]
    report_path: Path


def run_strategy(strat_key: str, bars: pd.DataFrame, out_dir: Path) -> StratRun:
    from xauusd_bot.config import StrategyCfg
    cfg = _base_cfg()
    if strat_key not in cfg.strategies:
        cfg.strategies[strat_key] = StrategyCfg(enabled=True, weight=1.0, timeframe="H1")
    cfg.strategies[strat_key].enabled = True

    eng = BacktestEngine(cfg)
    res = eng.run(bars)
    stats = compute_stats(res.equity_curve, res.trades, n_trials=4)
    mc = shuffle_trades(res.trades, n_runs=5000, seed=0,
                       starting_equity=cfg.account.starting_equity,
                       ruin_dd_pct=cfg.risk.max_drawdown_pct,
                       periods_per_year=252.0)

    wf = WalkForward(cfg, is_years=0.5, oos_months=3, step_months=3)
    wfo = wf.run(bars, start=bars.index[0].to_pydatetime(), end=bars.index[-1].to_pydatetime())

    sdir = out_dir / strat_key
    sdir.mkdir(parents=True, exist_ok=True)
    path = render_report(sdir, res.equity_curve, res.trades, stats=stats, mc_result=mc,
                         wfo_oos=wfo.oos_equity,
                         title=f"XAUUSD H1 -- {strat_key} (real data)")

    return StratRun(
        name=strat_key,
        stats=stats,
        mc=mc,
        final_equity=float(res.metadata.get("final_equity", 0.0)),
        n_trades=int(len(res.trades)),
        wfo_oos_curve=wfo.oos_equity,
        wfo_folds=wfo.fold_results,
        report_path=path,
    )


def render_combined(out_dir: Path, runs: list[StratRun], starting_equity: float) -> Path:
    rows = ""
    for r in runs:
        s = r.stats
        rows += (
            f"<tr>"
            f"<td><a href='{r.name}/report.html'>{r.name}</a></td>"
            f"<td>{r.n_trades}</td>"
            f"<td>${r.final_equity:,.0f}</td>"
            f"<td>{s.cagr:+.2%}</td>"
            f"<td>{s.sharpe:.2f}</td>"
            f"<td>{s.sortino:.2f}</td>"
            f"<td>{s.max_drawdown:.2%}</td>"
            f"<td>{s.calmar:.2f}</td>"
            f"<td>{s.win_rate:.2%}</td>"
            f"<td>{s.profit_factor:.2f}</td>"
            f"<td>{(s.deflated_sharpe or 0):.3f}</td>"
            f"<td>{r.mc.p_ruin:.2%}</td>"
            f"<td>{r.mc.p95_drawdown:.2%}</td>"
            f"<td>{len(r.wfo_folds)}</td>"
            f"</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>XAUUSD H1 -- strategy shootout</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ border-bottom: 1px solid #ccc; padding-bottom: .3em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; font-weight: 600; }}
th {{ background: #f5f5f7; }}
a {{ color: #0a66c2; }}
.note {{ background: #fff8c5; padding: 8px 12px; border-radius: 6px; margin: 1em 0; }}
</style></head><body>
<h1>XAUUSD H1 -- strategy shootout (real data)</h1>
<p>Data: 8,324 H1 bars, 2020-11-23 to 2022-04-22 (~17 months).
   Starting equity ${starting_equity:,.0f}. Fixed-fractional 0.5% risk, ATR(14) stops 2x SL / 3x TP, 25c static spread, 3-point slippage, $7/lot commission. Trailing stop on.</p>
<p class='note'>
   Deflated Sharpe uses n_trials=4 to penalise the multi-test setup. A
   strategy clears the validation bar when sharpe > 1.0 AND deflated_sharpe > 0.95
   AND P(ruin) &lt; 5% AND WFO produced at least 2 OOS folds and finished positive.
</p>
<table>
<thead><tr>
<th>strategy</th><th>n_trades</th><th>final_eq</th><th>CAGR</th>
<th>Sharpe</th><th>Sortino</th><th>MaxDD</th><th>Calmar</th>
<th>WinRate</th><th>PF</th><th>DefSh</th><th>P(ruin)</th><th>P95 DD</th><th>WFO folds</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
<p>Click a strategy name to open its full report (equity curve, drawdown, monthly heatmap, MC distributions).</p>
</body></html>
"""
    out = out_dir / "shootout.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    bars = load_h1(Path("data/cache/real/xauusd_h1.csv"))
    print(f"loaded {len(bars):,} H1 bars  {bars.index[0]} -> {bars.index[-1]}")

    out = Path("reports/output/real")
    out.mkdir(parents=True, exist_ok=True)

    runs: list[StratRun] = []
    for key in ("trend_donchian", "mr_bollinger", "breakout_london", "breakout_asian"):
        print(f"\n=== {key} ===")
        r = run_strategy(key, bars, out)
        s = r.stats
        print(f"  trades={r.n_trades}  final=${r.final_equity:,.0f}  "
              f"Sharpe={s.sharpe:.2f}  MaxDD={s.max_drawdown:.2%}  "
              f"WR={s.win_rate:.2%}  PF={s.profit_factor:.2f}  "
              f"DefSh={(s.deflated_sharpe or 0):.3f}  P(ruin)={r.mc.p_ruin:.2%}")
        runs.append(r)

    cfg = _base_cfg()
    path = render_combined(out, runs, cfg.account.starting_equity)
    print(f"\nshootout dashboard: {path}")


if __name__ == "__main__":
    main()

"""CLI entry-point. `xauusd <command>`."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from rich import print as rprint

from xauusd_bot.config import load_config

load_dotenv()

app = typer.Typer(add_completion=False, help="xauusd-bot CLI")


def _cfg_path(config: str | None) -> Path:
    p = config or os.environ.get("XAUUSD_BOT_CONFIG") or "configs/personal_aggressive.yaml"
    path = Path(p)
    if not path.exists():
        raise typer.BadParameter(f"Config not found: {path}")
    return path


@app.command()
def info(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Print loaded config (for sanity)."""
    cfg = load_config(_cfg_path(config))
    rprint(cfg.model_dump())


@app.command()
def data(
    action: str = typer.Argument(..., help="fetch | calendar | inspect"),
    years: int = typer.Option(10, "--years", help="Years of history to fetch (for action=fetch)."),
    cache_dir: Path = typer.Option(Path("data/cache"), "--cache-dir"),
    cross_asset_interval: str = typer.Option("1h", "--cross-interval"),
) -> None:
    """Data layer commands.

    Actions:
        fetch    -- Dukascopy XAUUSD ticks + yfinance cross-asset for the last N years.
        calendar -- ForexFactory current-window event calendar.
        inspect  -- Print cache summary (counts and sizes).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=int(365.25 * years))

    if action == "fetch":
        from xauusd_bot.data.dukascopy import DukascopyConfig, download_ticks
        from xauusd_bot.data.yfinance_loader import YFinanceConfig, fetch_cross_asset

        logger.info(f"fetching {years}y XAUUSD ticks from Dukascopy into {cache_dir} ...")
        ticks = download_ticks(start, end, DukascopyConfig(cache_dir=cache_dir))
        logger.info(f"ticks: {len(ticks):,} rows")
        logger.info(f"fetching cross-asset @ {cross_asset_interval} ...")
        wide = fetch_cross_asset(start, end, YFinanceConfig(cache_dir=cache_dir, interval=cross_asset_interval))
        logger.info(f"cross-asset: {len(wide):,} rows across {wide['ticker'].nunique() if not wide.empty else 0} tickers")
        return

    if action == "calendar":
        from xauusd_bot.data.calendar import fetch_current_window

        events = fetch_current_window(cache_dir=cache_dir)
        rprint(events.head(50))
        rprint(f"total events: {len(events)}")
        return

    if action == "inspect":
        sizes: dict[str, int] = {}
        if cache_dir.exists():
            for p in cache_dir.rglob("*.parquet"):
                sizes[p.relative_to(cache_dir).parts[0]] = sizes.get(p.relative_to(cache_dir).parts[0], 0) + 1
        rprint({"cache_dir": str(cache_dir), "parquet_counts_by_kind": sizes})
        return

    raise typer.BadParameter(f"unknown action: {action}")


@app.command()
def backtest(
    config: str | None = typer.Option(None, "--config", "-c"),
    bars_parquet: Path = typer.Option(..., "--bars", help="Path to OHLCV parquet (open/high/low/close)."),
    start: str | None = typer.Option(None, "--start", help="ISO date to slice from."),
    end: str | None = typer.Option(None, "--end", help="ISO date to slice to."),
) -> None:
    """Run a single-shot backtest on the configured strategies against a parquet bars file."""
    from xauusd_bot.backtest.engine import BacktestEngine
    from xauusd_bot.data.storage import load_parquet

    cfg = load_config(_cfg_path(config))
    bars = load_parquet(bars_parquet)
    if start is not None:
        bars = bars.loc[start:]
    if end is not None:
        bars = bars.loc[:end]
    logger.info(f"backtest profile={cfg.profile} bars={len(bars):,} range=[{bars.index[0]}..{bars.index[-1]}]")
    res = BacktestEngine(cfg).run(bars)
    rprint(res.metadata)
    rprint(f"trades: {len(res.trades)}  | equity curve points: {len(res.equity_curve)}")


@app.command()
def wfo(
    config: str | None = typer.Option(None, "--config", "-c"),
    bars_parquet: Path = typer.Option(..., "--bars"),
    is_years: float = typer.Option(2.0, "--is"),
    oos_months: int = typer.Option(6, "--oos"),
    step_months: int = typer.Option(6, "--step"),
) -> None:
    """Run walk-forward out-of-sample validation."""
    from xauusd_bot.data.storage import load_parquet
    from xauusd_bot.optimize.walk_forward import WalkForward

    cfg = load_config(_cfg_path(config))
    bars = load_parquet(bars_parquet)
    if bars.empty:
        raise typer.BadParameter("bars file is empty")
    logger.info(f"wfo profile={cfg.profile} IS={is_years}y OOS={oos_months}m step={step_months}m bars={len(bars):,}")
    wf = WalkForward(cfg, is_years=is_years, oos_months=oos_months, step_months=step_months)
    res = wf.run(bars, start=bars.index[0].to_pydatetime(), end=bars.index[-1].to_pydatetime())
    rprint(f"folds: {len(res.fold_results)}  stability: {res.stability_score:.3f}")
    rprint(f"OOS curve points: {len(res.oos_equity)}")
    for f in res.fold_results:
        rprint(f)


@app.command()
def report(
    config: str | None = typer.Option(None, "--config", "-c"),
    bars_parquet: Path = typer.Option(..., "--bars"),
    out: Path = typer.Option(Path("reports/output"), "--out"),
    mc_runs: int = typer.Option(2000, "--mc-runs"),
) -> None:
    """Run backtest + Monte Carlo and render the HTML report."""
    from xauusd_bot.backtest.engine import BacktestEngine
    from xauusd_bot.data.storage import load_parquet
    from xauusd_bot.optimize.metrics import compute_stats
    from xauusd_bot.optimize.monte_carlo import shuffle_trades
    from xauusd_bot.reports.html_report import render_report

    cfg = load_config(_cfg_path(config))
    bars = load_parquet(bars_parquet)
    logger.info(f"report profile={cfg.profile} bars={len(bars):,}")
    res = BacktestEngine(cfg).run(bars)
    stats = compute_stats(res.equity_curve, res.trades)
    mc = shuffle_trades(res.trades, n_runs=mc_runs, seed=0,
                       starting_equity=cfg.account.starting_equity,
                       ruin_dd_pct=cfg.risk.max_drawdown_pct)
    path = render_report(out, res.equity_curve, res.trades, stats=stats, mc_result=mc,
                        title=f"XAUUSD bot report -- {cfg.profile}")
    rprint(f"report written: {path}")


@app.command()
def paper(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Forward-test on live data via the paper execution engine."""
    cfg = load_config(_cfg_path(config))
    logger.info(f"paper profile={cfg.profile}")
    raise NotImplementedError("Wire up in execution phase.")


@app.command()
def agent(
    mode: str = typer.Option("both", "--mode", "-m", help="day | swing | both"),
    equity: float = typer.Option(100_000.0, "--equity", "-e", help="Account equity in $."),
    risk_pct: float = typer.Option(0.005, "--risk-pct", "-r",
                                   help="Per-trade risk as fraction of equity (0.005 = 0.5%)."),
    tickers: str | None = typer.Option(None, "--tickers",
                                       help="Comma-separated subset of internal tickers."),
    min_conviction: float = typer.Option(0.35, "--min-conviction"),
    no_network: bool = typer.Option(False, "--no-network",
                                    help="Skip yfinance and use cached/CSV data only."),
    cache_dir: Path = typer.Option(Path("data/cache/agent"), "--cache-dir"),
    csv_dir: Path = typer.Option(Path("data/cache/multi"), "--csv-dir"),
    json_out: Path | None = typer.Option(None, "--json", help="Optional path to write JSON output."),
) -> None:
    """Run the multi-asset macro agent and print today's opportunities.

    Examples:
        xauusd agent                                  # day + swing, $100k, 0.5% risk
        xauusd agent --mode swing --equity 25000
        xauusd agent --tickers GLD,SPX,BTC --mode day
        xauusd agent --no-network --csv-dir data/cache/multi
    """
    import json as _json
    from xauusd_bot.agent.data import DataConfig
    from xauusd_bot.agent.report import format_text, to_dict
    from xauusd_bot.agent.risk import RiskSettings
    from xauusd_bot.agent.scanner import AgentSettings, run_agent

    tickers_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    settings = AgentSettings(
        mode=mode,
        min_conviction=min_conviction,
        risk=RiskSettings(equity=equity, risk_pct_per_trade=risk_pct),
        data=DataConfig(cache_dir=cache_dir, csv_panel_dir=csv_dir,
                        allow_network=not no_network),
        tickers=tickers_list,
    )
    report = run_agent(settings)
    print(format_text(report, equity=equity))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(to_dict(report), indent=2, default=str))
        rprint(f"json written: {json_out}")


@app.command()
def live(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Run live on MT5. Requires MT5_* env vars and a healthy paper-test history."""
    cfg = load_config(_cfg_path(config))
    if cfg.execution.engine != "mt5":
        raise typer.BadParameter("Config execution.engine must be 'mt5' for live.")
    logger.warning("live trading: MAKE SURE backtest + paper validated this profile")
    raise NotImplementedError("Wire up in execution phase.")


if __name__ == "__main__":
    app()

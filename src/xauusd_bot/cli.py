"""CLI entry-point. `xauusd <command>`."""
from __future__ import annotations

import os
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
    action: str = typer.Argument(..., help="fetch | update | inspect"),
    years: int = typer.Option(10, "--years"),
) -> None:
    """Data layer commands (Dukascopy ticks, yfinance cross-asset, FF calendar)."""
    logger.info(f"data {action} years={years}")
    raise NotImplementedError("Wire up in data layer phase.")


@app.command()
def backtest(
    config: str | None = typer.Option(None, "--config", "-c"),
    start: str = typer.Option("2014-01-01", "--start"),
    end: str | None = typer.Option(None, "--end"),
) -> None:
    """Run a single-shot backtest on the configured strategies."""
    cfg = load_config(_cfg_path(config))
    logger.info(f"backtest profile={cfg.profile} start={start} end={end}")
    raise NotImplementedError("Wire up in backtest engine phase.")


@app.command()
def wfo(
    config: str | None = typer.Option(None, "--config", "-c"),
    is_years: float = typer.Option(2.0, "--is"),
    oos_months: int = typer.Option(6, "--oos"),
) -> None:
    """Run walk-forward optimisation."""
    cfg = load_config(_cfg_path(config))
    logger.info(f"wfo profile={cfg.profile} IS={is_years}y OOS={oos_months}m")
    raise NotImplementedError("Wire up in optimiser phase.")


@app.command()
def report(run: str = typer.Option("latest", "--run")) -> None:
    """Render HTML report for a backtest/WFO run."""
    logger.info(f"report run={run}")
    raise NotImplementedError("Wire up in reports phase.")


@app.command()
def paper(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Forward-test on live data via the paper execution engine."""
    cfg = load_config(_cfg_path(config))
    logger.info(f"paper profile={cfg.profile}")
    raise NotImplementedError("Wire up in execution phase.")


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

"""Smoke test: CLI `info` command loads and prints config."""
from __future__ import annotations

from typer.testing import CliRunner

from xauusd_bot.cli import app

runner = CliRunner()


def test_cli_info_personal() -> None:
    result = runner.invoke(app, ["info", "--config", "configs/personal_aggressive.yaml"])
    assert result.exit_code == 0, result.stdout
    assert "personal_aggressive" in result.stdout


def test_cli_info_propfirm() -> None:
    result = runner.invoke(app, ["info", "--config", "configs/propfirm_strict.yaml"])
    assert result.exit_code == 0, result.stdout
    assert "propfirm_strict" in result.stdout


def test_cli_agent_help() -> None:
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "macro agent" in result.stdout.lower()


def test_cli_agent_offline_no_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No network and an empty cache dir: should still exit 0 with a no-data report."""
    result = runner.invoke(app, [
        "agent", "--no-network",
        "--csv-dir", str(tmp_path / "missing"),
        "--cache-dir", str(tmp_path / "agent_cache"),
        "--equity", "10000",
    ])
    assert result.exit_code == 0, result.stdout
    assert "MULTI-ASSET MACRO AGENT" in result.stdout

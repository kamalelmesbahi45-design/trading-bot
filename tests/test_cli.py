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

"""Smoke tests for config loading. Both profiles must parse."""
from __future__ import annotations

from pathlib import Path

import pytest

from xauusd_bot.config import BotConfig, load_config


@pytest.mark.parametrize("name", ["personal_aggressive.yaml", "propfirm_strict.yaml"])
def test_config_loads(configs_dir: Path, name: str) -> None:
    cfg = load_config(configs_dir / name)
    assert isinstance(cfg, BotConfig)
    assert cfg.account.starting_equity > 0
    assert cfg.risk.per_trade_pct > 0
    assert cfg.execution.symbol == "XAUUSD"


def test_propfirm_has_daily_loss_limit(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "propfirm_strict.yaml")
    assert cfg.risk.daily_loss_limit_pct is not None
    assert cfg.risk.max_drawdown_pct <= 0.10


def test_kelly_capped_below_one(configs_dir: Path) -> None:
    """Sanity: nobody should be running anywhere near full Kelly."""
    for name in ("personal_aggressive.yaml", "propfirm_strict.yaml"):
        cfg = load_config(configs_dir / name)
        assert 0 < cfg.risk.kelly_fraction <= 0.5
        assert cfg.risk.hard_cap_per_trade_pct <= 0.05

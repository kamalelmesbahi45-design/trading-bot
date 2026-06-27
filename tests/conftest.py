"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir() -> Path:
    return ROOT / "configs"

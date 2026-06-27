"""Claude-as-analyst macro overlay.

Once per cadence (daily by default), feed Claude a structured digest:
    - latest US macro prints (CPI/NFP/PCE/FOMC)
    - rate path expectations
    - geopolitical headlines (Reuters/AP RSS)
    - cross-asset moves: DXY, US10Y, real yields, SPX, oil, BTC
And ask for a JSON-only response:
    { "score": -1.0..1.0, "confidence": 0..1, "drivers": ["...","..."] }

Cached per-day. Advisory only: bounded sizing multiplier, never opens trades alone.
NOT backtestable historically. Forward-test only.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from xauusd_bot.macro.base import MacroOverlay


class LLMAnalyst(MacroOverlay):
    name = "llm_analyst"

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        cache_dir: Path | None = None,
        max_sizing_multiplier: float = 1.2,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.max_sizing_multiplier = max_sizing_multiplier

    def _digest_for(self, day: date) -> str:
        """Build the prompt-input digest for a given day."""
        raise NotImplementedError

    def _query(self, digest: str) -> dict[str, float | list[str]]:
        raise NotImplementedError

    def score(self, ts: datetime) -> float:
        raise NotImplementedError

"""News-headline sentiment via FinBERT (transformers).

Pulls headlines from a configurable RSS/JSON feed, scores each with FinBERT, aggregates
daily into a score in [-1, +1]. Optional plug-in (disabled by default).
"""
from __future__ import annotations

from datetime import datetime

from xauusd_bot.macro.base import MacroOverlay


class FinbertSentiment(MacroOverlay):
    name = "sentiment_finbert"

    def __init__(self, model_id: str = "ProsusAI/finbert", feed_urls: list[str] | None = None) -> None:
        self.model_id = model_id
        self.feed_urls = feed_urls or []
        self._pipe = None

    def _load(self) -> None:
        raise NotImplementedError

    def score(self, ts: datetime) -> float:
        raise NotImplementedError

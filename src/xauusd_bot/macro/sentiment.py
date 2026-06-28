"""News-headline sentiment via FinBERT.

Optional dependency: install `xauusd-bot[sentiment]` to pull in `transformers`
and `torch`. If the dep is missing, the overlay logs a warning and returns 0.0
so the engine keeps running.

This v1 doesn't ship a feed integration (RSS/JSON polling lives at the runner
level). Use score(ts) by pre-feeding headlines via add_headlines(ts, headlines).
"""
from __future__ import annotations

from datetime import datetime

from loguru import logger

from xauusd_bot.macro.base import MacroOverlay


class FinbertSentiment(MacroOverlay):
    name = "sentiment_finbert"

    def __init__(self, model_id: str = "ProsusAI/finbert", feed_urls: list[str] | None = None) -> None:
        self.model_id = model_id
        self.feed_urls = feed_urls or []
        self._pipe: object | None = None
        self._cache: dict[datetime, float] = {}

    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("FinbertSentiment: 'transformers' not installed; "
                           "install with `pip install xauusd-bot[sentiment]`. Returning neutral scores.")
            return False
        self._pipe = pipeline("sentiment-analysis", model=self.model_id)
        return True

    def add_headlines(self, ts: datetime, headlines: list[str]) -> None:
        """Score a batch of headlines and cache the aggregate at ts."""
        if not headlines or not self._load() or self._pipe is None:
            self._cache[ts] = 0.0
            return
        pipe = self._pipe
        results = pipe(headlines)  # type: ignore[operator]
        agg = 0.0
        for r in results:
            label = str(r.get("label", "")).lower()
            score = float(r.get("score", 0.0))
            if label == "positive":
                agg += score
            elif label == "negative":
                agg -= score
        self._cache[ts] = max(-1.0, min(1.0, agg / max(len(headlines), 1)))

    def score(self, ts: datetime) -> float:
        if ts in self._cache:
            return self._cache[ts]
        if not self._cache:
            return 0.0
        # As-of: latest ts <= target
        keys = sorted(self._cache)
        latest = next((k for k in reversed(keys) if k <= ts), None)
        return self._cache.get(latest, 0.0) if latest is not None else 0.0

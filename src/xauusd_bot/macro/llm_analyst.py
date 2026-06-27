"""Claude-as-analyst macro overlay.

Once per cadence (daily by default), feed Claude a structured digest of macro
+ geopolitical signals and ask for a JSON score in [-1, +1]. Cached per-day to
disk so we don't re-query for the same input.

Optional dependency: install `xauusd-bot[llm]` to pull in `anthropic`. If
unavailable, score() returns 0.0 (neutral) so the engine keeps running.

NOT historically backtestable -- only useful forward. Treat as advisory:
multiplies sizing by (1 + alpha * score), bounded by max_sizing_multiplier.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from xauusd_bot.macro.base import MacroOverlay

_SYSTEM_PROMPT = (
    "You are a macro analyst for XAUUSD (gold). You will receive a daily "
    "digest with structured macro + cross-asset + geopolitical signals. "
    "Respond with ONLY a JSON object: "
    "{\"score\": <-1.0..1.0>, \"confidence\": <0..1>, \"drivers\": [\"...\",\"...\"]}. "
    "Positive score = gold-positive (risk-off, falling real yields, weak USD, "
    "stress). Negative = gold-negative. Be decisive but conservative; the "
    "score is used to scale position size, not direction."
)


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
        self._client: object | None = None

    def _client_ok(self) -> bool:
        if self._client is not None:
            return True
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("LLMAnalyst: ANTHROPIC_API_KEY not set; returning 0.0.")
            return False
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("LLMAnalyst: 'anthropic' not installed; "
                           "install with `pip install xauusd-bot[llm]`. Returning 0.0.")
            return False
        self._client = Anthropic()
        return True

    def _cache_path(self, day: date) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "llm_analyst" / f"{day.isoformat()}.json"

    def _load_cached(self, day: date) -> dict[str, float | list[str]] | None:
        p = self._cache_path(day)
        if p is None or not p.exists():
            return None
        try:
            return json.loads(p.read_text())  # type: ignore[no-any-return]
        except Exception:
            return None

    def _save_cached(self, day: date, payload: dict[str, float | list[str]]) -> None:
        p = self._cache_path(day)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))

    def build_digest(self, day: date, signals: dict[str, float], headlines: list[str] | None = None) -> str:
        """Caller assembles the input; we just format it for the model."""
        lines = [f"Date: {day.isoformat()}", "Signals:"]
        for k, v in signals.items():
            lines.append(f"  - {k}: {v:.4f}")
        if headlines:
            lines.append("Headlines:")
            for h in headlines[:30]:
                lines.append(f"  - {h}")
        return "\n".join(lines)

    def query(self, digest: str) -> dict[str, float | list[str]] | None:
        if not self._client_ok() or self._client is None:
            return None
        client = self._client
        try:
            resp = client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=400,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": digest}],
            )
        except Exception as e:
            logger.warning(f"LLMAnalyst: API error: {e}")
            return None
        text = "".join(blk.text for blk in resp.content if blk.type == "text").strip()
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            logger.warning(f"LLMAnalyst: non-JSON reply: {text[:200]}")
            return None

    def score(self, ts: datetime) -> float:
        day = ts.date() if hasattr(ts, "date") else ts
        cached = self._load_cached(day)
        if cached is not None:
            val = cached.get("score", 0.0)
            return float(val) if isinstance(val, (int, float)) else 0.0
        # No cached value AND no live query path triggered here. Caller is
        # responsible for invoking query()/save_cached() at the appropriate
        # cadence so score() stays a cheap lookup.
        return 0.0

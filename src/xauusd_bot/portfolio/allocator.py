"""Combine signals from N strategies into ordered, sized intents.

Resolution rules:
    1. Drop signals if filters (session/news/weekend/regime) say no-trade.
    2. If multiple strategies signal same side: keep highest confidence, scale qty.
    3. If conflicting sides: net out by weighted confidence.
    4. Cap by max_concurrent_positions.
"""
from __future__ import annotations

from dataclasses import dataclass

from xauusd_bot.types import Signal


@dataclass
class AllocatorConfig:
    weights: dict[str, float]
    max_concurrent_positions: int


class Allocator:
    def __init__(self, cfg: AllocatorConfig) -> None:
        self.cfg = cfg

    def reconcile(self, signals: list[Signal]) -> list[Signal]:
        raise NotImplementedError

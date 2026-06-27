"""Macro overlay ABC. Emits a continuous risk-on/off score in [-1, +1] per timestamp.

Convention:
    +1.0  =  strong gold-positive regime (risk-off, falling real yields, weak DXY)
     0.0  =  neutral
    -1.0  =  strong gold-negative regime (risk-on, rising real yields, strong DXY)

The portfolio layer multiplies sizing by (1 + alpha * score) where alpha is bounded.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class MacroOverlay(ABC):
    name: str

    @abstractmethod
    def score(self, ts: datetime) -> float:
        ...

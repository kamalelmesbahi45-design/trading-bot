"""Telegram bot for live alerts.

Uses python-telegram-bot. Credentials from env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
Non-blocking: sends are fired into a background queue so a Telegram outage never
blocks the trading loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str


class TelegramNotifier:
    def __init__(self, cfg: TelegramConfig) -> None:
        self.cfg = cfg
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    def send(self, text: str) -> None:
        """Fire-and-forget. Drops messages if Telegram is unavailable (logged)."""
        raise NotImplementedError

    def alert_fill(self, side: str, qty: float, price: float, strategy: str) -> None:
        raise NotImplementedError

    def alert_dd_warning(self, current_dd_pct: float, threshold_pct: float) -> None:
        raise NotImplementedError

    def alert_killed(self, reason: str) -> None:
        raise NotImplementedError

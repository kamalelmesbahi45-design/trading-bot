"""Telegram bot for live alerts.

Uses the Telegram Bot API directly via httpx -- no python-telegram-bot
machinery needed for outbound-only messaging. send() is non-blocking: it
enqueues onto a background thread that POSTs each message and drops on
HTTP failure (logged) so a Telegram outage never blocks the trading loop.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import httpx
from loguru import logger

_API = "https://api.telegram.org"
_DROP_SENTINEL = object()


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    timeout_s: float = 10.0


class TelegramNotifier:
    def __init__(self, cfg: TelegramConfig) -> None:
        self.cfg = cfg
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telegram-sender", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._queue.put(_DROP_SENTINEL)
        self._thread.join(timeout=2.0)
        self._thread = None

    def send(self, text: str) -> None:
        """Fire-and-forget. Caller never blocks."""
        if self._thread is None:
            logger.warning("TelegramNotifier.send() called before start(); message dropped")
            return
        self._queue.put(text)

    # --- convenience helpers ---

    def alert_fill(self, side: str, qty: float, price: float, strategy: str) -> None:
        self.send(f"FILL {side} {qty} @ {price:.2f} ({strategy})")

    def alert_dd_warning(self, current_dd_pct: float, threshold_pct: float) -> None:
        self.send(f"DD WARN: {current_dd_pct*100:.2f}% (threshold {threshold_pct*100:.2f}%)")

    def alert_killed(self, reason: str) -> None:
        self.send(f"BOT KILLED: {reason}")

    # --- worker ---

    def _run(self) -> None:
        url = f"{_API}/bot{self.cfg.bot_token}/sendMessage"
        with httpx.Client(timeout=self.cfg.timeout_s) as client:
            while not self._stop.is_set():
                item = self._queue.get()
                if item is _DROP_SENTINEL:
                    return
                assert isinstance(item, str)
                try:
                    r = client.post(url, json={"chat_id": self.cfg.chat_id, "text": item})
                    if r.status_code != 200:
                        logger.warning(f"telegram: HTTP {r.status_code}: {r.text[:200]}")
                except httpx.HTTPError as e:
                    logger.warning(f"telegram: send failed: {e}")

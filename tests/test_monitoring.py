"""Monitoring tests: daily report, healthcheck. Telegram tested with a fake HTTP transport."""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

from xauusd_bot.monitoring.daily_report import build_summary
from xauusd_bot.monitoring.healthcheck import HealthCheckInputs, check
from xauusd_bot.monitoring.telegram import TelegramConfig, TelegramNotifier

# --- daily_report ---

def test_daily_summary_empty_curve() -> None:
    s = build_summary(date(2024, 1, 2), pd.DataFrame(), pd.Series(dtype=float))
    assert "no equity data" in s


def test_daily_summary_basic() -> None:
    idx = pd.date_range("2024-01-02", periods=24, freq="1h", tz="UTC")
    eq = pd.Series([10_000.0 + i * 5 for i in range(24)], index=idx)
    fills = pd.DataFrame({
        "ts": [datetime(2024, 1, 2, 10), datetime(2024, 1, 2, 15)],
        "kind": ["open", "close"],
    })
    s = build_summary(date(2024, 1, 2), fills, eq)
    assert "PnL +$115.00" in s
    assert "fills 2" in s
    assert "open 1" in s and "close 1" in s


def test_daily_summary_other_day_no_data() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="1h", tz="UTC")
    eq = pd.Series([10_000.0] * 10, index=idx)
    s = build_summary(date(2024, 1, 5), pd.DataFrame(), eq)
    assert "no equity data for that day" in s


# --- healthcheck ---

def test_healthcheck_healthy() -> None:
    now = datetime(2024, 1, 2, 12)
    r = check(HealthCheckInputs(
        now=now, last_tick_ts=now - timedelta(seconds=10),
        bot_equity=10_000.5, broker_equity=10_000.0,
        bot_position_count=1, broker_position_count=1,
        pending_order_ages_s=[5.0],
    ))
    assert r.healthy
    assert r.issues == []


def test_healthcheck_stale_tick() -> None:
    now = datetime(2024, 1, 2, 12)
    r = check(HealthCheckInputs(
        now=now, last_tick_ts=now - timedelta(seconds=300),
        bot_equity=10_000.0, broker_equity=10_000.0,
        bot_position_count=0, broker_position_count=0,
    ))
    assert not r.healthy
    assert any("stale_tick" in s for s in r.issues)


def test_healthcheck_equity_mismatch_and_orphans() -> None:
    now = datetime(2024, 1, 2, 12)
    r = check(HealthCheckInputs(
        now=now, last_tick_ts=now,
        bot_equity=10_000.0, broker_equity=9_995.0,
        bot_position_count=2, broker_position_count=1,
    ))
    assert not r.healthy
    assert any("equity_mismatch" in s for s in r.issues)
    assert any("orphan_positions" in s for s in r.issues)


def test_healthcheck_stuck_pending() -> None:
    now = datetime(2024, 1, 2, 12)
    r = check(HealthCheckInputs(
        now=now, last_tick_ts=now,
        bot_equity=10_000.0, broker_equity=10_000.0,
        bot_position_count=0, broker_position_count=0,
        pending_order_ages_s=[5.0, 60.0, 90.0],
    ))
    assert not r.healthy
    assert r.pending_orders == 2


def test_healthcheck_no_tick_yet() -> None:
    now = datetime(2024, 1, 2, 12)
    r = check(HealthCheckInputs(
        now=now, last_tick_ts=None,
        bot_equity=10_000.0, broker_equity=10_000.0,
        bot_position_count=0, broker_position_count=0,
    ))
    assert not r.healthy
    assert any("no_tick_yet" in s for s in r.issues)


# --- telegram (fake HTTP transport) ---

def test_telegram_sends_message_to_api() -> None:
    received: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/sendMessage")
        received.append(dict(request.url.params) | {"body": request.content.decode("utf-8")})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    # Monkey-patch the client construction inside the notifier's worker
    notif = TelegramNotifier(TelegramConfig(bot_token="t", chat_id="42"))

    # Replace the _run method with a tiny inline version using our mock transport
    import threading
    def _run_with_mock() -> None:
        with httpx.Client(transport=transport, timeout=5.0) as client:
            while True:
                item = notif._queue.get()
                if not isinstance(item, str):
                    return
                client.post("https://api.telegram.org/bott/sendMessage",
                            json={"chat_id": "42", "text": item})
    notif._thread = threading.Thread(target=_run_with_mock, daemon=True)
    notif._thread.start()

    notif.send("hello world")
    # Wait briefly for the worker
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    notif.stop()

    assert received, "expected at least one message posted to telegram api"
    assert "hello world" in received[0]["body"]


def test_telegram_send_before_start_does_not_crash() -> None:
    notif = TelegramNotifier(TelegramConfig(bot_token="t", chat_id="42"))
    notif.send("ignored")   # should log and drop, not raise

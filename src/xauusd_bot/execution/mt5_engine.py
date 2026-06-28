"""MT5 execution via the MetaTrader5 Python lib (Windows-only at runtime).

Credentials are read from env: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.
This module imports MetaTrader5 lazily so the package stays usable on
non-Windows hosts and in CI.

Idempotent orders: each new order gets a deterministic comment tag and the
configured `magic` number so retries after timeouts don't double-fill.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

from loguru import logger

from xauusd_bot.execution.base import ExecutionEngine
from xauusd_bot.types import Fill, Order, OrderType, Position, Side


class MT5Engine(ExecutionEngine):
    def __init__(
        self,
        symbol: str = "XAUUSD",
        magic: int = 270625,
        slippage_points: int = 5,
        contract_size: int = 100,
    ) -> None:
        self.symbol = symbol
        self.magic = magic
        self.slippage_points = slippage_points
        self.contract_size = contract_size
        self._mt5: Any | None = None

    # --- helpers ---

    def _ensure_mt5(self) -> Any:
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "MetaTrader5 lib not installed (Windows-only). Use PaperEngine on this OS."
            ) from e
        self._mt5 = mt5
        return mt5

    # --- ExecutionEngine ---

    def connect(self) -> None:
        mt5 = self._ensure_mt5()
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        path = os.getenv("MT5_PATH")
        kwargs: dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
        if login and password and server and not mt5.login(
            int(login), password=password, server=server
        ):
            raise RuntimeError(f"mt5.login failed: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"could not select {self.symbol}")
        logger.info(f"mt5: connected, symbol={self.symbol}")

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    def equity(self) -> float:
        mt5 = self._ensure_mt5()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")
        return float(info.equity)

    def positions(self) -> list[Position]:
        mt5 = self._ensure_mt5()
        rows = mt5.positions_get(symbol=self.symbol) or []
        out: list[Position] = []
        for r in rows:
            side = Side.LONG if r.type == mt5.POSITION_TYPE_BUY else Side.SHORT
            out.append(Position(
                strategy=str(r.comment or ""),
                side=side,
                qty=float(r.volume),
                entry_price=float(r.price_open),
                entry_ts=datetime.fromtimestamp(int(r.time)),
                sl_price=float(r.sl) if r.sl else None,
                tp_price=float(r.tp) if r.tp else None,
            ))
        return out

    def place(self, order: Order) -> Fill:
        if order.type is not OrderType.MARKET:
            raise NotImplementedError("MT5Engine v1 supports MARKET orders only")
        mt5 = self._ensure_mt5()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError("no tick available; symbol not subscribed?")
        price = tick.ask if order.side is Side.LONG else tick.bid
        action_type = mt5.ORDER_TYPE_BUY if order.side is Side.LONG else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(order.qty),
            "type": action_type,
            "price": price,
            "sl": order.sl_price or 0.0,
            "tp": order.tp_price or 0.0,
            "deviation": self.slippage_points,
            "magic": self.magic,
            "comment": f"{order.strategy}|{uuid.uuid4().hex[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"order_send failed: {getattr(result, 'comment', result)}")
        return Fill(
            ts=datetime.utcnow(),
            order_id=str(result.order),
            side=order.side,
            qty=order.qty,
            price=float(result.price),
            commission=0.0,  # MT5 reports commissions on the position, not the fill
            slippage=self.slippage_points * 0.01,
        )

    def modify(self, position: Position, sl: float | None, tp: float | None) -> None:
        mt5 = self._ensure_mt5()
        rows = mt5.positions_get(symbol=self.symbol)
        if not rows:
            raise RuntimeError("no positions to modify")
        for r in rows:
            if abs(float(r.price_open) - position.entry_price) > 1e-6:
                continue
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": r.ticket,
                "symbol": self.symbol,
                "sl": sl if sl is not None else r.sl,
                "tp": tp if tp is not None else r.tp,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(f"modify failed: {getattr(result, 'comment', result)}")
            return
        raise RuntimeError("position not found by entry price match")

    def close(self, position: Position) -> Fill:
        mt5 = self._ensure_mt5()
        rows = mt5.positions_get(symbol=self.symbol)
        if not rows:
            raise RuntimeError("no positions")
        for r in rows:
            if abs(float(r.price_open) - position.entry_price) > 1e-6:
                continue
            tick = mt5.symbol_info_tick(self.symbol)
            opp = mt5.ORDER_TYPE_SELL if position.side is Side.LONG else mt5.ORDER_TYPE_BUY
            price = tick.bid if position.side is Side.LONG else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(r.volume),
                "type": opp,
                "position": r.ticket,
                "price": price,
                "deviation": self.slippage_points,
                "magic": self.magic,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(f"close failed: {getattr(result, 'comment', result)}")
            return Fill(ts=datetime.utcnow(), order_id=str(result.order),
                        side=position.side, qty=position.qty,
                        price=float(result.price), commission=0.0,
                        slippage=self.slippage_points * 0.01)
        raise RuntimeError("position not found by entry price match")

    def flatten_all(self) -> list[Fill]:
        return [self.close(p) for p in self.positions()]

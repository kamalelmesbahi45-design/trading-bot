"""Paper execution. Reuses backtest fill simulator against a live tick/quote feed.

The runner feeds the engine market quotes via `on_quote(bid, ask, ts)`. Orders
placed via `place()` fill at the current quote with spread + slippage applied
(same model as backtest, so paper and backtest results converge). SL/TP for
each open position are checked on every quote update.

State (equity + open positions) is persisted to JSON on every fill, so a crash
or restart does not lose track of open trades.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from xauusd_bot.backtest.costs import POINT_VALUE, StaticSpread
from xauusd_bot.execution.base import ExecutionEngine
from xauusd_bot.types import Bar, Fill, Order, OrderStatus, OrderType, Position, Side


class PaperEngine(ExecutionEngine):
    """Same execution-engine surface as MT5, driven by a live quote stream."""

    def __init__(
        self,
        starting_equity: float,
        state_path: Path,
        spread_points: int = 25,
        slippage_points: int = 5,
        commission_per_lot: float = 7.0,
        contract_size: int = 100,
    ) -> None:
        self.state_path = state_path
        self.spread = StaticSpread(points=spread_points)
        self.slippage_points = slippage_points
        self.commission_per_lot = commission_per_lot
        self.contract_size = contract_size

        self._equity = starting_equity
        self._positions: list[Position] = []
        self._last_bid: float | None = None
        self._last_ask: float | None = None
        self._last_ts: datetime | None = None
        self._connected = False

    # --- ExecutionEngine ---

    def connect(self) -> None:
        if self.state_path.exists():
            try:
                self._load_state()
                logger.info(f"paper: loaded state from {self.state_path} eq={self._equity:.2f} "
                            f"positions={len(self._positions)}")
            except Exception as e:
                logger.warning(f"paper: failed to load state ({e}); starting fresh")
        self._connected = True

    def disconnect(self) -> None:
        self._persist()
        self._connected = False

    def equity(self) -> float:
        if self._last_bid is None or self._last_ask is None:
            return self._equity
        mark = (self._last_bid + self._last_ask) * 0.5
        unreal = sum(p.unrealized_pnl(mark, self.contract_size) for p in self._positions)
        return self._equity + unreal

    def positions(self) -> list[Position]:
        return list(self._positions)

    def place(self, order: Order) -> Fill:
        if self._last_bid is None or self._last_ask is None:
            raise RuntimeError("PaperEngine has no market data yet; call on_quote() first")
        if order.type is not OrderType.MARKET:
            raise NotImplementedError("PaperEngine only fills MARKET orders in v1")

        slip = self.slippage_points * POINT_VALUE
        price = self._last_ask + slip if order.side is Side.LONG else self._last_bid - slip

        order_id = order.id or str(uuid.uuid4())
        commission = order.qty * self.commission_per_lot
        self._equity -= commission

        pos = Position(
            strategy=order.strategy, side=order.side, qty=order.qty,
            entry_price=price, entry_ts=order.ts,
            sl_price=order.sl_price, tp_price=order.tp_price,
            initial_sl_price=order.sl_price, high_water_mark=price,
        )
        self._positions.append(pos)
        order.status = OrderStatus.FILLED
        order.id = order_id

        fill = Fill(ts=order.ts, order_id=order_id, side=order.side,
                    qty=order.qty, price=price, commission=commission, slippage=slip)
        self._persist()
        return fill

    def modify(self, position: Position, sl: float | None, tp: float | None) -> None:
        if position not in self._positions:
            raise ValueError("unknown position")
        if sl is not None:
            position.sl_price = sl
        if tp is not None:
            position.tp_price = tp
        self._persist()

    def close(self, position: Position) -> Fill:
        if position not in self._positions:
            raise ValueError("unknown position")
        if self._last_bid is None or self._last_ask is None:
            raise RuntimeError("no market data")
        slip = self.slippage_points * POINT_VALUE
        # Close long = sell at bid; close short = buy at ask
        price = (self._last_bid - slip) if position.side is Side.LONG else (self._last_ask + slip)
        sign = 1 if position.side is Side.LONG else -1
        gross = sign * (price - position.entry_price) * position.qty * self.contract_size
        commission = position.qty * self.commission_per_lot
        self._equity += gross - commission
        self._positions.remove(position)
        self._persist()
        return Fill(ts=self._last_ts or datetime.utcnow(), order_id=str(uuid.uuid4()),
                    side=position.side, qty=position.qty, price=price,
                    commission=commission, slippage=slip)

    def flatten_all(self) -> list[Fill]:
        return [self.close(p) for p in list(self._positions)]

    # --- live-feed surface ---

    def on_quote(self, bid: float, ask: float, ts: datetime) -> list[Fill]:
        """Called by the runner on every quote update.

        Returns a list of Fills generated by SL/TP hits in this quote.
        """
        self._last_bid = bid
        self._last_ask = ask
        self._last_ts = ts

        fills: list[Fill] = []
        # Convert the quote to a synthetic Bar so we can reuse intra-bar SL/TP logic.
        bar = Bar(ts=ts, open=ask, high=ask, low=bid, close=(bid + ask) * 0.5, volume=0.0)
        for pos in list(self._positions):
            if pos.side is Side.LONG:
                sl_hit = pos.sl_price is not None and bid <= pos.sl_price
                tp_hit = pos.tp_price is not None and ask >= pos.tp_price
                hit_price = pos.sl_price if sl_hit else pos.tp_price if tp_hit else None
            else:
                sl_hit = pos.sl_price is not None and ask >= pos.sl_price
                tp_hit = pos.tp_price is not None and bid <= pos.tp_price
                hit_price = pos.sl_price if sl_hit else pos.tp_price if tp_hit else None
            if hit_price is None:
                continue
            sign = 1 if pos.side is Side.LONG else -1
            gross = sign * (hit_price - pos.entry_price) * pos.qty * self.contract_size
            commission = pos.qty * self.commission_per_lot
            self._equity += gross - commission
            self._positions.remove(pos)
            fills.append(Fill(ts=ts, order_id=str(uuid.uuid4()), side=pos.side,
                              qty=pos.qty, price=hit_price,
                              commission=commission, slippage=0.0))
        if fills:
            self._persist()
        # Bar var silenced (kept for future intra-bar use):
        _ = bar
        return fills

    # --- persistence ---

    def _persist(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "equity": self._equity,
                "positions": [
                    {**{k: v for k, v in asdict(p).items() if k != "entry_ts"},
                     "entry_ts": p.entry_ts.isoformat(),
                     "side": p.side.value}
                    for p in self._positions
                ],
            }
            self.state_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:  # pragma: no cover - best-effort
            logger.warning(f"paper: persist failed ({e})")

    def _load_state(self) -> None:
        data = json.loads(self.state_path.read_text())
        self._equity = float(data["equity"])
        self._positions = []
        for p in data.get("positions", []):
            self._positions.append(Position(
                strategy=p["strategy"],
                side=Side(p["side"]),
                qty=float(p["qty"]),
                entry_price=float(p["entry_price"]),
                entry_ts=datetime.fromisoformat(p["entry_ts"]),
                sl_price=p.get("sl_price"),
                tp_price=p.get("tp_price"),
                initial_sl_price=p.get("initial_sl_price"),
                high_water_mark=p.get("high_water_mark"),
            ))

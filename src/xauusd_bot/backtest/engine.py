"""Event-driven backtest loop.

Bar-clock convention:
    Signals are emitted at the END of bar i and filled at the OPEN of bar i+1.
    SL/TP checks for an existing position happen on every subsequent bar's range.
    The same execution code path (fill_sim) is used in paper and live so
    behaviour is consistent.

This v1 supports a single strategy (DonchianTrend), the prop-firm risk gates,
session + volatility-regime filters, and dynamic spread/slippage costs.
News and weekend filters require historical calendar data (deferred); they are
left as no-ops here when their config flags are off or no events are passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

import pandas as pd

from xauusd_bot.backtest.costs import DynamicSpread, StaticSpread
from xauusd_bot.backtest.fill_sim import FillSimulator
from xauusd_bot.config import BotConfig, SessionWindow
from xauusd_bot.features.indicators import atr as atr_indicator
from xauusd_bot.features.regime_features import atr_pct
from xauusd_bot.risk.propfirm_gates import GateDecision, GateState, PropFirmGates
from xauusd_bot.risk.sizing import (
    FixedFractional,
    FractionalKelly,
    Sizer,
    SizingInputs,
)
from xauusd_bot.risk.stops import StopPolicy
from xauusd_bot.strategies.base import Strategy
from xauusd_bot.strategies.breakout import AsianRangeBreakout, LondonOpenBreakout
from xauusd_bot.strategies.mean_reversion import BollingerMR
from xauusd_bot.strategies.trend import DonchianTrend
from xauusd_bot.types import Bar, Order, OrderType, Position, Side, Signal


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    daily_pnl: pd.Series
    metadata: dict[str, str | float] = field(default_factory=dict)


@dataclass
class _PendingOrder:
    signal: Signal
    estimated_entry: float


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def _in_session(ts: datetime, windows: list[SessionWindow]) -> bool:
    if not windows:
        return True
    t = ts.time()
    for w in windows:
        start = _parse_hhmm(w.start)
        end = _parse_hhmm(w.end)
        if start <= end:
            if start <= t < end:
                return True
        else:  # window crosses midnight (rare)
            if t >= start or t < end:
                return True
    return False


def _build_sizer(cfg: BotConfig) -> Sizer:
    r = cfg.risk
    if r.sizer == "fixed_fractional":
        return FixedFractional(pct_per_trade=r.per_trade_pct, hard_cap_pct=r.hard_cap_per_trade_pct)
    if r.sizer == "fractional_kelly":
        return FractionalKelly(
            kelly_fraction=r.kelly_fraction,
            fallback_pct=r.per_trade_pct,
            hard_cap_pct=r.hard_cap_per_trade_pct,
            min_trades=r.kelly_min_trades,
            lookback_trades=r.kelly_lookback_trades,
        )
    raise ValueError(f"unsupported sizer: {r.sizer}")


def _build_spread(cfg: BotConfig) -> StaticSpread | DynamicSpread:
    if cfg.execution.spread_model == "static":
        return StaticSpread(points=cfg.execution.spread_static_points)
    return DynamicSpread(base_points=cfg.execution.spread_static_points)


def _build_strategies(cfg: BotConfig) -> list[Strategy]:
    out: list[Strategy] = []
    strats = cfg.strategies
    if strats.get("trend_donchian") and strats["trend_donchian"].enabled:
        out.append(DonchianTrend(donchian_period=20))
    if strats.get("mr_bollinger") and strats["mr_bollinger"].enabled:
        out.append(BollingerMR())
    if strats.get("breakout_london") and strats["breakout_london"].enabled:
        out.append(LondonOpenBreakout())
    if strats.get("breakout_asian") and strats["breakout_asian"].enabled:
        out.append(AsianRangeBreakout())
    if not out:
        raise ValueError("no strategies enabled in config")
    return out


def _build_gates(cfg: BotConfig) -> PropFirmGates:
    return PropFirmGates(
        daily_loss_limit_pct=cfg.risk.daily_loss_limit_pct,
        max_drawdown_pct=cfg.risk.max_drawdown_pct,
        profit_target_pct=cfg.risk.profit_target_pct,
        min_trading_days=cfg.risk.min_trading_days,
    )


class BacktestEngine:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        bars: pd.DataFrame,
        cross_asset: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
    ) -> BacktestResult:
        cfg = self.cfg
        if bars.empty:
            return BacktestResult(
                equity_curve=pd.Series(dtype=float),
                trades=pd.DataFrame(),
                fills=pd.DataFrame(),
                daily_pnl=pd.Series(dtype=float),
            )

        required = {"open", "high", "low", "close"}
        if not required.issubset(set(bars.columns)):
            raise ValueError(f"bars missing required columns: {required - set(bars.columns)}")

        # Pre-compute ATR series and regime helpers.
        atr_series = atr_indicator(bars["high"], bars["low"], bars["close"], cfg.stops.atr_period)
        atr_pct_series = atr_pct(bars["close"], atr_series)

        sizer = _build_sizer(cfg)
        spread = _build_spread(cfg)
        fill_sim = FillSimulator(
            spread_model=spread,
            slippage_points=cfg.execution.slippage_points,
            commission_per_lot=cfg.execution.commission_per_lot,
            contract_size=cfg.execution.contract_size,
        )
        stop_policy = StopPolicy(
            atr_mult_sl=cfg.stops.atr_mult_sl,
            atr_mult_tp=cfg.stops.atr_mult_tp,
            breakeven_at_r=cfg.stops.breakeven_at_r,
            trailing=cfg.risk.trailing_stop,
        )
        strategies = _build_strategies(cfg)
        gates = _build_gates(cfg)
        state = GateState(starting_equity=cfg.account.starting_equity)
        contract = cfg.execution.contract_size

        equity = cfg.account.starting_equity
        positions: list[Position] = []
        pending: list[_PendingOrder] = []
        trades: list[dict[str, float | str | datetime]] = []
        fills_log: list[dict[str, float | str | datetime]] = []
        equity_pts: list[tuple[datetime, float]] = []
        last_day = None
        is_killed = False

        ts_index = bars.index

        for i, ts in enumerate(ts_index):
            row = bars.iloc[i]
            bar = _to_bar(ts, row)
            atr_now = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
            atr_p_now = float(atr_pct_series.iloc[i]) if not pd.isna(atr_pct_series.iloc[i]) else 0.0

            # New trading day bookkeeping
            day = bar.ts.date()
            if last_day != day:
                state.on_day_start(day, equity)
                last_day = day

            # 1) Fill pending orders at this bar's open
            if pending and not is_killed:
                for po in pending:
                    sig = po.signal
                    order = Order(
                        ts=bar.ts,
                        strategy=sig.strategy,
                        side=sig.side,
                        type=OrderType.MARKET,
                        qty=0.0,
                    )
                    # Recompute SL/TP using the actual fill price
                    if atr_now <= 0:
                        continue
                    sl, tp = stop_policy.initial_levels(sig.side, bar.open, atr_now)
                    stop_distance = abs(bar.open - sl)
                    lots = sizer.lots(SizingInputs(
                        equity=equity, stop_distance_price=stop_distance,
                        contract_size=contract,
                    ))
                    if lots <= 0:
                        continue
                    proposed_risk = lots * stop_distance * contract
                    decision = gates.evaluate(bar.ts, equity, proposed_risk, state)
                    if decision == GateDecision.KILL:
                        is_killed = True
                        break
                    if decision == GateDecision.VETO:
                        continue
                    order.qty = lots
                    fill = fill_sim.fill_market(order, bar)
                    equity -= fill.commission
                    pos = Position(
                        strategy=sig.strategy, side=sig.side, qty=lots,
                        entry_price=fill.price, entry_ts=bar.ts,
                        sl_price=sl, tp_price=tp,
                        initial_sl_price=sl, high_water_mark=fill.price,
                    )
                    positions.append(pos)
                    fills_log.append({
                        "ts": bar.ts, "kind": "open", "side": sig.side.value,
                        "qty": lots, "price": fill.price,
                        "commission": fill.commission, "strategy": sig.strategy,
                    })
                    state.on_trade_day(day)
                pending = []

            # 2) Update open positions: SL/TP checks, then trailing
            still_open: list[Position] = []
            for pos in positions:
                hit = fill_sim.maybe_fill_sl_tp(pos.side, pos.sl_price, pos.tp_price, bar)
                if hit is not None:
                    sign = 1 if pos.side is Side.LONG else -1
                    gross = sign * (hit.fill_price - pos.entry_price) * pos.qty * contract
                    close_commission = pos.qty * cfg.execution.commission_per_lot
                    net = gross - close_commission
                    equity += net
                    r_unit = abs(pos.entry_price - (pos.initial_sl_price or pos.entry_price))
                    r_mult = (sign * (hit.fill_price - pos.entry_price) / r_unit) if r_unit > 0 else 0.0
                    trades.append({
                        "entry_ts": pos.entry_ts, "exit_ts": bar.ts,
                        "side": pos.side.value, "strategy": pos.strategy,
                        "qty": pos.qty, "entry_price": pos.entry_price,
                        "exit_price": hit.fill_price, "reason": hit.reason,
                        "pnl": net, "r_multiple": r_mult,
                    })
                    fills_log.append({
                        "ts": bar.ts, "kind": "close", "side": pos.side.value,
                        "qty": pos.qty, "price": hit.fill_price,
                        "commission": close_commission, "strategy": pos.strategy,
                    })
                    if isinstance(sizer, FractionalKelly):
                        sizer.update_stats([r_mult])
                    continue

                # Trailing update (uses bar.close as mark)
                new_sl, _ = stop_policy.update(pos, bar.close, atr_now)
                if new_sl is not None:
                    pos.sl_price = new_sl
                if pos.side is Side.LONG:
                    pos.high_water_mark = max(pos.high_water_mark or pos.entry_price, bar.high)
                else:
                    pos.high_water_mark = min(pos.high_water_mark or pos.entry_price, bar.low)
                still_open.append(pos)
            positions = still_open

            # 3) Equity mark-to-market and curve point
            unrealised = sum(p.unrealized_pnl(bar.close, contract) for p in positions)
            mtm = equity + unrealised
            equity_pts.append((bar.ts, mtm))
            state.on_equity_update(mtm)

            if is_killed:
                # No new orders after kill, but we keep marking and let trailing close out
                continue

            # 4) Filters
            if cfg.filters.session.enabled and not _in_session(bar.ts, cfg.filters.session.windows_utc):
                continue
            if cfg.filters.regime.enabled and atr_p_now > 0 and (
                atr_p_now < cfg.filters.regime.atr_pct_min
                or atr_p_now > cfg.filters.regime.atr_pct_max
            ):
                continue
            if len(positions) >= cfg.risk.max_concurrent_positions:
                continue

            # 5) Strategies -> signals (signals are timestamped at this bar)
            if i + 1 >= len(ts_index):
                continue  # no next bar to fill on
            bars_so_far = bars.iloc[: i + 1]
            for strat in strategies:
                sigs = strat.generate(bars_so_far)
                for s in sigs:
                    if s.ts == bar.ts:
                        pending.append(_PendingOrder(signal=s, estimated_entry=bar.close))
                        break  # one position per side per bar for v1

        # Final mark-to-market: flatten remaining positions at last close
        if positions:
            last_bar = _to_bar(ts_index[-1], bars.iloc[-1])
            for pos in positions:
                sign = 1 if pos.side is Side.LONG else -1
                gross = sign * (last_bar.close - pos.entry_price) * pos.qty * contract
                close_commission = pos.qty * cfg.execution.commission_per_lot
                net = gross - close_commission
                equity += net
                trades.append({
                    "entry_ts": pos.entry_ts, "exit_ts": last_bar.ts,
                    "side": pos.side.value, "strategy": pos.strategy,
                    "qty": pos.qty, "entry_price": pos.entry_price,
                    "exit_price": last_bar.close, "reason": "eod",
                    "pnl": net, "r_multiple": 0.0,
                })
            positions = []

        equity_curve = pd.Series(
            data=[v for _, v in equity_pts],
            index=pd.DatetimeIndex([t for t, _ in equity_pts], tz=ts_index.tz),
            name="equity",
        )
        trades_df = pd.DataFrame(trades)
        fills_df = pd.DataFrame(fills_log)
        if not equity_curve.empty:
            daily_pnl = equity_curve.resample("1D").last().diff().fillna(0.0).rename("daily_pnl")
        else:
            daily_pnl = pd.Series(dtype=float, name="daily_pnl")

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            fills=fills_df,
            daily_pnl=daily_pnl,
            metadata={
                "profile": cfg.profile,
                "starting_equity": cfg.account.starting_equity,
                "final_equity": float(equity),
                "n_trades": len(trades_df),
                "killed": float(state.killed),
                "kill_reason": state.kill_reason or "",
            },
        )


def _to_bar(ts: pd.Timestamp | datetime, row: pd.Series) -> Bar:
    ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return Bar(
        ts=ts_py,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]) if "volume" in row.index and not pd.isna(row["volume"]) else 0.0,
    )

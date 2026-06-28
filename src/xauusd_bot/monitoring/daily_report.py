"""Daily PnL summary text. Pure function -- formats a string for Telegram/email."""
from __future__ import annotations

from datetime import date

import pandas as pd


def build_summary(day: date, fills: pd.DataFrame, equity_curve: pd.Series) -> str:
    """Compose a daily summary suitable for Telegram (<4096 chars)."""
    if equity_curve.empty:
        return f"[{day}] no equity data"

    day_curve = equity_curve[
        (equity_curve.index.date == day) if hasattr(equity_curve.index, "date")
        else equity_curve.index.normalize().date == day
    ]
    if day_curve.empty:
        return f"[{day}] no equity data for that day"

    open_eq = float(day_curve.iloc[0])
    close_eq = float(day_curve.iloc[-1])
    high_eq = float(day_curve.max())
    low_eq = float(day_curve.min())
    pnl = close_eq - open_eq
    pct = (close_eq / open_eq - 1.0) * 100.0 if open_eq > 0 else 0.0

    if fills is None or fills.empty:
        n_trades = 0
        n_open = 0
        n_close = 0
    else:
        ts_col = pd.to_datetime(fills["ts"]) if "ts" in fills.columns else pd.Series(dtype="datetime64[ns]")
        same_day = fills[ts_col.dt.date == day]
        n_trades = len(same_day)
        n_open = int((same_day.get("kind", pd.Series([], dtype=str)) == "open").sum()) if not same_day.empty else 0
        n_close = int((same_day.get("kind", pd.Series([], dtype=str)) == "close").sum()) if not same_day.empty else 0

    sign = "+" if pnl >= 0 else "-"
    return (
        f"[{day}] PnL {sign}${abs(pnl):,.2f} ({pct:+.2f}%) | "
        f"open ${open_eq:,.2f} -> close ${close_eq:,.2f} "
        f"| high ${high_eq:,.2f} / low ${low_eq:,.2f} "
        f"| fills {n_trades} (open {n_open} / close {n_close})"
    )

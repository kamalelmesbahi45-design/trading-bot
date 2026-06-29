"""OHLCV data layer for the macro agent.

Two sources, in order of preference:
    1. Live yfinance pull, cached per (symbol, interval, day) to parquet.
    2. Pre-existing on-disk panel under ``data/cache/multi`` (CSV close-only),
       used as a fallback so the agent still runs offline / in restricted
       environments. Close-only mode synthesises plausible OHLC from the close
       so indicators that need highs/lows don't crash.

All output frames use a tz-aware UTC index named ``ts`` and columns
``open,high,low,close,volume``.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_bot.agent.universe import MACRO_CONTEXT, UNIVERSE_BY_TICKER

_OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class DataConfig:
    cache_dir: Path = Path("data/cache/agent")
    csv_panel_dir: Path = Path("data/cache/multi")
    interval: str = "1d"        # "1d" or "1h"
    lookback_days: int = 400    # enough for 12m TSMOM + warm-up
    allow_network: bool = True


def _empty_ohlcv() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="ts", tz="UTC")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in _OHLCV}, index=idx)


def _synth_ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    """Build a plausible OHLCV frame from a close-only series.

    Used only when the live feed is unavailable and the on-disk panel ships
    closes only. Highs/lows are inflated by the recent realised vol so ATR
    estimates remain in a sane range; volume is set to NaN so anything that
    relies on it stays honest.
    """
    close = close.dropna()
    if close.empty:
        return _empty_ohlcv()
    ret = close.pct_change()
    vol = ret.rolling(20, min_periods=5).std().fillna(ret.std() or 1e-3)
    span = (vol * close).clip(lower=close * 1e-4)
    high = close + span.fillna(0.0)
    low = (close - span.fillna(0.0)).clip(lower=close * 0.999)
    op = close.shift(1).fillna(close)
    out = pd.DataFrame(
        {
            "open": op.to_numpy(dtype="float64"),
            "high": high.to_numpy(dtype="float64"),
            "low": low.to_numpy(dtype="float64"),
            "close": close.to_numpy(dtype="float64"),
            "volume": np.full(len(close), np.nan, dtype="float64"),
        },
        index=pd.DatetimeIndex(close.index, name="ts", tz="UTC")
        if close.index.tz is not None
        else pd.DatetimeIndex(close.index, name="ts").tz_localize("UTC"),
    )
    return out


def _load_csv_close(panel_dir: Path, yahoo: str) -> pd.Series:
    """Best-effort close load from the existing research/panel CSV layout."""
    if not panel_dir.exists():
        return pd.Series(dtype="float64")
    # try direct match, then variants
    candidates = [
        panel_dir / f"{yahoo}.csv",
        panel_dir / f"{yahoo.upper()}.csv",
        panel_dir / f"{yahoo}_raw.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        if "date" in cols and "close" in cols:
            idx = pd.to_datetime(df[cols["date"]], utc=True)
            return pd.Series(df[cols["close"]].astype("float64").to_numpy(),
                             index=idx).sort_index()
        if "snapped_at" in cols and "price" in cols:
            idx = pd.to_datetime(df[cols["snapped_at"]], utc=True).normalize()
            s = pd.Series(df[cols["price"]].astype("float64").to_numpy(), index=idx)
            return s[~s.index.duplicated(keep="last")].sort_index()
    return pd.Series(dtype="float64")


def _normalise_yf(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_ohlcv()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={c: str(c).lower().replace(" ", "_") for c in raw.columns})
    keep = [c for c in _OHLCV if c in raw.columns]
    out = raw[keep].astype("float64").copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "ts"
    return out


def _fetch_yf(symbol: str, lookback_days: int, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception:
        return _empty_ohlcv()
    end = datetime.now(UTC)
    start = end - timedelta(days=lookback_days)
    try:
        raw = yf.download(
            symbol, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception:
        return _empty_ohlcv()
    return _normalise_yf(raw)


def load_asset(symbol_yf: str, cfg: DataConfig) -> pd.DataFrame:
    """Load one asset's OHLCV. Tries cache -> network -> CSV fallback."""
    cache_key = f"{symbol_yf.replace('/', '_').replace('=', '_').replace('^', '_')}"
    cache_file = cfg.cache_dir / cfg.interval / f"{cache_key}.parquet"
    today = datetime.now(UTC).date()

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            # consider stale if last bar older than 2 days (daily) / 12h (hourly)
            if not df.empty:
                last = df.index[-1].date() if hasattr(df.index[-1], "date") else None
                staleness = (today - last).days if last else 999
                stale_limit = 2 if cfg.interval == "1d" else 1
                if staleness <= stale_limit or not cfg.allow_network:
                    return df
        except Exception:
            pass

    df = _empty_ohlcv()
    if cfg.allow_network:
        df = _fetch_yf(symbol_yf, cfg.lookback_days, cfg.interval)

    if df.empty:
        # CSV fallback (close-only): synthesise OHLCV
        close = _load_csv_close(cfg.csv_panel_dir, symbol_yf)
        if not close.empty:
            df = _synth_ohlcv_from_close(close.iloc[-cfg.lookback_days * 3:])

    if not df.empty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            df.to_parquet(cache_file)
    return df


def load_universe(cfg: DataConfig, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return ``{internal_ticker: ohlcv_frame}``. Missing assets are skipped."""
    selected = tickers or list(UNIVERSE_BY_TICKER.keys())
    out: dict[str, pd.DataFrame] = {}
    for tk in selected:
        a = UNIVERSE_BY_TICKER.get(tk)
        if a is None:
            continue
        df = load_asset(a.yahoo, cfg)
        if not df.empty and len(df) >= 60:
            out[tk] = df
    return out


def load_macro_context(cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Load read-only macro context series for the regime model."""
    out: dict[str, pd.DataFrame] = {}
    for name, sym in MACRO_CONTEXT.items():
        df = load_asset(sym, cfg)
        if not df.empty:
            out[name] = df
    return out


def close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align a dict of OHLCV frames into a wide close-only panel (UTC daily)."""
    if not frames:
        return pd.DataFrame()
    series = {tk: df["close"].rename(tk) for tk, df in frames.items()}
    panel = pd.concat(series.values(), axis=1).sort_index()
    panel.columns = list(series.keys())
    # Forward-fill exchange holidays inside each series' lifetime.
    bounds = {c: (panel[c].first_valid_index(), panel[c].last_valid_index())
              for c in panel.columns}
    panel = panel.ffill()
    for c, (first, last) in bounds.items():
        if first is not None:
            panel.loc[panel.index < first, c] = np.nan
        if last is not None:
            panel.loc[panel.index > last, c] = np.nan
    return panel.astype("float64")

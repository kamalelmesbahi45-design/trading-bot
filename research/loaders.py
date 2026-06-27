"""Load and align the real-data palette into pandas Series indexed by UTC date.

All loaders return tz-aware UTC indices, daily frequency, and forward-fill across
business-day gaps so they can be joined cleanly to the gold series.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path("data/cache/macro")


def _to_utc_daily(s: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).normalize()
    out = pd.Series(s.values, index=idx).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.asfreq("D").ffill()


def load_xauusd_daily(path: Path | None = None,
                       min_price: float = 1000.0,
                       max_price: float = 5000.0) -> pd.DataFrame:
    """CUPID-l XAUUSD daily: Date,Open,High,Low,Close,Symbol -> OHLC.

    Drops bad rows where any OHLC value is outside [min_price, max_price].
    The source has a handful of dirty rows around 2019-03/2019-09 with
    prices in the tens of thousands (likely raw broker feed errors).
    """
    p = path or CACHE / "xauusd_daily.csv"
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df.rename(columns={c: c.lower() for c in df.columns}).set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index, tz="UTC").normalize()
    out = df[["open", "high", "low", "close"]].astype("float64").copy()
    in_range = (out >= min_price).all(axis=1) & (out <= max_price).all(axis=1)
    out = out.loc[in_range]
    # Iteratively drop big single-day jumps (data glitches). Real gold rarely > 7%/day.
    for _ in range(5):
        ret = out["close"].pct_change().abs()
        bad = ret > 0.07
        if not bad.any():
            break
        out = out.loc[~bad]
    out["volume"] = 0.0
    return out


def load_xauusd_h1(path: Path | None = None) -> pd.DataFrame:
    """ForexBot H1 XAUUSD."""
    p = path or CACHE / "xauusd_h1.csv"
    df = pd.read_csv(p)
    df = df.rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype("float64").copy()


def load_fred_two_col(path: Path, value_col: str, date_col: str = "DATE") -> pd.Series:
    """Generic FRED-style CSV: <date_col>,<value_col>."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dc = cols.get(date_col.lower()) or cols.get("observation_date") or cols.get("date")
    if dc is None:
        raise ValueError(f"no date column in {path} cols={df.columns.tolist()}")
    df[dc] = pd.to_datetime(df[dc], errors="coerce")
    df = df.dropna(subset=[dc]).sort_values(dc)
    vc = cols.get(value_col.lower()) or [c for c in df.columns if c != dc][0]
    s = pd.Series(pd.to_numeric(df[vc], errors="coerce").values,
                  index=pd.DatetimeIndex(df[dc].values))
    return _to_utc_daily(s.dropna())


def load_dgs10() -> pd.Series:
    """US 10y nominal yield %."""
    return load_fred_two_col(CACHE / "dgs10.csv", "DGS10", date_col="observation_date")


def load_dfii10() -> pd.Series:
    """US 10y TIPS (real) yield %. Coverage: 2003 - 2021."""
    return load_fred_two_col(CACHE / "dfii10.csv", "DFII10", date_col="date")


def load_t10yie() -> pd.Series:
    """US 10y breakeven inflation %."""
    return load_fred_two_col(CACHE / "t10yie.csv", "T10YIE", date_col="DATE")


def load_dxy() -> pd.Series:
    """USD index. The nepse CSV is series_name,date,value,units,source."""
    df = pd.read_csv(CACHE / "dxy.csv", parse_dates=["date"])
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.DatetimeIndex(df["date"].values)).dropna()
    return _to_utc_daily(s)


def load_vix() -> pd.Series:
    """VIX close. Format: skip first metadata rows, then Date,Ticker,Open,High,Low,Close."""
    raw = pd.read_csv(CACHE / "vix.csv", skiprows=2)
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date")
    s = pd.Series(pd.to_numeric(raw["Close"], errors="coerce").values,
                  index=pd.DatetimeIndex(raw["Date"].values)).dropna()
    return _to_utc_daily(s)


def load_all() -> dict[str, pd.Series | pd.DataFrame]:
    """Return everything aligned, ready for joining."""
    return {
        "xauusd_d": load_xauusd_daily(),
        "xauusd_h1": load_xauusd_h1(),
        "dgs10": load_dgs10(),
        "dfii10": load_dfii10(),
        "t10yie": load_t10yie(),
        "dxy": load_dxy(),
        "vix": load_vix(),
    }


def daily_join(parts: dict[str, pd.Series | pd.DataFrame]) -> pd.DataFrame:
    """Build a daily macro frame aligned to XAUUSD daily bars."""
    g = parts["xauusd_d"].copy()
    g.index = g.index.normalize()
    out = g.add_prefix("gold_")
    for name in ("dgs10", "dfii10", "t10yie", "dxy", "vix"):
        s = parts.get(name)
        if s is None or s.empty:
            continue
        idx = s.index.normalize() if hasattr(s.index, "normalize") else s.index
        s = pd.Series(s.values, index=idx, name=name)
        out = out.join(s, how="left")
    out = out.ffill()
    return out

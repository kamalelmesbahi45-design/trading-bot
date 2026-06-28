"""Multi-asset daily panel loader.

Sources two on-disk formats committed from public datasets:
    * ETF "<ticker>.us.txt": Date,Open,High,Low,Close,Volume,OpenInt  (Kaggle huge-stock)
    * BTC "snapped_at,price,market_cap,total_volume"                  (CoinGecko export)

We build one aligned daily CLOSE panel. Each asset keeps its own inception; the
panel is the union of dates, forward-filled across exchange holidays but left
NaN before an asset's first print (so momentum never sees pre-inception data).

ETF tickers are mapped to the economic exposure they proxy so the dashboard and
daily briefing speak in market terms (FXE -> EUR, GLD -> Gold, ...). Some FX
ETFs track the foreign currency vs USD; a couple are USD-strength plays. The
`inverse` flag marks series whose sign must be flipped to express the canonical
"USD up = positive" or quote convention; we keep raw price here and let the
signal layer decide -- momentum is sign-agnostic to a constant flip, but we
record it for labelling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AssetSpec:
    ticker: str          # file ticker (FXE, GLD, BTC, ...)
    label: str           # human label (EUR, Gold, Bitcoin, ...)
    asset_class: str     # fx | commodity | equity | rates | real_assets | crypto
    cluster: str         # correlation cluster for diversification accounting


# Diversified universe. Clusters chosen so position-level risk is spread across
# genuinely different drivers (USD, risk-on growth, safe havens, real assets).
UNIVERSE: list[AssetSpec] = [
    AssetSpec("FXE", "EUR",          "fx",          "usd_short"),
    AssetSpec("FXB", "GBP",          "fx",          "usd_short"),
    AssetSpec("FXA", "AUD",          "fx",          "risk_on_fx"),
    AssetSpec("FXF", "CHF",          "fx",          "safe_haven"),
    AssetSpec("FXY", "JPY",          "fx",          "safe_haven"),
    AssetSpec("UUP", "USD_index",    "fx",          "usd_long"),
    AssetSpec("GLD", "Gold",         "commodity",   "metals"),
    AssetSpec("SLV", "Silver",       "commodity",   "metals"),
    AssetSpec("GDX", "GoldMiners",   "commodity",   "metals"),
    AssetSpec("USO", "Oil",          "commodity",   "energy"),
    AssetSpec("UNG", "NatGas",       "commodity",   "energy"),
    AssetSpec("DBA", "Agriculture",  "commodity",   "agri"),
    AssetSpec("SPY", "SP500",        "equity",      "risk_on_eq"),
    AssetSpec("EEM", "EmergMkts",    "equity",      "risk_on_eq"),
    AssetSpec("VNQ", "RealEstate",   "real_assets", "real_assets"),
    AssetSpec("TLT", "Treasuries",   "rates",       "duration"),
    AssetSpec("BTC", "Bitcoin",      "crypto",      "crypto"),
]

UNIVERSE_BY_TICKER: dict[str, AssetSpec] = {a.ticker: a for a in UNIVERSE}


def _load_etf(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    close_col = cols.get("close")
    if date_col is None or close_col is None:
        raise ValueError(f"{path.name}: expected Date/Close columns, got {list(df.columns)}")
    s = pd.Series(df[close_col].to_numpy(dtype="float64"),
                  index=pd.to_datetime(df[date_col]).dt.tz_localize("UTC"))
    return s.sort_index()


def _load_btc(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    # CoinGecko: snapped_at,price,market_cap,total_volume
    if "snapped_at" in df.columns and "price" in df.columns:
        idx = pd.to_datetime(df["snapped_at"], utc=True).dt.normalize()
        s = pd.Series(df["price"].to_numpy(dtype="float64"), index=idx)
        return s[~s.index.duplicated(keep="last")].sort_index()
    # Fallback: Yahoo format Date,Open,...,Close
    cols = {c.lower(): c for c in df.columns}
    idx = pd.to_datetime(df[cols["date"]]).dt.tz_localize("UTC")
    return pd.Series(df[cols["close"]].to_numpy(dtype="float64"), index=idx).sort_index()


def load_close_panel(data_dir: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    """Build an aligned daily close panel. Columns are tickers, index is UTC dates.

    Pre-inception values stay NaN; exchange-holiday gaps inside an asset's life
    are forward-filled (a stale close is a fair mark for a non-trading day).
    """
    selected = tickers or [a.ticker for a in UNIVERSE]
    series: dict[str, pd.Series] = {}
    for tk in selected:
        if tk == "BTC":
            p = data_dir / "BTC_raw.csv"
            if p.exists():
                series[tk] = _load_btc(p)
            continue
        p = data_dir / f"{tk}.csv"
        if p.exists():
            series[tk] = _load_etf(p)

    if not series:
        return pd.DataFrame()

    panel = pd.DataFrame(series).sort_index()
    # Record each asset's true first/last real print BEFORE filling.
    bounds = {c: (panel[c].first_valid_index(), panel[c].last_valid_index())
              for c in panel.columns}
    panel = panel.ffill()
    # Re-mask: anything before first real print or after last real print is NaN,
    # so a stale close is never used outside the asset's actual trading life.
    for c, (first, last) in bounds.items():
        if first is not None:
            panel.loc[panel.index < first, c] = pd.NA
        if last is not None:
            panel.loc[panel.index > last, c] = pd.NA
    return panel.astype("float64")


def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. First valid return per column is NaN."""
    return panel.pct_change()

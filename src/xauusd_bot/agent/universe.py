"""Tradable universe for the macro agent.

Mirrors ``research.panel`` but adds the Yahoo symbol so we can pull live OHLCV,
plus per-asset overrides (typical bid/ask cost, contract notional hint). Assets
are grouped into correlation clusters so the risk layer can cap how much risk
sits in any single driver (e.g. don't stack EUR + GBP + AUD as if they were
independent USD-short bets).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    ticker: str           # internal short symbol
    label: str            # human-readable
    yahoo: str            # Yahoo Finance symbol
    asset_class: str      # fx | commodity | equity | rates | real_assets | crypto
    cluster: str          # correlation cluster
    cost_bps: float = 5.0 # round-trip transaction cost estimate
    min_adv_pct: float = 0.0  # min size as % of equity to bother trading


# Liquid, freely-available proxies. ETFs are used for FX/commodity/rates so the
# whole stack runs on free Yahoo data; on a real broker you would swap in
# futures / spot pairs by changing the Yahoo symbol only.
UNIVERSE: list[Asset] = [
    # --- FX (vs USD) ---
    Asset("EUR", "EUR/USD",  "FXE",       "fx",         "usd_short"),
    Asset("GBP", "GBP/USD",  "FXB",       "fx",         "usd_short"),
    Asset("AUD", "AUD/USD",  "FXA",       "fx",         "risk_on_fx"),
    Asset("JPY", "JPY/USD",  "FXY",       "fx",         "safe_haven"),
    Asset("CHF", "CHF/USD",  "FXF",       "fx",         "safe_haven"),
    Asset("DXY", "USD Index","UUP",       "fx",         "usd_long"),
    # --- Metals / commodities ---
    Asset("GLD", "Gold",     "GLD",       "commodity",  "metals"),
    Asset("SLV", "Silver",   "SLV",       "commodity",  "metals"),
    Asset("GDX", "GoldMiner","GDX",       "commodity",  "metals"),
    Asset("WTI", "Crude Oil","USO",       "commodity",  "energy"),
    Asset("NG",  "Nat Gas",  "UNG",       "commodity",  "energy"),
    Asset("AGR", "Agri",     "DBA",       "commodity",  "agri"),
    # --- Equity ---
    Asset("SPX", "S&P 500",  "SPY",       "equity",     "risk_on_eq"),
    Asset("NDX", "Nasdaq100","QQQ",       "equity",     "risk_on_eq"),
    Asset("EEM", "EmergMkt", "EEM",       "equity",     "risk_on_eq"),
    # --- Rates / real assets ---
    Asset("TLT", "20Y UST",  "TLT",       "rates",      "duration"),
    Asset("HYG", "HY Credit","HYG",       "rates",      "credit"),
    Asset("VNQ", "REITs",    "VNQ",       "real_assets","real_assets"),
    # --- Crypto ---
    Asset("BTC", "Bitcoin",  "BTC-USD",   "crypto",     "crypto", 15.0),
    Asset("ETH", "Ethereum", "ETH-USD",   "crypto",     "crypto", 20.0),
]

UNIVERSE_BY_TICKER: dict[str, Asset] = {a.ticker: a for a in UNIVERSE}


# Macro context tickers (read-only -- used by the regime model, not traded).
MACRO_CONTEXT: dict[str, str] = {
    "DXY":   "DX-Y.NYB",  # dollar index
    "VIX":   "^VIX",
    "US10Y": "^TNX",      # 10y yield (Yahoo quotes as yield * 10)
    "US02Y": "^IRX",      # 13w T-bill stands in when 2y not available
    "SPX":   "^GSPC",
    "GOLD":  "GC=F",
    "OIL":   "CL=F",
    "HYG":   "HYG",       # high-yield ETF (credit spread proxy vs TLT)
}


def cluster_of(ticker: str) -> str:
    a = UNIVERSE_BY_TICKER.get(ticker)
    return a.cluster if a else "unknown"


def label_of(ticker: str) -> str:
    a = UNIVERSE_BY_TICKER.get(ticker)
    return a.label if a else ticker

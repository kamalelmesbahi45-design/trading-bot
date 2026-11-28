"""
Data layer. Every loader returns tidy pandas objects and NEVER raises: on failure it
returns None and records the error, so one dead source never kills the daily run.

Sources (all free, no API key):
  * Yahoo Finance via the open-source `yfinance` package (github.com/ranaroussi/yfinance)
  * FRED (St. Louis Fed) CSV endpoint
  * CFTC Commitments of Traders via the CFTC public Socrata API
  * Economic calendar: the public ForexFactory weekly JSON feed (nfs.faireconomy.media)
  * Fallback "mirror" mode: public GitHub-hosted CSV datasets (used when Yahoo is unreachable)
"""
from __future__ import annotations

import io
import datetime as dt
import pandas as pd
import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

ERRORS: list[str] = []
UA = {"User-Agent": "Mozilla/5.0 (edge-dashboard; research use)"}


def _err(msg: str):
    ERRORS.append(msg)
    print("[data] " + msg)


def _get(url: str, timeout: int = 30):
    if requests is None:
        raise RuntimeError("requests not installed")
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            df[c] = df["Close"]
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    # repair bad bars: missing/zero O/H/L -> fall back to close
    for c in ["Open", "High", "Low"]:
        bad = df[c].isna() | (df[c] <= 0)
        df.loc[bad, c] = df.loc[bad, "Close"]
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    return df


def has_real_ohlc(df: pd.DataFrame) -> bool:
    """True if the series has genuine intraday range (not close-only)."""
    tail = df.tail(250)
    return bool(((tail["High"] - tail["Low"]) > 0).mean() > 0.8)


# --------------------------------------------------------------------------- Yahoo
def load_yahoo(ticker: str, years: int = 20, tries: int = 3) -> pd.DataFrame | None:
    """Daily OHLC from Yahoo. Retries with back-off (Yahoo rate-limits cloud IPs at times)."""
    import time
    if globals().get("OFFLINE", False):
        return None
    for k in range(tries):
        try:
            import yfinance as yf
            start = (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 400)).isoformat()
            df = yf.download(ticker, start=start, interval="1d", auto_adjust=False,
                             progress=False, threads=False)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return _clean_ohlc(df)
        except Exception as e:
            _err(f"yahoo {ticker} try {k + 1}: {e}")
        if k < tries - 1:
            time.sleep(3 * (k + 1))
    _err(f"yahoo {ticker}: no data after {tries} tries")
    return None


def load_yahoo_intraday(ticker: str) -> pd.DataFrame | None:
    """Hourly bars for the last ~2 years (Yahoo limit) -> intraday profile."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="730d", interval="1h", auto_adjust=False,
                         progress=False, threads=False)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).title() for c in df.columns]
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception as e:
        _err(f"yahoo intraday {ticker}: {e}")
        return None


# --------------------------------------------------------------------------- FRED
_FRED_DOWN = False


def load_fred(series: str, tries: int = 2) -> pd.Series | None:
    """FRED sometimes refuses cloud runners: after one total failure we stop trying (circuit breaker)."""
    global _FRED_DOWN
    if _FRED_DOWN:
        return None
    start = (dt.date.today() - dt.timedelta(days=3 * 365)).isoformat()
    for k in range(tries):
        try:
            r = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}", timeout=25)
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            s = pd.to_numeric(df["value"], errors="coerce")
            s.index = pd.to_datetime(df["date"])
            return s.dropna()
        except Exception as e:
            _err(f"fred {series} try {k + 1}: {e}")
    _FRED_DOWN = True
    return None


# --------------------------------------------------------------------------- CFTC COT
def load_cot(code: str, weeks: int = 160) -> pd.DataFrame | None:
    """Legacy futures-only report. Net non-commercial (speculator) position as % of OI."""
    try:
        url = ("https://publicreporting.cftc.gov/resource/6dca-aqww.json"
               f"?cftc_contract_market_code={code}"
               "&$order=report_date_as_yyyy_mm_dd%20DESC"
               f"&$limit={weeks}")
        rows = _get(url).json()
        if not rows:
            _err(f"cot {code}: no rows")
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        for c in ["noncomm_positions_long_all", "noncomm_positions_short_all", "open_interest_all"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["net_pct_oi"] = 100 * df["net"] / df["open_interest_all"]
        return df.set_index("date").sort_index()[["net", "net_pct_oi", "open_interest_all"]]
    except Exception as e:
        _err(f"cot {code}: {e}")
        return None


# --------------------------------------------------------------------------- Calendar
def load_calendar() -> list[dict]:
    """This week's macro events (ForexFactory public JSON feed)."""
    try:
        rows = _get("https://nfs.faireconomy.media/ff_calendar_thisweek.json").json()
        out = []
        for r in rows:
            out.append(dict(title=r.get("title"), country=r.get("country"),
                            date=r.get("date"), impact=r.get("impact"),
                            forecast=r.get("forecast"), previous=r.get("previous")))
        return out
    except Exception as e:
        _err(f"calendar: {e}")
        return []


# --------------------------------------------------------------------------- GitHub mirrors
RAW = "https://raw.githubusercontent.com"
MIRRORS = {
    "SPX500": ("ohlc", f"{RAW}/Mario-SO/ohlcv/main/data/sp500.csv"),
    "XAUUSD": ("ohlc", f"{RAW}/Mario-SO/ohlcv/main/data/gold.csv"),
    "BTCUSD": ("ohlc", f"{RAW}/Mario-SO/ohlcv/main/data/btc.csv"),
    "ETHUSD": ("ohlc", f"{RAW}/Mario-SO/ohlcv/main/data/eth.csv"),
    "WTI":    ("close", f"{RAW}/datasets/oil-prices/main/data/wti-daily.csv"),
    "VIX":    ("ohlc", f"{RAW}/datasets/finance-vix/main/data/vix-daily.csv"),
}
FX_MIRROR = f"{RAW}/datasets/exchange-rates/main/data/daily.csv"   # FRED H.10, foreign ccy per USD
FX_MAP = {"EURUSD": ("Euro", True), "GBPUSD": ("United Kingdom", True), "USDJPY": ("Japan", False),
          "AUDUSD": ("Australia", True), "USDCAD": ("Canada", False), "USDCHF": ("Switzerland", False)}
BTC_1M = f"{RAW}/ff137/bitstamp-btcusd-minute-data/main/data/updates/btcusd_bitstamp_1min_latest.csv"

_cache: dict = {}


def _read_csv(path_or_url: str) -> pd.DataFrame:
    if path_or_url in _cache:
        return _cache[path_or_url]
    if path_or_url.startswith("http"):
        df = pd.read_csv(io.StringIO(_get(path_or_url, timeout=120).text))
    else:
        df = pd.read_csv(path_or_url)
    _cache[path_or_url] = df
    return df


def load_mirror(mid: str, local_dir: str | None = None) -> pd.DataFrame | None:
    """Fallback loader. local_dir lets you point at pre-downloaded copies."""
    try:
        def src(url):
            if local_dir:
                import os
                name = {"sp500.csv": "sp500.csv", "gold.csv": "gold.csv", "btc.csv": "btc.csv",
                        "eth.csv": "eth.csv", "wti-daily.csv": "wti.csv", "vix-daily.csv": "vix.csv",
                        "daily.csv": "fx.csv",
                        "btcusd_bitstamp_1min_latest.csv": "btc1m.csv"}[url.rsplit("/", 1)[1]]
                p = os.path.join(local_dir, name)
                if os.path.exists(p):
                    return p
            return url

        if mid in FX_MAP:
            country, invert = FX_MAP[mid]
            fx = _read_csv(src(FX_MIRROR))
            s = fx[fx["Country"] == country].copy()
            s["v"] = pd.to_numeric(s.iloc[:, 2], errors="coerce")
            s = s.dropna(subset=["v"])
            close = (1.0 / s["v"]) if invert else s["v"]
            close.index = pd.to_datetime(s["Date"])
            df = pd.DataFrame({"Close": close.values}, index=close.index)
            return _clean_ohlc(df)
        if mid == "DXY":
            return _synthetic_dxy(src)
        if mid not in MIRRORS:
            return None
        kind, url = MIRRORS[mid]
        raw = _read_csv(src(url))
        raw.columns = [c.strip().title() for c in raw.columns]
        dcol = "Date"
        raw.index = pd.to_datetime(raw[dcol])
        if kind == "close":
            raw = raw.rename(columns={"Price": "Close"})
            df = _clean_ohlc(raw[["Close"]])
        else:
            df = _clean_ohlc(raw[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]])
        if mid == "BTCUSD":
            df = _extend_btc(df, src)
        return df
    except Exception as e:
        _err(f"mirror {mid}: {e}")
        return None


def _synthetic_dxy(src) -> pd.DataFrame | None:
    """ICE US Dollar Index rebuilt from its official 6-currency formula."""
    fx = _read_csv(src(FX_MIRROR))
    piv = fx.pivot_table(index="Date", columns="Country", values=fx.columns[2], aggfunc="last")
    piv.index = pd.to_datetime(piv.index)
    need = ["Euro", "Japan", "United Kingdom", "Canada", "Sweden", "Switzerland"]
    piv = piv[need].apply(pd.to_numeric, errors="coerce").dropna()
    eurusd, gbpusd = 1 / piv["Euro"], 1 / piv["United Kingdom"]
    dxy = (50.14348112 * eurusd ** -0.576 * piv["Japan"] ** 0.136 * gbpusd ** -0.119
           * piv["Canada"] ** 0.091 * piv["Sweden"] ** 0.042 * piv["Switzerland"] ** 0.036)
    return _clean_ohlc(pd.DataFrame({"Close": dxy}))


def btc_minutes(src=lambda u: u) -> pd.DataFrame | None:
    try:
        m = _read_csv(src(BTC_1M))
        m.index = pd.to_datetime(m["timestamp"], unit="s")
        return m[["open", "high", "low", "close", "volume"]].rename(columns=str.title)
    except Exception as e:
        _err(f"btc minutes: {e}")
        return None


def _extend_btc(daily: pd.DataFrame, src) -> pd.DataFrame:
    """Append the fresher Bitstamp 1-minute data (resampled to UTC days) after the daily file ends."""
    m = btc_minutes(src)
    if m is None:
        return daily
    d = m.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    d = d.dropna(subset=["Close"])
    d = d[d.index > daily.index[-1]]
    # drop an incomplete final UTC day (less than ~20h of minutes)
    last_day = m.index[-1].normalize()
    if (m.index[-1] - last_day) < pd.Timedelta(hours=20):
        d = d[d.index < last_day]
    return _clean_ohlc(pd.concat([daily, d]))

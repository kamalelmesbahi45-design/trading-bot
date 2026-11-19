"""
Exercise the LIVE code path (Yahoo/FRED/COT/calendar parsing) without network:
yfinance.download and requests.get are replaced by fakes that return data shaped exactly
like the real services (yfinance 1.x MultiIndex columns, FRED fredgraph CSV, CFTC JSON, FF JSON).
Usage: python tests/test_live_path_mock.py /path/to/rawdata
"""
import sys, os, json, types
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/rawdata"
import yfinance as yf
from engine import data as D

base = pd.read_csv(f"{RAW}/sp500.csv", index_col=0, parse_dates=True)

def fake_download(ticker, start=None, period=None, interval="1d", **kw):
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    if interval == "1h":
        idx = pd.date_range(end="2026-11-07", periods=24 * 400, freq="1h", tz="America/New_York")
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    else:
        idx = base.index[-5000:]
        c = base["Close"].values[-5000:] * (1 + 0.1 * rng.random())
    o = c * (1 + rng.normal(0, 0.002, len(c)))
    h = np.maximum(o, c) * 1.004; l = np.minimum(o, c) * 0.996
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], [ticker]], names=["Price", "Ticker"])
    return pd.DataFrame(np.column_stack([c, c, h, l, o, np.ones(len(c))]), index=pd.DatetimeIndex(idx, name="Date"), columns=cols)

class Resp:
    def __init__(self, text=None, js=None): self.text, self._js = text, js
    def raise_for_status(self): pass
    def json(self): return self._js

def fake_get(url, headers=None, timeout=30):
    if "fredgraph" in url:
        sid = url.split("id=")[1]
        d = pd.bdate_range("2024-01-01", "2026-11-06")
        vals = ["." if i % 50 == 0 else f"{3 + np.sin(i / 40):.2f}" for i in range(len(d))]
        return Resp(text="observation_date," + sid + "\n" + "\n".join(f"{a.date()},{v}" for a, v in zip(d, vals)))
    if "cftc" in url:
        d = pd.date_range(end="2026-11-03", periods=160, freq="7D")
        return Resp(js=[dict(report_date_as_yyyy_mm_dd=x.strftime("%Y-%m-%dT00:00:00.000"), noncomm_positions_long_all=str(100000 + i * 300),
                             noncomm_positions_short_all=str(90000), open_interest_all="500000") for i, x in enumerate(d)])
    if "faireconomy" in url:
        return Resp(js=[dict(title="CPI m/m", country="USD", date="2026-11-12T08:30:00-05:00", impact="High", forecast="0.3%", previous="0.2%")])
    raise RuntimeError("unexpected url " + url)

yf.download = fake_download
D.requests = types.SimpleNamespace(get=fake_get)

from engine.analysis import build
from engine.render import render
doc = build("yahoo", None, intraday=True)
assert len(doc["markets"]) == 15, doc["missing"]
assert not doc["snapshot"]
assert any(i["key"] == "BAMLH0A0HYM2" for i in doc["macro"]["items"])
assert any(m["cot"] for m in doc["markets"]), "COT parse failed"
assert all(m["hourly"] for m in doc["markets"]), "intraday parse failed"
assert doc["calendar"] and doc["calendar"][0]["impact"] == "High"
json.dumps(doc, allow_nan=False)
render(doc, "/tmp/mock_dashboard.html")
print("live path OK:", len(doc["markets"]), "markets,", doc["tests"], "macro", doc["macro"]["regime"], "errors:", doc["errors"][:3])

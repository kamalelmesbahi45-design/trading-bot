"""Downloads the factor data used by daytrade/research6_factors.py into research_data/out/ (run in GitHub Actions)."""
import os, time, io, requests, pandas as pd
os.makedirs("out", exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; edge-desk)"}
log = []
def get(url, path, timeout=90):
    for k in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=timeout); r.raise_for_status()
            open(path, "wb").write(r.content); log.append(f"OK {url} -> {path} ({len(r.content)} bytes)"); return True
        except Exception as e:
            log.append(f"try {k+1} FAIL {url}: {e}"); time.sleep(5)
    return False
get("https://squeezemetrics.com/monitor/static/DIX.csv", "out/DIX.csv")
for s in ["WALCL", "WTREGEN", "RRPONTSYD", "DFII10", "BAMLH0A0HYM2", "T10Y2Y"]:
    get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", f"out/FRED_{s}.csv")
get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "out/fomc_current.html")
for y in range(2014, 2021):
    get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", f"out/fomc_{y}.html")
import yfinance as yf
for t in ["^VIX", "^VIX3M", "^VIX9D", "^GSPC", "^NDX", "^GDAXI", "IEF", "TLT", "^TNX", "DX-Y.NYB", "HYG", "GC=F", "^MOVE"]:
    for k in range(3):
        try:
            df = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False, threads=False)
            if len(df):
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.to_csv(f"out/YF_{t.replace('^','').replace('=','_')}.csv"); log.append(f"OK yahoo {t} {len(df)}"); break
        except Exception as e:
            log.append(f"try {k+1} FAIL yahoo {t}: {e}")
        time.sleep(3)
open("out/LOG.txt", "w").write("\n".join(log)); print("\n".join(log))

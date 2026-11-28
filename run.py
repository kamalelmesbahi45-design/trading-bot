"""
Daily run:  python run.py            (live: Yahoo/FRED/CFTC/calendar, falls back to GitHub mirrors)
            python run.py --source mirror --local rawdata/   (offline demo on pre-downloaded CSVs)
Writes data/latest.json and docs/index.html (GitHub Pages serves /docs).
"""
import argparse, json, os, time
from engine.analysis import build
from engine.render import render
from engine import journal

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="auto", choices=["auto", "yahoo", "mirror"])
ap.add_argument("--local", default=None, help="folder with pre-downloaded mirror CSVs")
ap.add_argument("--no-intraday", action="store_true")
a = ap.parse_args()

t = time.time()
doc = build(a.source, a.local, intraday=not a.no_intraday)
os.makedirs("data", exist_ok=True); os.makedirs("docs", exist_ok=True)
with open("data/latest.json", "w") as fh:
    json.dump(doc, fh, separators=(",", ":"))
if os.path.exists("data/intraday.json"):           # keep the latest day-trading plan on the page
    doc["intraday"] = json.load(open("data/intraday.json"))
if os.path.exists("data/daytrade_log.csv"):
    import csv
    with open("data/daytrade_log.csv") as fh:
        doc["daytrade_log"] = list(csv.DictReader(fh))[-200:]
render(doc, "docs/index.html")
if doc["source"] != "mirror":          # only live runs feed the forward track record
    journal.append(doc["signals"])
os.makedirs("data/history", exist_ok=True)
with open(f"data/history/{doc['generated_at'][:10]}.json", "w") as fh:
    json.dump(dict(generated_at=doc["generated_at"], signals=doc["signals"], macro=doc["macro"]["regime"]), fh)
print(f"done in {time.time()-t:.0f}s | markets={len(doc['markets'])} signals={len(doc['signals'])} "
      f"tests={doc['tests']} missing={doc['missing']}")

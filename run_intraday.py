"""
Intraday run (every 30 minutes on weekdays): rebuild today's day-trading plan for the six systems and
re-render the dashboard. The swing part is reused from the last daily run (data/latest.json).
"""
import json, os, csv
from engine import intraday
from engine.render import render

intra = intraday.build()
os.makedirs("data", exist_ok=True)
with open("data/intraday.json", "w") as fh:
    json.dump(intra, fh, separators=(",", ":"))

# forward track record of day trades: one row per closed trade, written once
LOG = "data/daytrade_log.csv"
seen = set()
if os.path.exists(LOG):
    with open(LOG) as fh:
        seen = {(r["day"], r["system"], r["entry_time"]) for r in csv.DictReader(fh)}
new = []
for s in intra["systems"]:
    for t in (s.get("trades") or ([s["trade"]] if s.get("trade") else [])):
        if t.get("state") == "CLOSED" and t.get("R") is not None:
            key = (s["day"], s["id"], t.get("entry_time") or "")
            if key not in seen:
                new.append(dict(day=s["day"], system=s["id"], market=s["market"], dir=t.get("dir", ""), entry_time=t.get("entry_time") or "",
                                entry=t.get("entry"), exit=t.get("exit"), outcome=t.get("outcome"), R=t["R"]))
if new:
    exists = os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(new)

doc = json.load(open("data/latest.json")) if os.path.exists("data/latest.json") else {}
doc["intraday"] = intra
if os.path.exists(LOG):
    with open(LOG) as fh:
        doc["daytrade_log"] = list(csv.DictReader(fh))[-200:]
render(doc, "docs/index.html")
print("intraday:", [(s["id"], s["status"]) for s in intra["systems"]], "errors:", intra["errors"][:3], "| logged", len(new))

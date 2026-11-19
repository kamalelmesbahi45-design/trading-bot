import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "dashboard")


def render(doc: dict, out_path: str):
    """Inline the data and the charting library -> one self-contained HTML file."""
    tpl = open(os.path.join(ROOT, "template.html"), encoding="utf-8").read()
    lwc = open(os.path.join(ROOT, "vendor", "lightweight-charts.js"), encoding="utf-8").read()
    payload = json.dumps(doc, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = tpl.replace("/*__LWC__*/", lwc.replace("</script", "<\\/script")).replace("/*__DATA__*/null", payload)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

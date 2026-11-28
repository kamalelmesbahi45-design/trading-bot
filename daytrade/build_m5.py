"""Combine Dukascopy M1 CSV chunks (bid) into one UTC M5 parquet file per instrument."""
import glob, sys, os
import pandas as pd

inst, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(src, "*.csv")))
parts = []
for f in files:
    try:
        d = pd.read_csv(f)
        if len(d):
            parts.append(d)
    except Exception as e:
        print("skip", f, e)
df = pd.concat(parts, ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
m5 = df.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
m5 = m5.astype("float32")
os.makedirs(os.path.dirname(out), exist_ok=True)
m5.to_parquet(out, compression="zstd")
print(inst, "m1 rows", len(df), "m5 rows", len(m5), m5.index[0], "->", m5.index[-1])

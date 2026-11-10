# Day-trading research (Dukascopy M5, 2014 → today, 8 markets)

Data: `intraday-data` branch (built by `.github/workflows/intraday-data.yml`).
Run: `M5_DIR=m5 python daytrade/research.py out1` then `R1=out1/results.csv python daytrade/research2.py out2`.

## Findings so far (honest)
* Round 1: 116 pre-declared tests (ORB 5/15/30, Zarattini noise-momentum, London breakout, liquidity sweep).
  **0 pass the false-discovery control.** 5 weak edges (p<0.05, positive out-of-sample 2022+), each ≈ +0.05–0.07R/trade.
* The popular "liquidity sweep / stop-hunt reversal" is clearly NEGATIVE after costs on all 8 markets.
* Round 2: Crabel NR4/inside-day ORB on US indices +0.06R (OOS +0.12–0.15R) but not significant; break-continuation ≈ 0.
* Walk-forward ML meta-labeling (after removing a look-ahead leak) adds nothing significant.
* Portfolio of the 3 uncorrelated weak edges (NOISE NAS100, London breakout USDJPY, DAX ORB30 with trend):
  ≈ 35R/yr, Sharpe ≈ 1.0, 12/13 positive years, max DD ≈ 48R — before selection-bias haircut.

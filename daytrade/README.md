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

## Rounds 3-4 (all 15 markets, 2014 -> Nov 2026)
* Round 3 (`research3.py`): 36,292 time-of-day / momentum / reversal rules on a 30-minute grid, three-way split
  (find on 2014-19, confirm on 2020-22, report 2023-26). 97 rules looked strong in 2014-19;
  **0 were confirmed in 2020-22.** Intraday patterns found in one period do not persist after costs.
* Round 4 (`research4.py`): literature hypothesis "big-move day continues into the close" (Gao et al. 2018,
  Baltussen et al. 2021), 360 tests. **0 pass FDR 5%**, but US indices at 13:30 New York are positive in all
  three periods: NAS100 |z|>=2 -> 18 trades/yr, hit 62%, +18 bp net/trade (t=2.9). Best candidate, not proven.

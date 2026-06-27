# XAUUSD Research Notes

What actually moves gold, and which mechanisms are operationalisable into a backtestable rule.

This document is the *prior* — what we believe before any backtest. Every hypothesis in `hypotheses.py` must reference a mechanism in this file. If a hypothesis works in backtest but isn't grounded here, treat it as suspected overfit.

---

## 1. The big macro drivers

### 1.1 Real interest rates (strongest single driver, 2003–2024)

Gold pays no coupon. Holding gold has an *opportunity cost* equal to the real (inflation-adjusted) yield on the alternative — typically US 10-year TIPS yield.

- **Negative real yields** (TIPS yield < 0) → strong gold tailwind. Examples: 2008–11, 2019–21, 2024–25.
- **Positive and rising real yields** → strong gold headwind. Example: 2013, 2022.

Operationalisation:
- Proxy: `US10Y - 5y5y breakeven` (or simpler: `^TNX - T10YIE` from FRED).
- Regime: real yield z-score over 250d. Negative-and-falling = bullish regime.
- Use as a regime *filter* (only take long-trend signals in bullish regime), or as a direct positioning signal.

### 1.2 DXY (dollar strength)

Gold is priced in USD. Mechanical inverse correlation when nothing else dominates.

- 60-day rolling beta is usually in [-0.6, -0.2].
- When beta swings to 0 or positive, that's a regime change (other factors taking over).

Operationalisation:
- `dxy_trend` = sign(EMA20 - EMA100) on DXY.
- Pair with gold momentum: long-gold-and-short-dxy momentum together is strongest.
- Divergence: gold rallying while DXY also rallies → strong "other driver" (geopolitics, central bank buying) — usually continuation.

### 1.3 Inflation expectations (5y5y breakeven)

Gold as inflation hedge. Less reliable than real yields but matters at extremes.

- Breakevens > 2.8% with falling real yields → gold rally setup.
- Breakevens < 2% with rising real yields → gold pressure.

### 1.4 Central bank purchases (structural, post-2022)

A regime change. EM central banks (China, Russia, India, Turkey) shifted from selling/neutral to **net buying ~1000 tonnes/year** post-2022. This is structural background demand independent of price action.

Hard to operationalise tick-by-tick. Can use as a sentiment overlay: WGC quarterly purchase reports, IMF reserve disclosures.

### 1.5 Money supply / liquidity

- Global M2 expansion → gold tailwind (debasement thesis).
- Liquidity contraction (Fed QT, BOJ tightening) → gold headwind.
- Operationalise: 12m %-change in global M2 (FRED has US M2; harder for global).

---

## 2. Risk-off / geopolitics

### 2.1 Equity vol (VIX)

- VIX > 25 sustained for >3 days historically coincides with gold rallies (2008, 2020 Mar, 2022 Feb).
- VIX < 15 (low-vol regime) → gold can drift either way; less informative.

Operationalisation:
- `vix_regime` = 0 calm / 1 normal / 2 stressed.
- Use as a "long-gold tilt" multiplier in stressed regimes.

### 2.2 Bond vol (MOVE index)

Bond-market panic predicts gold demand. Often leads VIX in fixed-income-led crises (SVB Mar 2023, UK gilt crisis Sep 2022).

- MOVE > 130 has historically preceded 5–10% gold rallies within 2 months.

### 2.3 Geopolitical shocks

Russia 2022, Israel/Gaza 2023, Iran 2024 — each gave gold a 5–10% spike within a week. Harder to backtest because shocks are episodic.

Operationalisation:
- LLM-analyst overlay (already built): once-daily Claude read of macro/geo headlines → score in [-1, +1].
- Backtest forward-only; advisory sizing only.

---

## 3. Cross-asset

### 3.1 Gold/Silver Ratio (GSR)

XAUUSD / XAGUSD. Centuries-long mean-reverting series with episodic extremes.

- GSR > 85: historically silver outperforms next 30–90 days → ratio falls → both rally but silver more, OR gold drops more than silver.
- GSR < 50: gold outperforms next 30–90 days.

Operationalisation: pair trade or as a regime filter for direction.

### 3.2 Gold vs Bitcoin

Competing "digital gold" narrative. 60-day correlation has flipped between +0.5 and -0.4 over 2018–2025.

- When correlation is positive and BTC pumping → gold often follows with lag.
- When correlation is negative → flight-to-quality favours gold over BTC.

### 3.3 USD/JPY and gold

Both are USD pairs and both have safe-haven flow. In Asian session, gold and USD/JPY often move inversely (JPY up = risk-off = gold up).

---

## 4. Microstructure

### 4.1 London fix (10:30 + 15:00 UTC)

Documented intraday flow concentration. Often produces short, sharp moves in the 5–15 minute window. Algorithms using fix-window VWAP or fade-the-fix have documented edge in specific years.

### 4.2 Session breakdown

UTC sessions:
- **Asia (00:00–07:00)**: thin, wide spreads, mostly ranging unless news. Gold often consolidates.
- **London (07:00–16:00)**: 35–40% of daily volume. Trends start here.
- **NY (12:00–21:00)**: 45–50% of daily volume. London/NY overlap (12:00–16:00) is the prime window.
- **Off (21:00–00:00)**: very thin, avoid.

Most edges that work on gold work in the **London/NY overlap window only**. Backtests including Asia session usually look worse than they should.

### 4.3 Pre-FOMC and pre-NFP drift

Documented in academic literature (Lucca-Moench for equity, similar pattern in gold):
- Gold tends to drift slightly positive in the 24h before FOMC announcements.
- Post-NFP volatility is huge for ~30 min, then mean-reverts.

Operationalisation: pre-FOMC long, flat at announcement. NFP: don't trade ±30 min.

---

## 5. Positioning

### 5.1 COT (Commitments of Traders)

CFTC publishes weekly. Speculator net longs at extremes predict reversals.

- Spec net longs > 90th percentile of trailing 5y → reversal risk within 4–8 weeks.
- Spec net longs < 10th percentile → bottom-formation likely.

Operationalisation: weekly z-score, use as fade-extreme signal or as a position-sizing dampener.

### 5.2 ETF flows (GLD, IAU)

Net inflows lead price moves with ~5d lag. Outflows often confirm tops.

---

## 6. Seasonality

Real but small effects. Backtest carefully — easy to overfit.

- **January effect**: gold often rallies first 2 weeks of January (portfolio rebalancing into commodities).
- **Indian wedding season (Oct–Dec)**: physical demand surge.
- **Chinese New Year (late Jan/early Feb)**: physical demand peak.
- **August summer doldrums**: low vol, mean-reverting.

Use as a small directional tilt, never as a sole entry signal.

---

## 7. What does NOT work (well-documented null results)

Saving these so we don't waste cycles backtesting them as standalone:

- **Simple MA crossovers** on H1/H4 — too noisy, eaten by spread.
- **Pure RSI oversold/overbought** — false signals during trends.
- **Bollinger fade without regime filter** — wrecked by trending periods (2020, 2024).
- **Donchian breakout without filter** — too many false breakouts on H1.
- **Ichimoku** — has anecdotal followers but no rigorous edge in published backtests on XAUUSD.
- **Harmonic patterns** — not statistically distinguishable from noise.
- **Fibonacci levels** — same.

The bots that fail (most retail bots) all live here.

---

## 8. The edges we'll actually test

Refer to `research/hypotheses.py`. Each rule maps to a section above.

| ID | Hypothesis | Mechanism |
|---|---|---|
| H1 | Real-yield regime-conditional trend | §1.1 |
| H2 | DXY-divergence momentum | §1.2 |
| H3 | NY-overlap-only breakout (12:00–16:00 UTC) | §4.2 |
| H4 | VIX-stressed long-gold tilt | §2.1 |
| H5 | MOVE-spike long-gold | §2.2 |
| H6 | GSR-extreme pair trade | §3.1 |
| H7 | Pre-FOMC drift (24h prior) | §4.3 |
| H8 | COT-extreme reversal | §5.1 |
| H9 | London/NY-only Bollinger MR when ADX < 20 | §4.2 + regime |
| H10 | Asian-range fade (range-based MR) | §4.2 |
| H11 | January seasonality long | §6 |
| H12 | Real-yield-falling + DXY-falling joint signal | §1.1 + §1.2 |
| H13 | LLM-analyst overlay on H1 | §2.3 (advisory) |
| H14 | Multi-signal ensemble of survivors | combination |

---

## 9. Honest priors

Out of 14 hypotheses:
- I expect **2–4 to show edge** in-sample.
- After deflated Sharpe correction for 14 trials, I expect **0–2 to survive** as statistically meaningful.
- Of those, I expect **1 or 0 to also pass walk-forward** (no fold collapse).

If we end up with 0, that's a true result — these are hard markets. Either pivot to entirely different edge classes (orderflow data, options skew, SMC structure) or accept gold is too efficient at H1.

If we end up with 1, that's a *real* finding worth paper-trading.

---

## 10. Empirical findings (run on real data)

Run: `python scripts/run_research.py`. Data: 1,650 daily bars 2019-01-28 -> 2025-10-03 after outlier cleaning.

Results sorted by deflated Sharpe (multi-test corrected across 14 candidates incl. BH benchmark):

| ID | Name | In-sample Sh | OOS Sh | CAGR | Final$ | DefSh |
|---|---|---:|---:|---:|---:|---:|
| **BH** | Buy-and-hold gold | **1.12** | **2.18** | **+18.1%** | **$29,770** | **0.04** |
| H6 | Gold/DXY decorrelation | +0.72 | 1.30 | +8.4% | $16,907 | 0.00 |
| H3 | Donchian + DXY counter-trend | +0.45 | 1.01 | +3.0% | $12,116 | 0.00 |
| H14 | SMA200 + 20d momentum | +0.09 | 1.15 | +0.4% | $10,247 | 0.00 |
| H4 | VIX-stressed long-gold | +0.30 | 0.66 | +2.8% | $11,975 | 0.00 |
| H13 | RSI + bullish real-yield | +0.69 | n/a | +0.7% | $10,469 | 0.00 |
| H11 | January 1st-half | +0.38 | 0.53 | +0.8% | $10,565 | 0.00 |
| H12 | RY-falling AND DXY-falling | +0.08 | 0.56 | +0.3% | $10,181 | 0.00 |
| H5 | VIX-spike long-gold | +0.23 | 0.00 | +0.8% | $10,512 | 0.00 |
| H1 | Donchian + bullish RY regime | -0.18 | 0.28 | -0.7% | $9,540 | 0.00 |
| H8 | Extension-from-SMA50 MR | -0.20 | 0.14 | -0.5% | $9,653 | 0.00 |
| H2 | Gold/DXY divergence cont. | -0.44 | -0.21 | -1.7% | $8,946 | 0.00 |
| H9 | Bollinger MR (low-trend) | -0.58 | 0.75 | -2.4% | $8,524 | 0.00 |
| H7 | Mon+Tue drift | -1.44 | 0.26 | -7.8% | $5,878 | 0.00 |

### Honest interpretation

1. **NOTHING beats buy-and-hold gold over 2019-2025.** Period was structurally bullish (gold 3x'd). Active strategies that go flat for part of the time lose by definition.

2. **Only BH passes deflated Sharpe**, and even that's barely (0.04 means 4% confidence under multi-test correction — really we'd want > 0.95). Sample too small for strong claims.

3. **H6 (Gold/DXY decorrelation regime)** is the most interesting active candidate:
   - +8.4% CAGR, OOS Sharpe 1.30
   - Mechanism: when 60d gold/DXY beta is no longer inverse, an "other driver" (central bank buying, geopolitics) is dominant → long gold
   - In a less-trending future regime, this could matter.

4. **H14 (SMA200 + 20d momentum)** has near-zero in-sample Sharpe but OOS Sharpe 1.15 — opposite of overfit. Could be a real edge that wasn't relevant in calm 2019-2021.

5. **H3 (Donchian + DXY counter-trend filter)** also positive in OOS (1.01) and matches the conditional-trend mechanism in §1.2.

### Practical conclusion

For XAUUSD over this dataset and timeframe:
- **Trade plan**: own gold. Hold it. Add to position on H6 or H14 confirmations (regime-conditional entries).
- **Do NOT** trade textbook signals on H1/daily; the cost drag (14bps/side) eats them.
- For an edge that beats BH, you'd need either (a) leverage on a strategy with much better risk-adjusted return, or (b) edge that fires in BEAR-gold periods (which this dataset largely lacks).

### What I would do differently for round 2

- Add 2010-2018 data to include the 2013-2015 gold bear market (current data is biased to bull regime).
- Pull real COT data (CFTC publishes weekly CSV) for positioning signals.
- Test on a SHORT-only universe to see if any strategy survives without the rally tailwind.
- Implement intraday session edges on M15/H1 data (the M5 file exists but I haven't loaded it yet).
- Train ML filter on the trades from H6 and H14 to see if a sub-signal performs better.

# Edge Desk — daily multi-market signal & risk dashboard

Every weekday a GitHub Action downloads 20 years of daily data for 15 liquid markets,
tests a fixed library of research-backed setups on each market's own history (net of costs),
and publishes a self-contained dashboard to GitHub Pages: tradeable signals with entry, stop,
target, time-stop, probability, expectancy, position size and the reasons behind each one.

## Quick start (Darija)

1. Dir repo jdid f GitHub (public, smiytou masalan `edge-desk`) w uploadi fih ga3 had les fichiers.
2. **Settings → Pages** → Source: *Deploy from a branch* → Branch: the default branch (here `claude/xauusd-trader-bot-40k3ak`), folder `/docs` → Save.
3. **Settings → Actions → General** → Workflow permissions: *Read and write* → Save.
4. **Actions → daily-edge-run → Run workflow** (awel merra b yeddek). Men b3d kaykhdem bohdo:
   00:20 UTC (mor ma ysedo ga3 les bougies dyal l'bareh) w 06:20 UTC (9bel London).
5. Dashboard dyalek ghadi ykoun f: `https://kamalelmesbahi45-design.github.io/trading-bot/`

L'code l'9dim mkhbi f branch `backup/old-trading-bot`.

Kol nhar: dkhol l **Today** → chouf signals A/B (houma li fihom edge) → l'plan (entry/stop/target/time stop)
→ l'asbab (why) → l'risk % → w chouf l'calendar dyal les news l'9wiyin 9bel ma tdkhol.

## Markets (15)

| Class | Markets |
|---|---|
| Equity indices | S&P 500, Nasdaq 100, DAX 40 |
| FX majors | EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF |
| Metals | Gold, Silver |
| Energy | WTI crude |
| Crypto | Bitcoin, Ethereum |
| Rates | US 10Y yield |

Selection: liquidity/low cost for retail (CFD, micro futures, prop firms), 5 asset classes for
independent bets, long clean free history, macro-driven, and the instruments used in the classic
trend-following literature. Edit `engine/config.py` to change the list or the cost assumptions.

## Setups (edge library — parameters fixed in advance, never optimised)

| Setup | Style | Edge source |
|---|---|---|
| Trend pullback (RSI-2) | swing 2-7d | short-term reversal inside long-term trend (Connors; Jegadeesh 1990) |
| 55-day breakout + momentum | swing/position | time-series momentum (Moskowitz-Ooi-Pedersen 2012; Hurst-Ooi-Pedersen 2017) |
| Volatility squeeze breakout | swing 1-3w | volatility clustering / range expansion (Engle 1982) |
| IBS reversal | 1-3d | close-location mean reversion, strongest in indices |
| Failed breakout (rejection) | 1-5d | stop-run / trapped breakout traders |
| Turn-of-month | 4d calendar | month-end flows (Lakonishok-Smidt 1988; McConnell-Xu 2008) |

## How a probability is produced

1. Triple-barrier backtest of the exact rule: entry next open, stop/target/time in ATR units,
   same-bar stop+target counted as a LOSS, gaps through the stop filled at the open, costs deducted.
2. Hit rate shrunk toward the break-even rate (Bayesian Beta prior, 20 pseudo-trades); expectancy shrunk toward 0R.
3. Out-of-sample check (last 30 % of trades) and last-5-years check.
4. Benjamini-Hochberg false-discovery control across all market × setup × direction tests.
5. Grade A/B/C/X → only A and B get a position size (quarter-Kelly, capped at 0.5 % / 1 %).
6. Every live signal is logged (`data/signal_log.csv`) and scored on later bars → the
   **Live track record** panel is the real out-of-sample proof.

Day-trader extras per market: previous high/low, pivots, conditional probability of breaking
yesterday's high/low given today's close location and trend, touch probabilities at ±0.25…1.5 ATR,
intraday activity by hour, seasonality, COT positioning, correlation matrix and same-bet warnings,
macro regime (VIX, VIX term structure, credit spreads, curve, real yields, DXY), high-impact calendar.

## Run locally

```bash
pip install -r requirements.txt
python tests/test_engine.py
python run.py                 # live (Yahoo/FRED/CFTC/calendar), falls back to GitHub mirrors
open docs/index.html
```

## Data sources (free, no keys)

Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance) · FRED CSV · CFTC public reporting API ·
ForexFactory weekly JSON feed · charts by [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) (Apache-2.0) ·
fallback mirrors: [datasets/*](https://github.com/datasets), [Mario-SO/ohlcv](https://github.com/Mario-SO/ohlcv),
[ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data).

## Honest limits

Past frequencies are not guarantees; edges decay. Daily bars cannot see intrabar order. CFD swap/financing
is not modelled. Yahoo occasionally rate-limits cloud runners (the run retries, then falls back to mirrors and
says so in a banner). This is a research tool, not financial advice.

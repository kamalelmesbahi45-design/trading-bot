# xauusd-bot

Adaptive multi-strategy quantitative trading bot for XAUUSD.

Backtest-first. Walk-forward validated. Prop-firm aware. Plug-in macro regime, sentiment, and LLM-analyst overlays. Single codebase runs backtest, paper, and live.

## Status

Early skeleton. Modules are stubbed with interfaces and `NotImplementedError`. See `TaskList` for the build plan.

## Layout

```
configs/                 personal vs prop-firm risk profiles
src/xauusd_bot/
  data/                  Dukascopy ticks, yfinance cross-asset, FF calendar
  features/              indicators, cross-asset features, regime features
  strategies/            trend, mean-reversion, breakout, ML-filter
  macro/                 quant regime, FinBERT sentiment, Claude analyst (plug-in)
  risk/                  fractional Kelly sizing, ATR stops, prop-firm gates
  portfolio/             multi-strategy allocator, regime router
  backtest/              event-driven engine, fill sim, spread/swap/commission
  execution/             ExecutionEngine ABC + paper + MT5 impls
  optimize/              walk-forward, Monte Carlo, deflated Sharpe
  reports/               HTML equity/DD/per-strategy report
  monitoring/            Telegram alerts, daily PnL, healthcheck
tests/
scripts/
```

## Quickstart (once filled in)

```bash
pip install -e ".[dev]"
cp .env.example .env       # fill MT5 + Telegram

xauusd data fetch --years 10
xauusd backtest --config configs/personal_aggressive.yaml
xauusd wfo      --config configs/personal_aggressive.yaml
xauusd report   --run latest
xauusd paper    --config configs/propfirm_strict.yaml
xauusd live     --config configs/propfirm_strict.yaml
```

## Profiles

- `personal_aggressive.yaml`: 1% risk, fractional Kelly 0.25, no daily-DD cap
- `propfirm_strict.yaml`: 0.5% risk, fractional Kelly 0.15, FTMO 5% daily / 10% max DD / 8% target

## Multi-asset macro agent

The `xauusd agent` command runs a self-contained multi-asset macro trader that
produces ranked day-trade and swing-trade ideas every day. It combines:

- **Real data**: live OHLCV via yfinance for ~20 liquid ETF / spot / crypto
  proxies (FX, metals, energy, equity, rates, real assets, BTC, ETH), cached
  to parquet. Falls back to the on-disk CSV panel when offline.
- **Real macro regime**: risk-on/risk-off score from DXY trend, VIX level,
  US10Y trend, SPX 50/200 alignment, and the HYG/Gold credit-appetite ratio.
- **Real signals**: blended 3/6/12-month time-series momentum + cross-sectional
  rank for swing; 5/20-day momentum, RSI(14) mean-reversion, Donchian breakout
  with volume confirmation for day.
- **Real risk math**: ATR-based stops (1.5xATR day / 2.5xATR swing), R-multiple
  targets, inverse-vol position sizing scaled by `equity * risk_pct`, cluster
  exposure caps, and a portfolio correlation check that prunes redundant
  longs / shorts before reporting.

```bash
xauusd agent                                  # day + swing, $100k, 0.5% per trade
xauusd agent --mode swing --equity 25000      # swing only on a $25k account
xauusd agent --mode day  --risk-pct 0.0025    # 0.25% per trade, day setups only
xauusd agent --tickers GLD,SPX,BTC,EUR        # filter to a watchlist
xauusd agent --no-network --csv-dir data/cache/multi   # fully offline
xauusd agent --json out/today.json            # also write a machine-readable report
```

Output for each idea: side, entry, ATR stop, R-multiple target, units, dollar
risk, % weight, conviction, and the rationale (which horizons fired, which
day-trading sub-strategy, current regime bias).

The agent only ranks and sizes -- it does **not** place orders. Wire it into
the existing `paper` / `live` execution path when you've validated it on your
own broker, with your own slippage, and during your own trading hours.

## Validation policy

No strategy ships to paper until:
- 10+ years in-sample profitable
- Walk-forward OOS profitable
- Monte Carlo (trade shuffle + bootstrap) shows < 25% chance of ruin at chosen risk
- Deflated Sharpe > 1.0 after multiple-testing correction
- Parameter stability across WFO folds (no cliff edges)

No strategy ships to live until 30+ days of paper trading match backtest expectations within tolerance.

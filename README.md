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

## Validation policy

No strategy ships to paper until:
- 10+ years in-sample profitable
- Walk-forward OOS profitable
- Monte Carlo (trade shuffle + bootstrap) shows < 25% chance of ruin at chosen risk
- Deflated Sharpe > 1.0 after multiple-testing correction
- Parameter stability across WFO folds (no cliff edges)

No strategy ships to live until 30+ days of paper trading match backtest expectations within tolerance.

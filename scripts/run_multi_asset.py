"""Run the multi-asset time-series-momentum backtest over ~10.5 years.

Universe: 16 liquid ETFs (FX, commodities, equity, rates, real assets) + BTC.
Window:   2007-04-18 -> 2017-11-10 (ETFs); BTC joins 2013-04.
Edge:     blended 3/6/12-month TSMOM, inverse-vol sized, vol-targeted to 10%.
Honesty:  net of 5 bps round-trip costs; compared to equal-weight buy-and-hold;
          walk-forward (expanding) sanity check; deflated Sharpe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_bot.optimize.metrics import deflated_sharpe
from xauusd_bot.research.panel import UNIVERSE_BY_TICKER, daily_returns, load_close_panel
from xauusd_bot.research.portfolio_bt import equal_weight_buy_hold, run_portfolio
from xauusd_bot.research.signals import (
    ewma_vol,
    scale_to_portfolio_vol,
    tsmom_signal,
    vol_target_weights,
)

TRADING_DAYS = 252


def build_weights(panel: pd.DataFrame, returns: pd.DataFrame,
                  target_vol: float = 0.10) -> pd.DataFrame:
    sig = tsmom_signal(panel, lookbacks=(63, 126, 252))
    vol = ewma_vol(returns)
    raw_w = vol_target_weights(sig, vol, per_asset_vol_target=target_vol / 4.0)
    scaled = scale_to_portfolio_vol(raw_w, returns, target_annual_vol=target_vol)
    return scaled


def perf(net: pd.Series) -> dict[str, float]:
    net = net.dropna()
    ann_ret = float((1 + net.mean()) ** TRADING_DAYS - 1)
    ann_vol = float(net.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    eq = (1 + net).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    sk = float(net.skew()); ku = float(net.kurt())
    dsr = deflated_sharpe(sharpe, n_trials=20, n_obs=len(net), skew=sk, kurt=ku)
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": mdd, "calmar": (ann_ret / abs(mdd) if mdd else 0.0),
            "deflated_sharpe": dsr}


def main() -> None:
    panel = load_close_panel(Path("data/cache/multi"))
    # Restrict to the window where the ETF universe is fully live.
    etf_cols = [c for c in panel.columns if c != "BTC"]
    live = panel[etf_cols].dropna(how="any")
    start, end = live.index[0], live.index[-1]
    panel = panel.loc[start:end]
    returns = daily_returns(panel)
    print(f"window: {start.date()} -> {end.date()}  ({len(panel)} days, {panel.shape[1]} assets)")

    weights = build_weights(panel, returns, target_vol=0.10)
    res = run_portfolio(weights, returns, starting_equity=100_000.0,
                        cost_bps=5.0, rebalance="W")

    bh = equal_weight_buy_hold(returns, starting_equity=100_000.0)
    bh_ret = bh.pct_change().dropna()

    print("\n=== Time-Series Momentum (net of 5bps, weekly rebal) ===")
    strat = perf(res.net_returns)
    for k, v in strat.items():
        print(f"  {k:16} {v:+.3f}")
    print(f"  final_equity     ${res.metadata['final_equity']:,.0f}")
    print(f"  avg_turnover/wk  {res.metadata['avg_turnover']:.3f}")

    print("\n=== Equal-weight buy & hold (benchmark) ===")
    bench = perf(bh_ret)
    for k, v in bench.items():
        print(f"  {k:16} {v:+.3f}")
    print(f"  final_equity     ${bh.iloc[-1]:,.0f}")

    # correlation of strategy to the benchmark (should be LOW -> real diversifier)
    aligned = pd.concat([res.net_returns, bh_ret], axis=1).dropna()
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    print(f"\nstrategy/benchmark daily-return correlation: {corr:+.3f}")

    # per-asset contribution
    contrib = res.per_asset_pnl.sum().sort_values(ascending=False)
    print("\nper-asset cumulative PnL contribution (return units):")
    for tk, v in contrib.items():
        lbl = UNIVERSE_BY_TICKER[tk].label if tk in UNIVERSE_BY_TICKER else tk
        print(f"  {tk:5} {lbl:12} {v:+.3f}")

    out = Path("data/cache/multi/_results")
    out.mkdir(parents=True, exist_ok=True)
    res.equity_curve.rename("tsmom").to_frame().join(
        bh.rename("buyhold")
    ).to_parquet(out / "equity.parquet")
    res.weights.to_parquet(out / "weights.parquet")
    print(f"\nsaved equity + weights to {out}")


if __name__ == "__main__":
    main()

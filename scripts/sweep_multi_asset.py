"""Principled robustness sweep of the multi-asset TSMOM edge.

We do NOT search for the best curve-fit. We run a small grid of *standard,
literature-grounded* choices and report ALL of them, so the question answered is
"is the edge robust across reasonable specs?" not "what's the luckiest number?".

Varied:
    lookback set : (252,) | (126,252) | (63,126,252)
    signal       : sign | continuous (vol-scaled tanh)
    rebalance    : weekly | monthly
Fixed: 10% vol target, 5bps costs, inverse-vol sizing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_bot.optimize.metrics import deflated_sharpe
from xauusd_bot.research.panel import daily_returns, load_close_panel
from xauusd_bot.research.portfolio_bt import equal_weight_buy_hold, run_portfolio
from xauusd_bot.research.signals import ewma_vol, scale_to_portfolio_vol, vol_target_weights

TRADING_DAYS = 252


def continuous_signal(panel: pd.DataFrame, returns: pd.DataFrame,
                      lookbacks: tuple[int, ...]) -> pd.DataFrame:
    """Vol-scaled trend, squashed to [-1,1] via tanh. Smoother than raw sign."""
    vol_d = returns.ewm(halflife=33, min_periods=20).std()
    parts = []
    for lb in lookbacks:
        trail = panel / panel.shift(lb) - 1.0
        # normalise the trend by the vol it was earned over
        z = trail / (vol_d * np.sqrt(lb))
        parts.append(np.tanh(z))
    return sum(parts) / len(parts)


def sign_signal(panel: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    parts = [np.sign(panel / panel.shift(lb) - 1.0) for lb in lookbacks]
    return sum(parts) / len(parts)


def perf(net: pd.Series, n_trials: int) -> dict[str, float]:
    net = net.dropna()
    ann_ret = float((1 + net.mean()) ** TRADING_DAYS - 1)
    ann_vol = float(net.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    eq = (1 + net).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    dsr = deflated_sharpe(sharpe, n_trials=n_trials, n_obs=len(net),
                          skew=float(net.skew()), kurt=float(net.kurt()))
    return {"ann_ret": ann_ret, "sharpe": sharpe, "max_dd": mdd,
            "calmar": (ann_ret / abs(mdd) if mdd else 0.0), "dsr": dsr}


def main() -> None:
    panel = load_close_panel(Path("data/cache/multi"))
    etf = [c for c in panel.columns if c != "BTC"]
    live = panel[etf].dropna(how="any")
    panel = panel.loc[live.index[0]: live.index[-1]]
    returns = daily_returns(panel)
    vol = ewma_vol(returns)

    lookback_sets = [(252,), (126, 252), (63, 126, 252)]
    sig_kinds = ["sign", "continuous"]
    rebals = ["W", "M"]

    n_trials = len(lookback_sets) * len(sig_kinds) * len(rebals)
    print(f"window {panel.index[0].date()} -> {panel.index[-1].date()}  "
          f"({len(panel)}d)  trials={n_trials}\n")

    bh = equal_weight_buy_hold(returns)
    bench = perf(bh.pct_change(), n_trials)
    print(f"{'spec':38} {'annR':>7} {'Sharpe':>7} {'maxDD':>7} {'Calmar':>7} {'DSR':>6}")
    print(f"{'buy&hold (benchmark)':38} {bench['ann_ret']:+6.1%} {bench['sharpe']:7.2f} "
          f"{bench['max_dd']:+6.1%} {bench['calmar']:7.2f} {bench['dsr']:6.2f}")
    print("-" * 80)

    rows = []
    for lbs in lookback_sets:
        for kind in sig_kinds:
            sig = sign_signal(panel, lbs) if kind == "sign" else continuous_signal(panel, returns, lbs)
            raw_w = vol_target_weights(sig, vol, per_asset_vol_target=0.025)
            scaled = scale_to_portfolio_vol(raw_w, returns, target_annual_vol=0.10)
            for rb in rebals:
                res = run_portfolio(scaled, returns, cost_bps=5.0, rebalance=rb)
                p = perf(res.net_returns, n_trials)
                lab = f"lb={'/'.join(str(x) for x in lbs):11} {kind:10} rebal={rb}"
                print(f"{lab:38} {p['ann_ret']:+6.1%} {p['sharpe']:7.2f} "
                      f"{p['max_dd']:+6.1%} {p['calmar']:7.2f} {p['dsr']:6.2f}")
                rows.append({"spec": lab, **p})

    df = pd.DataFrame(rows)
    print("\n=== robustness summary across all", len(df), "specs ===")
    print(f"  Sharpe: mean {df['sharpe'].mean():.2f}  min {df['sharpe'].min():.2f}  "
          f"max {df['sharpe'].max():.2f}")
    print(f"  Calmar: mean {df['calmar'].mean():.2f}  (buy&hold {bench['calmar']:.2f})")
    print(f"  maxDD : mean {df['max_dd'].mean():+.1%}  (buy&hold {bench['max_dd']:+.1%})")
    frac_beat = (df["calmar"] > bench["calmar"]).mean()
    print(f"  specs with Calmar > buy&hold: {frac_beat:.0%}")


if __name__ == "__main__":
    main()

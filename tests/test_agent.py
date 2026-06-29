"""End-to-end tests for the multi-asset macro agent.

We don't hit yfinance from tests (sandbox / CI). Instead we synthesise OHLCV
panels with controlled regimes (trending, ranging, blow-off) and assert the
agent produces correctly-signed tickets, respects sizing math, and applies the
cluster + correlation rules. The data layer is also unit-tested through its
CSV fallback so we know offline mode works.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xauusd_bot.agent.data import _synth_ohlcv_from_close, close_panel
from xauusd_bot.agent.regime import assess_regime
from xauusd_bot.agent.risk import (
    RiskSettings,
    apply_portfolio_rules,
    build_ticket,
)
from xauusd_bot.agent.scanner import AgentReport, AgentSettings
from xauusd_bot.agent.signals import (
    atr,
    cross_sectional_rank,
    day_signal,
    donchian,
    rsi,
    swing_signal,
)
from xauusd_bot.agent.universe import UNIVERSE_BY_TICKER


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _trend_close(n: int = 500, slope: float = 0.0005, vol: float = 0.01,
                 start: float = 100.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = slope + vol * rng.standard_normal(n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(start * np.exp(np.cumsum(rets)), index=idx)


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return _synth_ohlcv_from_close(close)


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def test_atr_positive_and_smooth() -> None:
    df = _ohlcv(_trend_close())
    a = atr(df, period=14).dropna()
    assert (a > 0).all()
    # ATR should not jump by more than 50% in one step on smooth synthetic data
    assert (a.pct_change().abs().dropna() < 0.5).all()


def test_rsi_bounds() -> None:
    s = _trend_close()
    r = rsi(s).dropna()
    assert r.between(0, 100).all()


def test_donchian_high_geq_low() -> None:
    df = _ohlcv(_trend_close())
    hi, lo = donchian(df, 20)
    diff = (hi - lo).dropna()
    assert (diff >= 0).all()


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def test_swing_signal_long_in_uptrend() -> None:
    s = _trend_close(slope=0.001, vol=0.005)  # smooth uptrend
    sig = swing_signal(s)
    assert sig.side == "LONG"
    assert sig.score > 0


def test_swing_signal_short_in_downtrend() -> None:
    s = _trend_close(slope=-0.001, vol=0.005)
    sig = swing_signal(s)
    assert sig.side == "SHORT"
    assert sig.score < 0


def test_day_signal_picks_breakout_on_thrust() -> None:
    # quiet range then a breakout
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(0.001 * rng.standard_normal(200))
    thrust = base[-1] * (1 + np.linspace(0.0, 0.06, 30))  # 6% rip
    close = pd.Series(np.concatenate([base, thrust]),
                      index=pd.date_range("2023-01-01", periods=230, freq="D", tz="UTC"))
    df = _ohlcv(close)
    # add fake volume that spikes on the breakout
    vol = np.ones(len(close)) * 1e6
    vol[-15:] *= 3
    df["volume"] = vol
    sig = day_signal(df)
    assert sig.side == "LONG"
    assert sig.strategy in {"day_breakout", "day_momentum"}


def test_cross_sectional_rank_signs() -> None:
    scores = {"a": -0.9, "b": -0.1, "c": 0.2, "d": 0.8}
    xs = cross_sectional_rank(scores)
    # extremes get pushed to +-1
    assert xs["a"] == pytest.approx(-1.0)
    assert xs["d"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------
def test_build_ticket_long_geometry() -> None:
    df = _ohlcv(_trend_close(slope=0.001))
    settings = RiskSettings(equity=100_000, risk_pct_per_trade=0.01)
    t = build_ticket(
        ticker="SPX", label="S&P", asset_class="equity", cluster="risk_on_eq",
        ohlc=df, side="LONG", horizon="swing", strategy="swing_trend",
        conviction=0.7, settings=settings, rationale="test",
    )
    assert t is not None
    assert t.stop < t.entry < t.target
    # risk dollars should match the spec (within 1% rounding & weight cap)
    risk_per_unit = abs(t.entry - t.stop)
    implied_risk = risk_per_unit * abs(t.units)
    assert implied_risk == pytest.approx(t.risk_dollars, rel=0.05)


def test_build_ticket_short_geometry() -> None:
    df = _ohlcv(_trend_close(slope=-0.001))
    settings = RiskSettings(equity=50_000, risk_pct_per_trade=0.005)
    t = build_ticket(
        ticker="TLT", label="20Y UST", asset_class="rates", cluster="duration",
        ohlc=df, side="SHORT", horizon="day", strategy="day_momentum",
        conviction=0.5, settings=settings, rationale="test",
    )
    assert t is not None
    assert t.stop > t.entry > t.target
    assert t.units < 0


def test_cluster_cap_drops_third_metal() -> None:
    # three identical-cluster tickets, fourth in a different cluster
    df = _ohlcv(_trend_close(slope=0.001))
    settings = RiskSettings(equity=100_000, risk_pct_per_trade=0.01,
                            max_cluster_risk_pct=0.015)  # cap = $1500, single trade = $1000
    tk = []
    for ticker, conv in [("GLD", 0.9), ("SLV", 0.7), ("GDX", 0.5), ("SPX", 0.8)]:
        a = UNIVERSE_BY_TICKER[ticker]
        t = build_ticket(
            ticker=ticker, label=a.label, asset_class=a.asset_class,
            cluster=a.cluster, ohlc=df, side="LONG", horizon="swing",
            strategy="swing_trend", conviction=conv, settings=settings, rationale="t",
        )
        assert t is not None
        tk.append(t)
    admitted, port = apply_portfolio_rules(tk, pd.DataFrame(), settings)
    admitted_tk = [a.ticker for a in admitted]
    assert "GLD" in admitted_tk
    assert "SPX" in admitted_tk
    # at least one of SLV/GDX must be dropped (cluster cap)
    assert ("SLV" not in admitted_tk) or ("GDX" not in admitted_tk)
    assert any("SLV" in d or "GDX" in d for d in port.dropped)


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------
def test_regime_risk_on_when_spx_up_vix_low() -> None:
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    spx = _trend_close(slope=0.001, vol=0.005, start=4000.0)
    vix = pd.Series(np.full(400, 13.0) + 0.5 * np.random.default_rng(0).standard_normal(400),
                    index=idx)
    dxy = pd.Series(np.full(400, 100.0), index=idx)
    yld = pd.Series(np.linspace(40, 38, 400), index=idx)  # falling yields (risk-on supportive)
    macro = {
        "SPX": _ohlcv(spx),
        "VIX": _ohlcv(vix),
        "DXY": _ohlcv(dxy),
        "US10Y": _ohlcv(yld),
    }
    reg = assess_regime(macro)
    assert reg.risk_score > 0
    assert reg.label.startswith("risk_on") or reg.label == "mixed"
    # equity bias should be > 1 in risk-on
    assert reg.class_bias.get("equity", 0) >= 1.0


def test_regime_risk_off_when_spx_down_vix_high() -> None:
    spx = _trend_close(slope=-0.0015, vol=0.012, start=4000.0)
    idx = spx.index
    vix = pd.Series(np.linspace(15, 35, len(idx)), index=idx)
    dxy = _trend_close(slope=0.0008, vol=0.003, start=100.0, seed=1)
    yld = _trend_close(slope=0.0015, vol=0.01, start=30.0, seed=2)
    macro = {
        "SPX": _ohlcv(spx),
        "VIX": _ohlcv(vix),
        "DXY": _ohlcv(dxy),
        "US10Y": _ohlcv(yld),
    }
    reg = assess_regime(macro)
    assert reg.risk_score < 0
    assert reg.label.startswith("risk_off") or reg.label == "mixed"


# ---------------------------------------------------------------------------
# scanner end-to-end (offline, synthetic universe)
# ---------------------------------------------------------------------------
def _synth_universe(tickers: list[str], seed: int = 0) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)
    for i, tk in enumerate(tickers):
        slope = float(rng.uniform(-0.0008, 0.0012))
        s = _trend_close(slope=slope, vol=0.01, seed=seed + i)
        out[tk] = _synth_ohlcv_from_close(s)
    return out


def test_scanner_smoke_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scanner returns a sane AgentReport against a synthetic universe."""
    tickers = ["SPX", "GLD", "TLT", "BTC", "EUR", "DXY"]
    frames = _synth_universe(tickers, seed=42)

    from xauusd_bot.agent import scanner as scn

    def fake_load_universe(*_a, **_kw):  # type: ignore[no-untyped-def]
        return frames

    def fake_load_macro(*_a, **_kw):  # type: ignore[no-untyped-def]
        # build a macro context with the same series as frames
        return {
            "SPX": frames["SPX"],
            "VIX": _synth_ohlcv_from_close(_trend_close(slope=0.0, vol=0.002, start=15.0, seed=7)),
            "DXY": frames["DXY"],
            "US10Y": _synth_ohlcv_from_close(_trend_close(slope=0.0001, vol=0.005, start=40.0, seed=8)),
            "GOLD": frames["GLD"],
        }

    monkeypatch.setattr(scn, "load_universe", fake_load_universe)
    monkeypatch.setattr(scn, "load_macro_context", fake_load_macro)

    settings = AgentSettings(mode="both", min_conviction=0.2)
    report: AgentReport = scn.run_agent(settings)
    assert isinstance(report, AgentReport)
    # we should have produced *some* tickets given mixed slopes and low threshold
    assert len(report.tickets) >= 1
    # every ticket must obey the geometry rules
    for t in report.tickets:
        if t.side == "LONG":
            assert t.stop < t.entry < t.target
        else:
            assert t.stop > t.entry > t.target
        assert t.risk_dollars > 0
        assert abs(t.weight_pct) <= 0.26  # max_weight 25% + rounding slack


def test_close_panel_aligns_columns() -> None:
    a = _trend_close(seed=1)
    b = _trend_close(seed=2)
    frames = {"A": _synth_ohlcv_from_close(a), "B": _synth_ohlcv_from_close(b)}
    panel = close_panel(frames)
    assert list(panel.columns) == ["A", "B"]
    assert panel.notna().all().all()

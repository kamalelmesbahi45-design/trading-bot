"""Config loader. YAML -> pydantic models. Validates at startup so bad config fails loud."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AccountCfg(BaseModel):
    starting_equity: float
    currency: str = "USD"
    leverage: int = 100


class RiskCfg(BaseModel):
    per_trade_pct: float
    sizer: Literal["fractional_kelly", "fixed_fractional", "vol_target"]
    kelly_fraction: float = 0.25
    kelly_lookback_trades: int = 100
    kelly_min_trades: int = 30
    hard_cap_per_trade_pct: float = 0.02
    max_concurrent_positions: int = 3
    daily_loss_limit_pct: float | None = None
    max_drawdown_pct: float = 0.25
    profit_target_pct: float | None = None
    min_trading_days: int | None = None
    trailing_stop: bool = True


class StopsCfg(BaseModel):
    type: Literal["atr", "structural", "fixed"] = "atr"
    atr_period: int = 14
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0
    breakeven_at_r: float = 1.0


class SessionWindow(BaseModel):
    start: str
    end: str


class SessionFilter(BaseModel):
    enabled: bool = True
    windows_utc: list[SessionWindow] = Field(default_factory=list)


class NewsFilter(BaseModel):
    enabled: bool = True
    blackout_minutes_before: int = 15
    blackout_minutes_after: int = 15
    impact_levels: list[str] = Field(default_factory=lambda: ["high"])


class WeekendFilter(BaseModel):
    flatten_friday_utc: str = "20:00"
    no_trades_until_monday_utc: str = "07:00"


class RegimeFilter(BaseModel):
    enabled: bool = True
    atr_pct_min: float = 0.10
    atr_pct_max: float = 1.50


class FiltersCfg(BaseModel):
    session: SessionFilter
    news: NewsFilter
    weekend: WeekendFilter
    regime: RegimeFilter


class StrategyCfg(BaseModel):
    enabled: bool = False
    weight: float = 1.0
    timeframe: str = "H1"
    model: str | None = None
    threshold: float | None = None


class MacroModuleCfg(BaseModel):
    enabled: bool = False
    model: str | None = None
    cadence: str | None = None


class MacroCfg(BaseModel):
    quant_regime: MacroModuleCfg
    sentiment: MacroModuleCfg
    llm_analyst: MacroModuleCfg


class ExecutionCfg(BaseModel):
    engine: Literal["paper", "mt5"]
    symbol: str = "XAUUSD"
    contract_size: int = 100
    spread_model: Literal["static", "dynamic"] = "dynamic"
    spread_static_points: int = 25
    slippage_points: int = 5
    commission_per_lot: float = 7.0
    swap_long_pct: float = -0.0002
    swap_short_pct: float = -0.0003


class TelegramCfg(BaseModel):
    enabled: bool = True
    on_fills: bool = True
    on_dd_warn_pct: float = 0.10
    daily_summary_utc: str = "21:30"


class AlertsCfg(BaseModel):
    telegram: TelegramCfg


class ReportingCfg(BaseModel):
    output_dir: str = "reports/output"
    open_in_browser: bool = False


class BotConfig(BaseModel):
    profile: str
    account: AccountCfg
    risk: RiskCfg
    stops: StopsCfg
    filters: FiltersCfg
    strategies: dict[str, StrategyCfg]
    macro: MacroCfg
    execution: ExecutionCfg
    reporting: ReportingCfg
    alerts: AlertsCfg


def load_config(path: str | Path) -> BotConfig:
    """Load and validate a YAML config. Raises pydantic ValidationError on bad input."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return BotConfig.model_validate(data)

"""Multi-asset macro trading agent.

A self-contained module that turns a panel of OHLCV bars into a ranked list of
day-trade and swing-trade opportunities, sized for a given account, filtered by
a macro regime score, and capped by cluster correlation.

Public entry point: ``run_agent`` in ``xauusd_bot.agent.scanner``.
"""
from xauusd_bot.agent.scanner import AgentReport, AgentSettings, run_agent

__all__ = ["AgentReport", "AgentSettings", "run_agent"]

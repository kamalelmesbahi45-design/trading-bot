"""
Market universe and global settings.

Why these 15 markets (selection criteria, in order):
  1. Liquidity / low trading cost for a retail day/swing trader (CFD, futures, prop firms).
  2. Coverage of the 5 major asset classes -> many *independent* bets
     (equity indices, FX majors, metals, energy, crypto).
  3. Documented, persistent trend + mean-reversion behaviour in the academic
     literature (these are the instruments used in Moskowitz/Ooi/Pedersen 2012
     and Hurst/Ooi/Pedersen 2017 trend-following studies).
  4. Long, clean free daily history (>=10y) so edge statistics are meaningful.
  5. Strongly macro-driven -> the macro panel actually adds information.

cost_bps = assumed ROUND-TRIP cost (spread + commission + slippage) in basis points
of price for a typical retail CFD / micro-futures account. Deliberately conservative.
Every backtest result is NET of this cost.
"""

MARKETS = [
    # id        name                 class        yahoo       cftc_code  cot_invert cost_bps  macro_beta
    dict(id="SPX500", name="S&P 500",            cls="Equity index", yf="^GSPC",    cftc="13874A", cot_inv=False, cost_bps=1.5, risk=+1, usd=0),
    dict(id="NAS100", name="Nasdaq 100",         cls="Equity index", yf="^NDX",     cftc="209742", cot_inv=False, cost_bps=1.5, risk=+1, usd=0),
    dict(id="GER40",  name="DAX 40",             cls="Equity index", yf="^GDAXI",   cftc=None,     cot_inv=False, cost_bps=1.5, risk=+1, usd=0),
    dict(id="EURUSD", name="EUR/USD",            cls="FX major",     yf="EURUSD=X", cftc="099741", cot_inv=False, cost_bps=1.0, risk=0,  usd=-1),
    dict(id="GBPUSD", name="GBP/USD",            cls="FX major",     yf="GBPUSD=X", cftc="096742", cot_inv=False, cost_bps=1.3, risk=+1, usd=-1),
    dict(id="USDJPY", name="USD/JPY",            cls="FX major",     yf="JPY=X",    cftc="097741", cot_inv=True,  cost_bps=1.0, risk=+1, usd=+1),
    dict(id="AUDUSD", name="AUD/USD",            cls="FX major",     yf="AUDUSD=X", cftc="232741", cot_inv=False, cost_bps=1.6, risk=+1, usd=-1),
    dict(id="USDCAD", name="USD/CAD",            cls="FX major",     yf="CAD=X",    cftc="090741", cot_inv=True,  cost_bps=1.5, risk=-1, usd=+1),
    dict(id="USDCHF", name="USD/CHF",            cls="FX major",     yf="CHF=X",    cftc="092741", cot_inv=True,  cost_bps=1.8, risk=+1, usd=+1),
    dict(id="XAUUSD", name="Gold",               cls="Metal",        yf="GC=F",     cftc="088691", cot_inv=False, cost_bps=1.5, risk=-1, usd=-1),
    dict(id="XAGUSD", name="Silver",             cls="Metal",        yf="SI=F",     cftc="084691", cot_inv=False, cost_bps=6.0, risk=0,  usd=-1),
    dict(id="WTI",    name="WTI Crude Oil",      cls="Energy",       yf="CL=F",     cftc="067651", cot_inv=False, cost_bps=4.0, risk=+1, usd=-1),
    dict(id="BTCUSD", name="Bitcoin",            cls="Crypto",       yf="BTC-USD",  cftc="133741", cot_inv=False, cost_bps=8.0, risk=+1, usd=0),
    dict(id="ETHUSD", name="Ethereum",           cls="Crypto",       yf="ETH-USD",  cftc="146021", cot_inv=False, cost_bps=10.0, risk=+1, usd=0),
    dict(id="US10Y",  name="US 10Y yield (ZN inverse)", cls="Rates", yf="^TNX",     cftc=None,     cot_inv=False, cost_bps=2.0, risk=+1, usd=+1),
]
# risk = sign of the market's typical reaction to risk-on (stocks up, VIX down)
# usd  = sign of the market's reaction to a stronger US dollar

# Macro inputs (not traded, used for the regime panel)
MACRO_YF = {
    "VIX": "^VIX", "VIX3M": "^VIX3M", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "US3M": "^IRX",
    "COPPER": "HG=F", "HYG": "HYG", "IEF": "IEF",
}
# FRED series (downloaded as CSV from fredgraph, no API key needed)
MACRO_FRED = {
    "T10Y2Y": "10Y-2Y Treasury curve (%)",
    "BAMLH0A0HYM2": "US High-Yield credit spread (OAS, %)",
    "T5YIE": "5Y breakeven inflation (%)",
    "DFII10": "10Y real yield (TIPS, %)",
    "DFF": "Fed funds effective rate (%)",
}

HISTORY_YEARS = 20          # backtest window: recent enough to be relevant, long enough for N
OOS_FRACTION = 0.30         # last 30% of trades = out-of-sample check
PRIOR_STRENGTH = 20         # Bayesian shrinkage: pseudo-trades pulling stats toward "no edge"
FDR_Q = 0.10                # Benjamini-Hochberg false discovery rate across ALL tests
BASE_RISK_PCT = 0.50        # default risk per trade (% of equity)
MAX_RISK_PCT = 1.00         # hard cap per trade
KELLY_FRACTION = 0.25       # quarter-Kelly
TARGET_VOL = 0.15           # for vol-managed exposure scaling (annualised)

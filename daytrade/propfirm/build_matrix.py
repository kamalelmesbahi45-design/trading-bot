import pandas as pd, numpy as np, itertools
def L(f, day='day'):
    t=pd.read_csv(f,parse_dates=[day]); return t
S={}
S['NOISE_NAS']=L('rout/trades_NOISE__NAS100__.csv')
S['LDN_JPY']=L('rout/trades_LONDON__USDJPY__stop_mode-opp_target_mode-EOD_trend_filter-False.csv')
S['ORB_DAX']=L('rout/trades_ORB_RANGE__GER40__W-30_target_mode-EOD_trend_filter-True.csv')
S['NAS_1330']=L('rout4/tr_NAS100_1330_2.0_half.csv')
S['SPX_1330']=L('rout4/tr_SPX500_1330_2.0_half.csv')
S['XAU_1000']=L('rout4/tr_XAUUSD_1000_2.0_half.csv')
A=pd.read_csv('rout5/high_rr_trades.csv',parse_dates=['day'])
sw=A[(A.dir==1)&(A.setup=='B5R')].copy()
sw['day']=sw.day+pd.to_timedelta((sw.bars*1.45).round(),unit='D')   # P&L booked at (approx) exit
S['SWING_5R']=sw[['day','R']]
for k,v in S.items(): print(f"{k:10s} n={len(v):5d} meanR={v.R.mean():+.3f} per_yr={len(v)/12.5:5.0f}")
idx=pd.bdate_range('2014-10-01','2026-11-06')
M=pd.DataFrame({k:v.groupby(v.day.dt.normalize()).R.sum().reindex(idx).fillna(0) for k,v in S.items()})
M.to_parquet('/tmp/M.parquet')

import pandas as pd, numpy as np, itertools, json
M=pd.read_parquet('/tmp/M.parquet')
# per-day trade counts for the haircut (haircut = remove h * mean edge from EVERY trade)
import importlib.util
src=open('/tmp/combo.py').read().split("idx=pd.bdate_range")[0]; ns={}; exec(src,ns); S=ns['S']
idx=M.index
C=pd.DataFrame({k:v.groupby(v.day.dt.normalize()).size().reindex(idx).fillna(0) for k,v in S.items()})
MEAN={k:v.R.mean() for k,v in S.items()}
GROUPS={'INTRA3':['NOISE_NAS','LDN_JPY','ORB_DAX'],'US1330':['NAS_1330','SPX_1330'],'XAU':['XAU_1000'],'SWING':['SWING_5R']}
def series(w,h):
    p=np.zeros(len(M))
    for g,cols in GROUPS.items():
        for c in cols:
            p+= w[g]*(M[c].values - h*MEAN[c]*C[c].values)
    return p   # % of initial balance per day
def phase(p,s,target,maxloss=10,daily=5):
    q=p[s:]; cum=np.cumsum(q)
    bad=np.where((cum<=-maxloss)|(q<=-daily))[0]; ok=np.where(cum>=target)[0]
    b=bad[0] if len(bad) else 10**9; o=ok[0] if len(ok) else 10**9
    if o<b: return 1,o+1
    if b<o: return 0,b+1
    return -1,None
def challenge(p,s,t1=10,t2=5):
    r,d1=phase(p,s,t1)
    if r!=1: return r,None
    r2,d2=phase(p,s+d1,t2)
    if r2!=1: return r2,None
    return 1,d1+d2
starts=np.arange(60,len(M)-260,10)
dates=M.index[starts]
def evaluate(w,h,mask):
    p=series(w,h); res=[];days=[]
    for s in starts[mask]:
        r,d=challenge(p,s); res.append(r); days.append(d if d else np.nan)
    res=np.array(res); days=np.array(days,float); fin=res>=0
    return dict(pass_=float((res==1).sum()/max(1,fin.sum())), fail=float((res==0).sum()/max(1,fin.sum())),
                in60=float(((res==1)&(days<=60)).mean()), in90=float(((res==1)&(days<=90)).mean()),
                med=float(np.nanmedian(days)) if (res==1).any() else None)
grid=[0,0.25,0.5,0.75,1.0]
train=dates<pd.Timestamp('2020-01-01'); test=~train
rows=[]
for a,b,c,d in itertools.product(grid,grid,grid,grid):
    if a==b==c==d==0: continue
    w=dict(INTRA3=a,US1330=b,XAU=c,SWING=d)
    tr=evaluate(w,0.5,train)
    rows.append(dict(**w,**{'tr_'+k:v for k,v in tr.items()}))
R=pd.DataFrame(rows).sort_values('tr_in90',ascending=False)
R.to_csv('/tmp/grid.csv',index=False)
print(R.head(12).round(3).to_string(index=False))

def challenge_dyn(p_unit,s,k,t1=10,t2=5,mode='fixed',daily_stop=None):
    """p_unit = daily R-sum at 1% per unit weight; k = base risk multiplier. Dynamic: shrink risk when in drawdown."""
    out_days=0
    for tgt in (t1,t2):
        cum=0.0; d=0
        while True:
            if s+d>=len(p_unit): return -1,None
            m=k
            if mode=='protect': m=k*min(1.0,max(0.35,(10+cum)/10))          # risk shrinks as max-loss gets closer
            if mode=='protect+lock': 
                m=k*min(1.0,max(0.35,(10+cum)/10))
                if tgt-cum<=2: m=min(m,k*0.5)                              # don't rush the last 2%
            x=p_unit[s+d]*m
            if daily_stop is not None: x=max(x,-daily_stop)                # approximation of a hard daily stop
            cum+=x; d+=1
            if x<=-5 or cum<=-10: return 0,None
            if cum>=tgt: break
        s+=d; out_days+=d
    return 1,out_days
MIX=dict(INTRA3=1.0,US1330=1.0,XAU=1.0,SWING=0.5)   # fixed proportions, chosen by reasoning, not by the grid
def frontier(h,mask,mode,ks=(0.15,0.25,0.35,0.5,0.75,1.0)):
    out=[]
    pu=series(MIX,h)
    for k in ks:
        res=[];days=[]
        for s in starts[mask]:
            r,d=challenge_dyn(pu,s,k,mode=mode); res.append(r); days.append(d if d else np.nan)
        res=np.array(res); dd=np.array(days,float); fin=res>=0
        out.append(dict(mode=mode,h=h,risk_per_trade=k,pass_=round((res==1).sum()/fin.sum(),3),fail=round((res==0).sum()/fin.sum(),3),
                        in60=round(((res==1)&(dd<=60)).mean(),3),in90=round(((res==1)&(dd<=90)).mean(),3),
                        in180=round(((res==1)&(dd<=180)).mean(),3),med_days=np.nanmedian(dd) if (res==1).any() else None))
    return out
rows=[]
for h in (0.0,0.5,1.0):
    for mode in ('fixed','protect','protect+lock'):
        rows+=frontier(h,test,mode)
F=pd.DataFrame(rows); F.to_csv('/tmp/frontier.csv',index=False)
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
print("\nTEST PERIOD (challenges started 2020-2025). h = share of the edge removed (1.0 = zero edge)")
print(F.to_string(index=False))

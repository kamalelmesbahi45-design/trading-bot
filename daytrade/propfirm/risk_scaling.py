exec(open('/tmp/sim.py').read().split("grid=[0,0.25")[0])
MIX=dict(INTRA3=1.0,US1330=1.0,XAU=1.0,SWING=0.5)
train=dates<pd.Timestamp('2020-01-01'); test=~train
def mult(scheme,cum,tgt,streak):
    if scheme=='fixed': return 1.0
    if scheme=='protect (down only)': return min(1.0,max(0.35,(10+cum)/10))
    if scheme=='equity-scaled (up & down)': return min(2.0,max(0.35,(10+cum)/10))
    if scheme=='steps': 
        return 2.0 if cum>=6 else 1.5 if cum>=3 else 1.0 if cum>-3 else 0.5 if cum>-6 else 0.35
    if scheme=='house money': 
        return (1.0+0.15*max(0,cum)) * (min(1.0,max(0.35,(10+cum)/10)) if cum<0 else 1.0)
    if scheme=='win-streak': 
        b=min(1.0,max(0.35,(10+cum)/10)); return b*(1.5 if streak>=2 else 1.0)
    if scheme=='steps + slow at target':
        m=2.0 if cum>=6 else 1.5 if cum>=3 else 1.0 if cum>-3 else 0.5 if cum>-6 else 0.35
        return min(m,1.0) if tgt-cum<=1.5 else m
def run(pu,s,k,scheme,t1,t2):
    tot=0
    for tgt in (t1,t2):
        cum=0.0; d=0; streak=0
        while True:
            if s+d>=len(pu): return -1,None
            x=pu[s+d]*k*mult(scheme,cum,tgt,streak)
            if pu[s+d]!=0: streak = streak+1 if x>0 else 0
            cum+=x; d+=1
            if x<=-5 or cum<=-10: return 0,None
            if cum>=tgt: break
        s+=d; tot+=d
    return 1,tot
SCH=['fixed','protect (down only)','equity-scaled (up & down)','steps','house money','win-streak','steps + slow at target']
rows=[]
for h in (0.0,0.5):
    pu=series(MIX,h)
    for (t1,t2) in ((10,5),(8,5)):
        for k in (0.4,0.5,0.6):
            for sc in SCH:
                res=[];dd=[]
                for s in starts[test]:
                    r,d=run(pu,s,k,sc,t1,t2); res.append(r); dd.append(d if d else np.nan)
                res=np.array(res); dd=np.array(dd,float)
                rows.append(dict(edge=f"{100-h*100:.0f}%",rules=f"{t1}+{t2}",base=k,scheme=sc,
                    in60=((res==1)&(dd<=60)).mean(),in90=((res==1)&(dd<=90)).mean(),in180=((res==1)&(dd<=180)).mean(),
                    ever=np.mean(res==1),blown=np.mean(res==0),median=np.nanmedian(dd)))
R=pd.DataFrame(rows); R.to_csv('/tmp/scaling.csv',index=False)
pd.set_option('display.width',220); pd.set_option('display.max_rows',300)
f=lambda x: f"{x:.0%}"
for col in ('in60','in90','in180','ever','blown'): R[col]=R[col].map(f)
print(R[(R.base==0.5)].to_string(index=False))

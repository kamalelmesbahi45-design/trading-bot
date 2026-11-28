exec(open('/tmp/sim.py').read().split("grid=[0,0.25")[0])
exec("def challenge_dyn"+open('/tmp/sim.py').read().split("def challenge_dyn")[1].split("MIX=dict")[0])
MIX=dict(INTRA3=1.0,US1330=1.0,XAU=1.0,SWING=0.5)
train=dates<pd.Timestamp('2020-01-01'); test=~train
for h in (0.0,0.5):
    pu=series(MIX,h)
    for (t1,t2) in ((10,5),(8,5)):
        for k,mode in ((0.5,'protect'),(0.5,'fixed'),(0.75,'protect')):
            res=[];dd=[]
            for s in starts[test]:
                r,d=challenge_dyn(pu,s,k,t1=t1,t2=t2,mode=mode); res.append(r); dd.append(d if d else np.nan)
            res=np.array(res); dd=np.array(dd,float); n=len(res)
            print(f"edge kept {100-h*100:3.0f}%  target {t1}%+{t2}%  risk {k}% {mode:8s}  funded<=60d {((res==1)&(dd<=60)).mean():4.0%}  <=90d {((res==1)&(dd<=90)).mean():4.0%}  <=180d {((res==1)&(dd<=180)).mean():4.0%}  ever {np.mean(res==1):4.0%}  blown {np.mean(res==0):4.0%}  median {np.nanmedian(dd):4.0f}d")

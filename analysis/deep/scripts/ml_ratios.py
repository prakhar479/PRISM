"""Exploratory: per layer, ratio of each set's measure to English (en_pile) with 95% bootstrap CIs over windows (independent
resampling per set, ratio of sums). Measures: ce_delta (splice cost, nats/token), abl_range (mean-ablation cost), rec_mean,
fvu, l0, and 'sensitivity' = ce_delta / fvu."""
import numpy as np, pandas as pd
SP="/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
w=pd.read_parquet(f"{SP}/deep/ml_windows.parquet"); w=w[w.set!="en_pilot100"]
COLS=["nll_clean","nll_splice","nll_mean","pred","fvu_num","fvu_den","l0_sum","pos"]
def stats(a):  # a: [n, len(COLS)] sums
    s=a.sum(0); c,sp,m,p,fn,fd,l0,pos=s
    d=(sp-c)/p; rng=(m-c)/p; fvu=fn/fd
    return np.array([d, rng, (m-sp)/(m-c), fvu, l0/pos, d/fvu])
NAMES=["ce_delta","abl_range","rec_mean","fvu","l0","sens"]
rng=np.random.default_rng(0); B=2000; rows=[]
for L,g in w.groupby("layer"):
    arrs={s:x[COLS].to_numpy() for s,x in g.groupby("set")}
    boots={s:np.array([stats(a[rng.integers(0,len(a),len(a))]) for _ in range(B)]) for s,a in arrs.items()}
    pts={s:stats(a) for s,a in arrs.items()}
    en_b, en_p = boots["en_pile"], pts["en_pile"]
    for s in arrs:
        r={"layer":L,"set":s}
        for j,n in enumerate(NAMES):
            r[n]=pts[s][j]
            if s!="en_pile":
                rb=boots[s][:,j]/en_b[:,j]; r[n+"_ratio"]=pts[s][j]/en_p[j]; r[n+"_ratio_lo"],r[n+"_ratio_hi"]=np.percentile(rb,[2.5,97.5])
        rows.append(r)
t=pd.DataFrame(rows); t.to_csv(f"{SP}/deep/ml_ratios.csv",index=False)
pd.set_option("display.width",250)
show=["layer","set","ce_delta","ce_delta_ratio","ce_delta_ratio_lo","ce_delta_ratio_hi","abl_range_ratio","fvu_ratio","fvu_ratio_lo","fvu_ratio_hi","sens","sens_ratio","sens_ratio_lo","sens_ratio_hi","l0_ratio"]
print(t[show].round(3).to_string(index=False))

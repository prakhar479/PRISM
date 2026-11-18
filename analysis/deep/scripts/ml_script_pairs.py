"""Exploratory: Dakshina same-sentence-pool natural experiment (native vs romanized script, same language, same genre).
Per (language, layer): native/romanized ratios of ce_delta, fvu, eps_x, l0, sensitivity (ce_delta/fvu), mean-ablation range,
with 95% bootstrap CIs (windows resampled within each set). Also the pooled OLS of ce_delta on eps_x (instead of fvu) + ce_clean
+ group indicators, as a robustness check of ml_matched.py; and the ce_clean overlap between groups."""
import numpy as np, pandas as pd
SP="/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
w=pd.read_parquet(f"{SP}/deep/ml_windows.parquet"); w=w[w.set!="en_pilot100"].copy()
C=["nll_clean","nll_splice","nll_mean","pred","fvu_num","fvu_den","eps_ratio_sum","l0_sum","pos"]
def st(a):
    c,s,m,p,fn,fd,er,l0,pos=a.sum(0); d=(s-c)/p; f=fn/fd
    return np.array([d,f,er/pos,l0/pos,d/f,(m-c)/p,c/p])
N=["ce_delta","fvu","eps_x","l0","sens","abl_range","ce_clean"]
rng=np.random.default_rng(1); rows=[]
for lang,(nat,rom) in {"hi":("hi_deva_dakshina","hi_latn_dakshina"),"ta":("ta_taml_dakshina","ta_latn_dakshina"),"bn":("bn_beng_dakshina","bn_latn_dakshina")}.items():
    for L in sorted(w.layer.unique()):
        a=w[(w.set==nat)&(w.layer==L)][C].to_numpy(); b=w[(w.set==rom)&(w.layer==L)][C].to_numpy()
        if not len(a) or not len(b): continue
        pa,pb=st(a),st(b); bs=np.array([st(a[rng.integers(0,len(a),len(a))])/st(b[rng.integers(0,len(b),len(b))]) for _ in range(2000)])
        r={"lang":lang,"layer":L}
        for j,n in enumerate(N):
            r[n+"_nat"],r[n+"_rom"]=pa[j],pb[j]; r[n+"_ratio"]=pa[j]/pb[j]; r[n+"_lo"],r[n+"_hi"]=np.percentile(bs[:,j],[2.5,97.5])
        rows.append(r)
t=pd.DataFrame(rows); t.to_csv(f"{SP}/deep/ml_script_pairs.csv",index=False)
pd.set_option("display.width",250)
for n in ["ce_delta","fvu","eps_x","sens","l0","ce_clean"]:
    print(n); print(t[["lang","layer",n+"_nat",n+"_rom",n+"_ratio",n+"_lo",n+"_hi"]].round(3).to_string(index=False))
NATIVE={"hi_deva_dakshina","hi_deva_web","ta_taml_dakshina","ta_web","bn_beng_dakshina","bn_web"}; ROMAN={"hi_latn_dakshina","ta_latn_dakshina","bn_latn_dakshina"}
w["grp"]=np.where(w.set.isin(NATIVE),"native",np.where(w.set.isin(ROMAN),"roman",np.where(w.set=="hinglish_phinc","hinglish",np.where(w.set=="fr_web","fr","en"))))
print("== ce_clean quantiles by group (L12)"); print(w[w.layer==12].groupby("grp").ce_clean.describe(percentiles=[.05,.5,.95]).round(2))
print("== OLS ce_delta ~ eps_x + ce_clean + groups (English baseline), 200 bootstraps")
for L in [6,12,18,24]:
    g=w[w.layer==L]; grps=[x for x in ["native","roman","hinglish","fr"] if (g.grp==x).any()]
    def fit(gg):
        X=np.column_stack([np.ones(len(gg)),gg.eps_x,gg.ce_clean]+[(gg.grp==x).astype(float) for x in grps]); return np.linalg.lstsq(X,gg.ce_delta.to_numpy(),rcond=None)[0]
    b=fit(g); idx=[np.concatenate([rng.choice(np.where(g.set.to_numpy()==s)[0],(g.set==s).sum()) for s in g.set.unique()]) for _ in range(200)]
    bs=np.array([fit(g.iloc[i]) for i in idx]); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    print(L,"; ".join(f"{n} {b[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]" for i,n in enumerate(["const","eps_x","ce_clean"]+grps)))

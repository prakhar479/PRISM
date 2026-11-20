"""Exploratory: is the native-script excess splice cost explained by text predictability (ce_clean) or reconstruction size (fvu)?
(1) Coarsened exact matching: windows binned on ce_clean (0.25-nat bins) x fvu (0.025 bins), per layer; within each common
bin, mean ce_delta of the set minus English's; weighted by the set's windows in common bins. Bootstrap over windows.
(2) OLS (numpy) pooled over sets: ce_delta ~ fvu + ce_clean + group indicators (native Indic, romanized Indic, Hinglish, French),
English baseline; 95% CI by bootstrap over windows within sets."""
import numpy as np, pandas as pd
SP = str(__import__("layout_shim").root())
w=pd.read_parquet(f"{SP}/deep/ml_windows.parquet"); w=w[w.set!="en_pilot100"].copy()
NATIVE={"hi_deva_dakshina","hi_deva_web","ta_taml_dakshina","ta_web","bn_beng_dakshina","bn_web"}
ROMAN={"hi_latn_dakshina","ta_latn_dakshina","bn_latn_dakshina"}
w["grp"]=np.where(w.set.isin(NATIVE),"native",np.where(w.set.isin(ROMAN),"roman",np.where(w.set=="hinglish_phinc","hinglish",np.where(w.set=="fr_web","fr","en"))))
w["b"]=(np.floor(w.ce_clean/0.25)).astype(int).astype(str)+"_"+(np.floor(w.fvu/0.025)).astype(int).astype(str)
rng=np.random.default_rng(0)
def cem(e,s):
    m=pd.concat([e.assign(k="en"),s.assign(k="s")]); t=m.groupby(["b","k"]).ce_delta.mean().unstack().dropna()
    ns=s.groupby("b").size().reindex(t.index)
    return float(((t.s-t.en)*ns).sum()/ns.sum()), int(ns.sum()), float((t.en*ns).sum()/ns.sum())
rows=[]
for L in [6,12,18,24]:
    g=w[w.layer==L]; e=g[g.set=="en_pile"]
    for s,x in g.groupby("set"):
        if s=="en_pile": continue
        d,nm,base=cem(e,x)
        bs=[cem(e.sample(len(e),replace=True,random_state=int(rng.integers(1e9))),x.sample(len(x),replace=True,random_state=int(rng.integers(1e9))))[0] for _ in range(300)]
        raw=x.ce_delta.mean()-e.ce_delta.mean()
        rows.append({"layer":L,"set":s,"raw_diff":raw,"matched_diff":d,"lo":np.percentile(bs,2.5),"hi":np.percentile(bs,97.5),"matched_ratio":(base+d)/base,"n_matched":nm,"n":len(x)})
t=pd.DataFrame(rows); t.to_csv(f"{SP}/deep/ml_matched.csv",index=False)
pd.set_option("display.width",250); print(t.round(3).to_string(index=False))
print("== pooled OLS with group indicators (English baseline)")
for L in [6,12,18,24]:
    g=w[w.layer==L]
    grps=[x for x in ["native","roman","hinglish","fr"] if (g.grp==x).any()]
    def fit(gg):
        X=np.column_stack([np.ones(len(gg)),gg.fvu,gg.ce_clean]+[(gg.grp==x).astype(float) for x in grps])
        return np.linalg.lstsq(X,gg.ce_delta.to_numpy(),rcond=None)[0]
    b=fit(g); bs=np.array([fit(g.groupby("set",group_keys=False).apply(lambda x:x.sample(len(x),replace=True,random_state=int(rng.integers(1e9))))) for _ in range(200)])
    lo,hi=np.percentile(bs,[2.5,97.5],axis=0); names=["const","fvu","ce_clean"]+grps
    print(L, "; ".join(f"{n} {b[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]" for i,n in enumerate(names)))

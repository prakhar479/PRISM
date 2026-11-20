"""Exploratory: E14 task-state loss by language (native fit, each family's own pairs). Per (task, family, filter):
NLDR of feat_A_errB / feat_A_meanPos / feat_A_zero (ratio of means, as harness/e8.py), the gap 1 - NLDR, the absolute logit-diff
loss mean(D_fullA) - mean(D_cond), the clean margin mean(D_A) - mean(D_B), and the difference of the NLDR gap from English's
(same task) with a 95% bootstrap CI (independent resampling of pairs). Gate A/B from tables/final/E13_behaviour_lang.csv."""
import numpy as np, pandas as pd, glob
SP = str(__import__("layout_shim").root())
pp=pd.read_parquet(glob.glob(f"{SP}/ana_runs/E14/*/per_pair.parquet")[0])
pp=pp[(pp.fit_families==pp.family)]
gates=pd.read_csv(f"{SP}/hw/tables/final/E13_behaviour_lang.csv"); gates=gates[gates.split=="test"].set_index("family")
CONDS=["feat_A_errB","feat_A_meanPos","feat_A_zero","feat_B_errA"]
rng=np.random.default_rng(0); B=4000
def arrays(g,flt):
    g=g[g[flt]]; f={c:x.drop_duplicates("pair_id").set_index("pair_id") for c,x in g[g.estimator.fillna("")==""].groupby("condition")}
    ids=f["clean_A"].index.intersection(f["clean_B"].index).intersection(f["full_A"].index)
    D=lambda c:(f[c].loc[ids].logit_A-f[c].loc[ids].logit_B).to_numpy()
    return {c:D(c) for c in ["clean_A","clean_B","full_A"]+[c for c in CONDS if c in f and f[c].index.isin(ids).all()]}
def stat(d,ix,c):
    m=lambda k:d[k][ix].mean(); return np.array([1-(m(c)-m("clean_B"))/(m("full_A")-m("clean_B")), m("full_A")-m(c), m("clean_A")-m("clean_B")])
rows=[]; boots={}
for (task,fam),g in pp.groupby(["task","family"]):
    for flt in ["f_A_correct","f_none"]:
        d=arrays(g,flt); n=len(d["clean_A"])
        idx=[np.arange(n)]+[rng.integers(0,n,n) for _ in range(B)]
        for c in CONDS:
            if c not in d: continue
            s=np.array([stat(d,ix,c) for ix in idx]); boots[(task,fam,flt,c)]=s[1:,0]
            lo,hi=np.percentile(s[1:],[2.5,97.5],axis=0)
            rows.append({"task":task,"family":fam,"filter":flt,"condition":c,"n":n,"gap_nldr":s[0,0],"gap_lo":lo[0],"gap_hi":hi[0],
                         "abs_logit_loss":s[0,1],"abs_lo":lo[1],"abs_hi":hi[1],"clean_margin":s[0,2],
                         "acc_clean":gates.acc_clean.get(fam),"gateA":gates.gate_A_viable.get(fam),"gateB":gates.gate_B_within_10pts.get(fam)})
t=pd.DataFrame(rows)
for i,r in t.iterrows():
    en=r.family.split("_")[0]+"_en"; k=(r.task,en,r["filter"],r.condition)
    if r.family!=en and k in boots:
        dd=boots[(r.task,r.family,r["filter"],r.condition)]-boots[k]
        t.loc[i,"gap_minus_en"]=r.gap_nldr-t[(t.task==r.task)&(t.family==en)&(t["filter"]==r["filter"])&(t.condition==r.condition)].gap_nldr.iloc[0]
        t.loc[i,"diff_lo"],t.loc[i,"diff_hi"]=np.percentile(dd,[2.5,97.5])
t.to_csv(f"{SP}/deep/ml_e14_gaps.csv",index=False)
pd.set_option("display.width",250)
print(t[(t["filter"]=="f_A_correct")].round(3).to_string(index=False))

"""Exploratory: is the non-English splice cost a shift or a tail? Per layer, quantiles (10/25/50/75/90/99) of per-window ce_delta
and their ratio to English's same quantile; share of the set's total excess (sum over windows of ce_delta - English mean) that
comes from its worst 10% windows; fraction of windows above English's 90th percentile. Plus depth profile of the ce_delta ratio
and L0 ratio (from ml_ratios.csv)."""
import numpy as np, pandas as pd
SP = str(__import__("layout_shim").root())
w=pd.read_parquet(f"{SP}/deep/ml_windows.parquet"); w=w[w.set!="en_pilot100"]
Q=[10,25,50,75,90,99]; rows=[]
for L,g in w.groupby("layer"):
    e=g[g.set=="en_pile"].ce_delta; eq=np.percentile(e,Q); p90=np.percentile(e,90)
    for s,x in g.groupby("set"):
        q=np.percentile(x.ce_delta,Q); ex=x.ce_delta-e.mean(); top=np.sort(ex.to_numpy())[::-1][:len(x)//10]
        rows.append({"layer":L,"set":s,**{f"q{k}_ratio":q[i]/eq[i] for i,k in enumerate(Q)},"frac_above_en_p90":(x.ce_delta>p90).mean(),
                     "top10pct_share_of_excess":top.sum()/ex.sum() if s!="en_pile" else np.nan})
t=pd.DataFrame(rows); t.to_csv(f"{SP}/deep/ml_tails.csv",index=False)
pd.set_option("display.width",250); print(t[t.layer.isin([12,18])].round(2).to_string(index=False))
r=pd.read_csv(f"{SP}/deep/ml_ratios.csv")
print(r.pivot(index="set",columns="layer",values="ce_delta_ratio").round(2)); print(r.pivot(index="set",columns="layer",values="l0_ratio").round(2))
print(r.pivot(index="set",columns="layer",values="ce_delta").round(3))

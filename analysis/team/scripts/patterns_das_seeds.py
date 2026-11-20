"""E17 (RS L18): do different DAS fits (seed / epochs / lr) restore the SAME pairs? Per-pair full_along_U NLDR correlation
between fits of the same rank, vs. correlation with error-PCA r8 (a different estimator). Exploratory/post hoc."""
import numpy as np, pandas as pd
from scipy import stats
import patterns_common as C

d = C.add_refs(C.load("E17", filt=lambda x: x.condition.isin(["clean_A", "clean_B", "full_along_U"]) & (x.estimator != "random")))
d = d[d.A_ok & d.B_ok & (d.condition == "full_along_U")]
d["key"] = d.estimator + "|r" + d.r.astype(str)
W = d.pivot_table(index="pair_id", columns="key", values="nldr").clip(-1, 3)
W = W[[c for c in W.columns if c.startswith("das") and int(c.split("|r")[1]) == int(c.split("|")[0][3:].split("_")[0])] + ["pca_pooled|r8"]]
C_ = W.corr(method="spearman")
C_.round(2).to_csv(C.OUT / "das_seed_pair_corr.csv")
rank = lambda c: int(c.split("|")[0][3:].split("_")[0]) if c.startswith("das") else -1
rows = []
cols = list(W.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        a, b = cols[i], cols[j]
        rows.append(dict(a=a, b=b, same_rank=rank(a) == rank(b), rho=C_.loc[a, b], mean_a=W[a].mean(), mean_b=W[b].mean()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print("n pairs", len(W)); print(W.mean().round(3).to_string())
print(R.groupby("same_rank").rho.describe().round(3))
print(R[R.same_rank].sort_values("rho").round(3).to_string())
print(R[R.b == "pca_pooled|r8"].sort_values("rho").round(3).to_string())

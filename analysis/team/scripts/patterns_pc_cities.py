"""People-cities: DAS effect is city-selective. Which cities, and is the city ranking the same across layers (E18 L12/L16),
languages (E22 en/fr/hi/hinglish, native-fit vs en-fit) and DAS ranks? Cities aligned across languages by family city index
(prompt field state_clean = cities.index(A's city)). Exploratory/post hoc."""
import numpy as np
import pandas as pd
from scipy import stats

import patterns_common as C

frames = []
for exp in ["E18", "E22"]:
    d = C.add_refs(C.load(exp, filt=lambda x: (x.task == "people_cities") & x.condition.isin(["full_along_U", "feat_A_along_U", "clean_A", "clean_B"]) & (x.estimator != "random")))
    d = d[d.A_ok & d.B_ok & d.condition.isin(["full_along_U", "feat_A_along_U"])]
    sn = C.set_name_for(C.run_dirs(exp)[0], "people_cities")
    P = C.prompts(sn)
    d["cityA_idx"] = d.pair_id.map(lambda p: P[p]["state_clean"])
    d["cityB_idx"] = d.pair_id.map(lambda p: P[p]["state_corrupt"])
    d["cityA_name"] = d.pair_id.map(lambda p: P[p]["clean_target"].strip())
    d["exp"] = exp
    frames.append(d)
d = pd.concat(frames)
d["key"] = C.ckey(d)
ff = []
for exp in ["E18", "E22"]:
    for rd in C.run_dirs(exp):
        x = pd.read_parquet(rd / "per_pair.parquet", columns=["pair_id", "task", "layer", "condition", "estimator", "r", "family", "fit_families"],
                            filters=[("task", "=", "people_cities")])
        ff.append(x.assign(exp=exp))
ff = pd.concat(ff).drop_duplicates(["exp", "pair_id", "layer", "condition", "estimator", "r", "family"])
# NB: E22 rows of the same (pair, condition, estimator, r) exist once per fit family; keep fit_families by merging on index order
d = d.reset_index(drop=True)
d["cell"] = d.exp + ":L" + d.layer.astype(str) + ":" + d.family
print(d.groupby(["cell", "key"]).size().groupby("cell").describe()[["count", "min", "max"]])

# per-city mean of each DAS key per cell (duplicates from different fits averaged only if present: report n)
das = d[d.estimator.str.startswith("das")]
M = das.groupby(["cell", "key", "cityA_idx"]).agg(m=("nldr", "mean"), n=("nldr", "size"), name=("cityA_name", "first")).reset_index()
M.to_csv(C.OUT / "pc_city_means.csv", index=False)
W = M[M.key.str.startswith("full_along_U")].pivot_table(index="cityA_idx", columns=["cell", "key"], values="m")
names = M[M.cell.str.contains("pc_en")].groupby("cityA_idx").name.first()
cols = list(W.columns)
rows = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        a, b = W[cols[i]], W[cols[j]]
        ok = a.notna() & b.notna()
        if ok.sum() >= 8:
            r, p = stats.spearmanr(a[ok], b[ok])
            rows.append(dict(c1=cols[i][0], k1=cols[i][1], c2=cols[j][0], k2=cols[j][1], n_cities=int(ok.sum()), rho=r, p=p))
R = pd.DataFrame(rows); R["q_bh"] = C.bh(R.p.values); R.to_csv(C.OUT / "pc_city_rank_corr.csv", index=False)
R["same_key"] = R.k1 == R.k2; R["same_cell"] = R.c1 == R.c2
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"{len(R)} city-ranking correlations (BH):")
print(R.groupby(["same_cell", "same_key"]).agg(n=("rho", "size"), rho_med=("rho", "median"), rho_min=("rho", "min"), sig=("q_bh", lambda q: (q < .05).mean())).round(3))
print(R[R.same_key & ~R.same_cell].sort_values("rho").round(3).to_string())
T = W.copy(); T.index = [names.get(i, i) for i in T.index]
print(T.xs("full_along_U|das4|r4", axis=1, level=1).round(2).to_string())
print(T.xs("full_along_U|das1|r1", axis=1, level=1).round(2).to_string() if "full_along_U|das1|r1" in W.columns.get_level_values(1) else "")

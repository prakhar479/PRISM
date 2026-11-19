"""Cross-experiment consistency: (1) identical-condition reproducibility of per-pair logits across runs (bug check);
(2) is pair 'hardness' (feature-only NLDR) a stable pair property across runs, layers and SAE widths? Exploratory/post hoc."""
import numpy as np
import pandas as pd
from scipy import stats

import patterns_common as C

COND = ["clean_A", "clean_B", "feat_A_errB", "feat_A_meanPos", "full_A", "full_B"]
rows = []
for e in ["E4", "E9", "E16", "E17", "E18", "E19", "E20", "E22", "E2r", "E3", "E8", "E11"]:
    try:
        df = C.load(e, filt=lambda x: x.condition.isin(COND))
    except Exception as ex:
        print(e, "skip", ex); continue
    rows.append(df)
D = pd.concat(rows, ignore_index=True)
D = D[D.task.isin(["add_chain", "entity_track", "people_cities"])]
print(D.groupby(["exp", "task"]).pair_id.nunique().unstack().to_string())

# (1) clean runs do not depend on layer/SAE: the same pair_id + family must give the same D everywhere (same prompt file?)
cl = D[D.condition.isin(["clean_A", "clean_B"])]
g = cl.groupby(["task", "family", "pair_id", "condition"]).D.agg(["min", "max", "count", "std"])
g["range"] = g["max"] - g["min"]
print("\nclean D range across all runs/cells for the same pair (logits):")
print(g.groupby(["task", "condition"])["range"].describe(percentiles=[.5, .9, .99]).round(4).to_string())
big = g[g["range"] > 0.5].reset_index()
print("pairs with clean-D range > 0.5:", len(big))
if len(big):
    ex = big.head(3)
    for _, r in ex.iterrows():
        print(cl[(cl.pair_id == r.pair_id) & (cl.condition == r.condition) & (cl.family == r.family)]
              .groupby(["exp", "run", "layer", "sae"]).D.first().round(3).to_string())
g.reset_index().to_csv(C.OUT / "consistency_clean_range.csv", index=False)

# (2) same layer + same SAE + same condition across runs: per-pair feat_A_errB D reproducibility
D = C.add_refs(D)
fa = D[D.condition == "feat_A_errB"]
rep = []
for (t, L, s, fam), g in fa.groupby(["task", "layer", "sae", "family"]):
    P = g.pivot_table(index="pair_id", columns="run", values="D")
    if P.shape[1] < 2:
        continue
    P = P.dropna()
    if len(P) < 20:
        continue
    cols = list(P.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = P[cols[i]], P[cols[j]]
            rep.append(dict(task=t, layer=L, sae=s, family=fam, run1=cols[i], run2=cols[j], n=len(P),
                            pearson=np.corrcoef(a, b)[0, 1], max_abs_diff=float((a - b).abs().max()),
                            med_abs_diff=float((a - b).abs().median())))
R = pd.DataFrame(rep)
run2exp = D.drop_duplicates("run").set_index("run").exp
R["exp1"] = R.run1.map(run2exp); R["exp2"] = R.run2.map(run2exp)
R.to_csv(C.OUT / "consistency_featAerrB_runs.csv", index=False)
print("\nfeat_A_errB per-pair D reproducibility across runs (same task/layer/sae/family):")
print(R.groupby(["task", "layer", "sae"]).agg(n_pairs_of_runs=("n", "size"), min_r=("pearson", "min"), med_maxdiff=("max_abs_diff", "median"),
                                              max_maxdiff=("max_abs_diff", "max")).round(4).to_string())

# (3) pair hardness across layers / widths: per-pair feature-only NLDR (eligible, den>=1), rank corr between cells,
# raw and after regressing out log(den) (gain-model margin) in each cell; pilot families only
el = fa[fa.A_ok & fa.B_ok & (fa.den >= 1) & fa.family.isin(["pilot", "pilot_story", "pilot_calc"])].copy()
el = el.sort_values("run").drop_duplicates(["task", "layer", "sae", "pair_id"])  # one run per cell


def resid(g):
    x = np.log(g.den.values); y = g.nldr.clip(-1, 2).values
    b = np.polyfit(x, y, 1)
    return pd.Series(y - np.polyval(b, x), index=g.pair_id.values)


out = []
for t, g in el.groupby("task"):
    cells = sorted(g.groupby(["layer", "sae"]).groups.keys())
    raw = {c: g[(g.layer == c[0]) & (g.sae == c[1])].set_index("pair_id").nldr.clip(-1, 2) for c in cells}
    res = {c: resid(g[(g.layer == c[0]) & (g.sae == c[1])]) for c in cells}
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = raw[cells[i]].align(raw[cells[j]], join="inner")
            if len(a) < 30:
                continue
            ra, rb = res[cells[i]].align(res[cells[j]], join="inner")
            out.append(dict(task=t, cell1=f"L{cells[i][0]}:{cells[i][1]}", cell2=f"L{cells[j][0]}:{cells[j][1]}", n=len(a),
                            rho_raw=stats.spearmanr(a, b)[0], p_raw=stats.spearmanr(a, b)[1],
                            rho_resid=stats.spearmanr(ra, rb)[0], p_resid=stats.spearmanr(ra, rb)[1]))
H = pd.DataFrame(out); H["q_raw"] = C.bh(H.p_raw.values); H["q_resid"] = C.bh(H.p_resid.values)
H.to_csv(C.OUT / "consistency_hardness.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(f"\npair hardness rank correlations ({len(H)} tests, BH):")
print(H.round(3).to_string())

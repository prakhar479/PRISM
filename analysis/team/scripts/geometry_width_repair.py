"""Error-U repair by SAE width in NLDR (flip is near ceiling for RS repair, so the frozen 10-pt flip rule cannot be met there).
share_closed = (NLDR(repair) - NLDR(feat_A_errB)) / (NLDR(full_A) - NLDR(feat_A_errB)), per estimator and r, vs the 20 random-U
draws at the same r (p95 / max of the random share). Also full_along_U / feat_A_along_U NLDR with random p95.
Post hoc, exploratory. f_pilot and f_none; E9 (16k) and E20 (65k/262k). Out: team/geometry_out/width_repair_nldr.csv"""
import sys, glob
SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
sys.path.insert(0, f"{SP}/hw")
import numpy as np, pandas as pd
from harness import e8 as E
COLS = ["pair_id", "family", "task", "sae", "condition", "estimator", "r", "seed", "ans_A_id", "ans_B_id", "f_none", "f_pilot",
        "logit_A", "logit_B", "lse", "top1_id"]
P = pd.concat([pd.read_parquet(glob.glob(f"{SP}/ana_runs/{e}/*/per_pair.parquet")[0], columns=COLS) for e in ("E9", "E20")])
P = P[P.family.astype(str).str.startswith("pilot")]
P["estimator"] = P.estimator.fillna(""); P["r"] = P.r.fillna(-1); P["seed"] = P.seed.fillna(-1)
rows = []
for (task, sae), T in P.groupby(["task", "sae"]):
    for flt in ["f_pilot", "f_none"]:
        cA = T[T.condition == "clean_A"].drop_duplicates("pair_id").set_index("pair_id")
        pairs = cA.index[cA[flt]].sort_values(); n = len(pairs)
        ansA, ansB = cA.loc[pairs].ans_A_id.values, cA.loc[pairs].ans_B_id.values
        def fr(cond, est="", r=-1, seed=None):
            x = T[(T.condition == cond) & (T.estimator == est) & (T.r == r)]
            if seed is not None: x = x[x.seed == seed]
            return x.drop_duplicates("pair_id").set_index("pair_id").reindex(pairs)
        A, B = E.Arr(cA.loc[pairs], ansA), E.Arr(fr("clean_B"), ansA)
        elig = (fr("clean_B").top1_id.values == ansB).astype(float)
        W = np.ones((1, n))
        m = lambda f: E.metric_samples(E.Arr(f, ansA), A, B, elig, W)["nldr"][0]
        fullA, fe = m(fr("full_A")), m(fr("feat_A_errB"))
        share = lambda v: (v - fe) / (fullA - fe)
        ests = sorted({(e, int(r)) for e, r in T[T.condition == "repair"][["estimator", "r"]].drop_duplicates().itertuples(index=False) if e != "random"})
        for est, r in ests:
            rnd = {c: [m(fr(c, "random", r, s)) for s in sorted(T[(T.estimator == "random") & (T.r == r) & (T.condition == c)].seed.unique())]
                   for c in ("repair", "full_along_U", "feat_A_along_U")}
            row = dict(task=task, sae=sae, filter=flt, n=n, est=est, r=r, gap_nldr=fullA - fe)
            for c in ("repair", "full_along_U", "feat_A_along_U"):
                v = m(fr(c, est, r)); rv = np.array(rnd[c])
                row[f"{c}"] = v; row[f"{c}_rand_p95"] = np.percentile(rv, 95) if len(rv) else np.nan; row[f"{c}_rand_n"] = len(rv)
            row["repair_share"] = share(row["repair"]); row["repair_share_rand_p95"] = share(row["repair_rand_p95"])
            rows.append(row)
d = pd.DataFrame(rows); d.to_csv(f"{SP}/team/geometry_out/width_repair_nldr.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(d[d["filter"] == "f_pilot"].drop(columns=["filter"]).round(3).to_string(index=False))

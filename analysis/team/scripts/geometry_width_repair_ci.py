"""Bootstrap CIs (2000 pair resamples, seed 0, same resamples across widths = paired) for the NLDR repair share
share = (NLDR(repair) - NLDR(feat_A_errB)) / (NLDR(full_A) - NLDR(feat_A_errB)) by width, and wide - 16k differences.
Each width uses its own fitted error U (pca_pooled / answer_sep), so the difference is 'error-U of that SAE'. f_pilot only.
Post hoc, exploratory. Out: team/geometry_out/width_repair_ci.csv"""
import sys, glob
SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
sys.path.insert(0, f"{SP}/hw")
import numpy as np, pandas as pd
from harness import e8 as E, stats as St
COLS = ["pair_id", "family", "task", "sae", "condition", "estimator", "r", "ans_A_id", "ans_B_id", "f_pilot", "logit_A", "logit_B", "lse", "top1_id"]
P = pd.concat([pd.read_parquet(glob.glob(f"{SP}/ana_runs/{e}/*/per_pair.parquet")[0], columns=COLS) for e in ("E9", "E20")])
P = P[P.family.astype(str).str.startswith("pilot") & (P.estimator.fillna("") != "random")]
P["estimator"] = P.estimator.fillna(""); P["r"] = P.r.fillna(-1)
rows = []
for task, ref in [("add_chain", "L18_16k_l074"), ("entity_track", "L12_16k_l082")]:
    T = P[P.task == task]; saes = sorted(T.sae.unique(), key=lambda s: int(s.split("_")[1][:-1]))
    get = lambda s, c, e="", r=-1: T[(T.sae == s) & (T.condition == c) & (T.estimator == e) & (T.r == r)].drop_duplicates("pair_id").set_index("pair_id")
    cA = get(ref, "clean_A"); pairs = cA.index[cA.f_pilot].sort_values(); n = len(pairs)
    ansA, ansB = cA.loc[pairs].ans_A_id.values, cA.loc[pairs].ans_B_id.values
    A, B = E.Arr(cA.loc[pairs], ansA), E.Arr(get(ref, "clean_B").loc[pairs], ansA)
    elig = (get(ref, "clean_B").loc[pairs].top1_id.values == ansB).astype(float)
    W = E.weights([np.arange(n)] + list(St.boot_indices(n, 2000, 0)), n)
    m = lambda f: E.metric_samples(E.Arr(f.loc[pairs], ansA), A, B, elig, W)["nldr"]
    sh = {}
    for s in saes:
        fa, fe = m(get(s, "full_A")), m(get(s, "feat_A_errB"))
        for est in ("pca_pooled", "answer_sep", "das4", "das8"):
            for r in ((1, 2, 4, 8) if not est.startswith("das") else (int(est[3:]),)):
                sh[(s, est, r)] = (m(get(s, "repair", est, r)) - fe) / (fa - fe)
    for (s, est, r), v in sh.items():
        d = v - sh[(ref, est, r)]
        lo, hi = E._ci(v[1:]); dlo, dhi = E._ci(d[1:])
        rows.append(dict(task=task, sae=s, est=est, r=r, n=n, repair_share=v[0], lo=lo, hi=hi, d_vs16k=d[0], d_lo=dlo, d_hi=dhi))
d = pd.DataFrame(rows); d.to_csv(f"{SP}/team/geometry_out/width_repair_ci.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(d.round(3).to_string(index=False))

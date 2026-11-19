"""Feature capture of the (SAE-independent) DAS subspace vs SAE width, paired across widths. Post hoc, exploratory.
E9 (16k) and E20 (65k, 262k) use bit-identical DAS rotations (checked: max |dU| = 0) and the same test pairs, so
full_along_U = x_B + P_U(x_A - x_B) is width-independent and feat_A_along_U = x̂_B + P_U(x̂_A - x̂_B) + ε_B isolates how much of
the causal subspace's A-vs-B difference each SAE's reconstruction carries.
capture = M(feat_A_along_U) / M(full_along_U) (M = NLDR or flip rate, set-level, same resamples); width effect = paired
difference of M(feat_A_along_U) wide - 16k (same pairs, same bootstrap resamples, 2000 pair resamples, seed 0).
Also: feat_A_errB / feat_A_meanPos gap differences wide - 16k, and error-U conditions (U differs by width there).
Out: team/geometry_out/width_capture.csv"""
import sys, glob
SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
sys.path.insert(0, f"{SP}/hw")
import numpy as np, pandas as pd
from harness import e8 as E, stats as St
COLS = ["pair_id", "split", "family", "task", "sae", "condition", "estimator", "r", "seed", "ans_A_id", "ans_B_id",
        "f_none", "f_pilot", "logit_A", "logit_B", "lse", "top1_id"]
def pp(exp):
    f = glob.glob(f"{SP}/ana_runs/{exp}/*/per_pair.parquet")[0]
    d = pd.read_parquet(f, columns=COLS)
    return d[(d.estimator.fillna("") != "random")]
P = pd.concat([pp("E9"), pp("E20")])
print("splits", P.split.unique())
P["estimator"] = P.estimator.fillna(""); P["r"] = P.r.fillna(-1)
rows = []
NB = 2000
for task, ref in [("add_chain", "L18_16k_l074"), ("entity_track", "L12_16k_l082")]:
    T = P[(P.task == task) & P.family.astype(str).str.startswith("pilot")]
    saes = sorted(T.sae.unique(), key=lambda s: int(s.split("_")[1][:-1]))
    get = lambda sae, cond, est="", r=-1: T[(T.sae == sae) & (T.condition == cond) & (T.estimator == est) & (T.r == r)].drop_duplicates("pair_id").set_index("pair_id")
    for flt in ["f_pilot", "f_none"]:
        cA = get(ref, "clean_A"); keep = cA.index[cA[flt]]
        # same pair set across widths: intersect, and check clean runs agree across runs
        for s in saes:
            keep = keep.intersection(get(s, "clean_A").index)
        pairs = keep.sort_values(); n = len(pairs)
        for s in saes[1:]:
            dA = np.abs(get(s, "clean_A").loc[pairs].logit_A.values - cA.loc[pairs].logit_A.values).max()
            dB = np.abs(get(s, "clean_B").loc[pairs].logit_A.values - get(ref, "clean_B").loc[pairs].logit_A.values).max()
            fl = (get(s, "clean_A").loc[pairs, flt].values != cA.loc[pairs, flt].values).sum()
            print(task, flt, s, "max|Δlogit| clean_A", dA, "clean_B", dB, "filter disagreements", fl)
        ca = cA.loc[pairs]; ansA, ansB = ca.ans_A_id.values, ca.ans_B_id.values
        A, B = E.Arr(ca, ansA), E.Arr(get(ref, "clean_B").loc[pairs], ansA)
        elig = (get(ref, "clean_B").loc[pairs].top1_id.values == ansB).astype(float)
        W = E.weights([np.arange(n)] + list(St.boot_indices(n, NB, 0)), n)
        ms = lambda f: E.metric_samples(E.Arr(f.loc[pairs], ansA), A, B, elig, W)
        full_A = {s: ms(get(s, "full_A")) for s in saes}
        for met in ["nldr", "flip_rate"]:
            def ci(v): lo, hi = E._ci(v[1:]); return float(v[0]), lo, hi
            # baselines
            for cond in ["feat_A_errB", "feat_A_meanPos"]:
                m = {s: ms(get(s, cond)) for s in saes}
                for s in saes:
                    g = full_A[s][met] - m[s][met]; dg = (full_A[s][met] - m[s][met]) - (full_A[ref][met] - m[ref][met])
                    rows.append(dict(task=task, filter=flt, n=n, metric=met, sae=s, kind=f"gap_{cond}", est="", value=ci(g)[0],
                                     lo=ci(g)[1], hi=ci(g)[2], d_vs16k=ci(dg)[0], d_lo=ci(dg)[1], d_hi=ci(dg)[2]))
            for est, r in [("das4", 4), ("das8", 8), ("pca_pooled", 1), ("pca_pooled", 4), ("pca_pooled", 8), ("answer_sep", 1), ("answer_sep", 8)]:
                try:
                    fu = {s: ms(get(s, "full_along_U", est, r)) for s in saes}
                    fe = {s: ms(get(s, "feat_A_along_U", est, r)) for s in saes}
                    rp = {s: ms(get(s, "repair", est, r)) for s in saes}
                except KeyError as e:
                    print("missing", task, est, r, e); continue
                for s in saes:
                    cap = fe[s][met] / fu[s][met]
                    dfe = fe[s][met] - fe[ref][met]
                    ill = fu[s][met] - fe[s][met]
                    for kind, v, d in [("feat_A_along_U", fe[s][met], dfe), ("full_along_U", fu[s][met], fu[s][met] - fu[ref][met]),
                                       ("capture_feat/full", cap, cap - fe[ref][met] / fu[ref][met]), ("illusion_full-feat", ill, ill - (fu[ref][met] - fe[ref][met])),
                                       ("repair", rp[s][met], rp[s][met] - rp[ref][met])]:
                        rows.append(dict(task=task, filter=flt, n=n, metric=met, sae=s, kind=kind, est=f"{est}_r{r}", value=ci(v)[0], lo=ci(v)[1],
                                         hi=ci(v)[2], d_vs16k=ci(d)[0], d_lo=ci(d)[1], d_hi=ci(d)[2]))
d = pd.DataFrame(rows)
d.to_csv(f"{SP}/team/geometry_out/width_capture.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 1000)
x = d[(d["filter"] == "f_pilot") & (d.kind.isin(["capture_feat/full", "feat_A_along_U", "illusion_full-feat"]) & d.est.str.startswith("das") | d.kind.str.startswith("gap"))]
print(x[x["filter"] == "f_pilot"].round(3).to_string(index=False))

"""Steering vs patching with the SAME U (E19 steering; E18 / E17 full_along_U and feat_A_along_U, f_pilot).

1. Per (task, estimator, r): patching flip / NLDR (full_along_U = x_B + P_U(x_A - x_B), pair's own x_A) next to steering
   (x_B + s|m| P_U m/|P_U m|, train mean m): flip and NLDR at s = 0.25/0.5/1, best flip over s, best specificity.
   Spearman across cells (uncorrected; n = cells).
2. Mean-component bound. Write P_U(x_A - x_B) = P_U m_test + P_U delta_i. The mean part alone equals steering at
   s0 = |P_U m| / |m| <= 1 (same direction; m_train ~ m_test). If steering NLDR is non-decreasing on (0, 1], then
   NLDR(mean part only) <= NLDR(steer s=1), so the share of the patching NLDR that the mean part can account for is
   <= NLDR_steer(1) / NLDR_patch. (Interpretation: needs monotonicity below s = 0.25, m_train ~ m_test, and treats the
   effects as comparable although the model is nonlinear.) |P_U m| itself is NOT recoverable: neither m nor activations are
   stored (directions/*.npz hold U only).
3. RS constant-shift ceiling: if a single additive vector moves every pair's answer by the same integer k, the best flip
   rate is max_k P(A - B = k) over eligible pairs. Compared with the observed best steering flip. E10b (fresh RS families)
   replicates the delta split.
Outputs: steering_out/patch_link.csv, rs_ceiling.csv, e10b_delta.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from steering_common import SP, OUT, prompts

S = pd.read_csv(OUT / "dose_summary.csv")
S = S[(S.exp == "E19") & (S["filter"] == "f_pilot")]
c18 = pd.read_csv(SP / "r5/tables/E18_conditions.csv")
c17 = pd.read_csv(SP / "r5/tables/E17_conditions.csv")
pat = pd.concat([c18.assign(src="E18"), c17[c17.estimator == "das8_e10"].assign(src="E17")])
pat = pat[(pat["filter"] == "f_pilot") & pat.condition.isin(["full_along_U", "feat_A_along_U"]) & (pat.layer.isin([12, 18]))
          & ~((pat.task == "people_cities") & (pat.layer != 12))]
pw = pat.pivot_table(index=["task", "estimator", "r", "src"], columns="condition", values=["flip_rate", "nldr"]).reset_index()
pw.columns = ["task", "estimator", "r", "src", "feat_flip", "patch_flip", "feat_nldr", "patch_nldr"]

rows = []
for (task, est, r), g in S.groupby(["task", "estimator", "r"]):
    g = g.set_index("s").sort_index()
    p = pw[(pw.task == task) & (pw.estimator == est) & (pw.r == r)]
    d = dict(task=task, estimator=est, r=r,
             steer_flip_025=g.flip.get(0.25), steer_flip_1=g.flip.get(1.0), steer_nldr_025=g.nldr.get(0.25),
             steer_nldr_05=g.nldr.get(0.5), steer_nldr_1=g.nldr.get(1.0), best_flip=g.flip.max(), s_best=g.flip.idxmax(),
             best_spec=g.spec.max(), s_best_spec=g.spec.idxmax(), max_nldr=g.nldr.max(),
             nldr_monotone_to_1=bool(np.all(np.diff(g.nldr.loc[[0.25, 0.5, 1.0]].values) >= -0.005)))
    if len(p):
        p = p.iloc[0]
        d.update(patch_src=p.src, patch_flip=p.patch_flip, patch_nldr=p.patch_nldr, feat_flip=p.feat_flip,
                 feat_nldr=p.feat_nldr)
        d["mean_share_bound"] = (d["steer_nldr_1"] / p.patch_nldr) if d["nldr_monotone_to_1"] and p.patch_nldr > 0.05 else np.nan
    rows.append(d)
L = pd.DataFrame(rows)
L.to_csv(OUT / "patch_link.csv", index=False)
pd.set_option("display.width", 250)
print(L.round(3).to_string(index=False))
ok = L.dropna(subset=["patch_flip"])
for col in ["best_flip", "best_spec", "steer_flip_1", "max_nldr"]:
    rho, pv = spearmanr(ok.patch_flip, ok[col])
    rho2, pv2 = spearmanr(ok.patch_nldr, ok[col])
    print(f"all cells n={len(ok)}: spearman(patch_flip, {col}) = {rho:.3f} p={pv:.3g}; (patch_nldr) {rho2:.3f} p={pv2:.3g}")
for task, g in ok.groupby("task"):
    rho, pv = spearmanr(g.patch_nldr, g.max_nldr)
    print(f"{task} n={len(g)}: spearman(patch_nldr, max steer nldr) = {rho:.3f} p={pv:.3g}")

# ---- RS constant-shift ceiling
pp = pd.read_parquet(OUT / "pp_E19.parquet")
Pr = prompts("add_chain")
out = []
for flt in ["f_pilot", "f_none"]:
    e = pp[(pp.task == "add_chain") & (pp.condition == "clean_B") & pp[flt] & pp.elig].drop_duplicates("pair_id")
    dl = e.pair_id.map(Pr.state_clean - Pr.state_corrupt)
    vc = dl.value_counts(normalize=True).sort_index()
    best = S if flt == "f_pilot" else None
    out.append(dict(filter=flt, n=len(e), **{f"P(d={int(k):+d})": v for k, v in vc.items()}, ceiling=vc.max(),
                    k_ceiling=int(vc.idxmax()), ceiling_sign_pos=float((dl > 0).mean()), ceiling_sign_neg=float((dl < 0).mean())))
C = pd.DataFrame(out)
C.to_csv(OUT / "rs_ceiling.csv", index=False)
print(C.round(3).to_string(index=False))

# ---- E10b fresh families: delta split
e10b = pd.read_parquet(OUT / "pp_E10b.parquet")
import glob, gzip, json
from steering_common import PS
st = {}
for f in glob.glob(str(PS / "add_chain*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        st.setdefault(r["pair_id"], (r["state_clean"], r["state_corrupt"]))
x = e10b[e10b.condition.str.startswith("steer_") & e10b.elig & e10b.f_none].copy()
x["delta"] = x.pair_id.map(lambda p: st[p][0] - st[p][1])
x["grp"] = np.where(x.estimator == "random", "random", x.estimator + "_r" + x.r.astype(str))
E = (x.groupby(["family", "grp", "s"]).apply(lambda g: pd.Series(dict(
    n=g.pair_id.nunique(), flip=g.isA.mean(), flip_pos=g[g.delta > 0].isA.mean(), flip_neg=g[g.delta < 0].isA.mean(),
    flip_d2=g[g.delta == 2].isA.mean(), share_pos=(g.delta > 0).mean()))).reset_index())
E.to_csv(OUT / "e10b_delta.csv", index=False)
print(E[E.grp.str.startswith("das") | (E.grp == "random")].round(3).to_string(index=False))

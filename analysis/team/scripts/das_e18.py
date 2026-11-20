"""E18 rank curves (RS L18, OT L12, PC L12, PC L16; f_pilot test) for full_along_U / feat_A_along_U (illusion test) and repair,
with the random-U null; repair expressed as share of the feat_A_errB residual NLDR gap closed. Plus E20: does widening the SAE change
the feature-mediated share of the (identical) DAS rotation (paired McNemar on per-pair flips), and how much of the DAS learned
subspace sits in the 16k vs wider error-U. Post hoc, exploratory. Outputs: das_out/e18_rank.csv, e20_paired.csv, e20_geom.csv."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from das_lib import *
import pandas as pd
from scipy.stats import binomtest
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
T = f"{SP}/r5/tables"
il = pd.read_csv(f"{T}/E18_illusion.csv"); il = il[il["filter"] == "f_pilot"]
co = pd.read_csv(f"{T}/E18_conditions.csv"); co = co[co["filter"] == "f_pilot"]
il["cell"] = il.task.str[:2].str.upper() + il.layer.astype(str)
co["cell"] = co.task.str[:2].str.upper() + co.layer.astype(str)
base = co[co.estimator.isna() & (co.condition == "feat_A_errB")].set_index("cell")
rnd = co[(co.estimator == "random")]
rnull = rnd.groupby(["cell", "condition", "r"]).nldr.quantile(.95).rename("random_nldr_p95").reset_index()
out = []
for (cell, est, r), g in il[il.r > 0].groupby(["cell", "estimator", "r"]):
    g = g.set_index("condition"); fe = base.loc[cell]
    rep = g.loc["repair"]; nn = rnull[(rnull.cell == cell) & (rnull.condition == "repair") & (rnull.r == r)].random_nldr_p95
    fn = rnull[(rnull.cell == cell) & (rnull.condition == "full_along_U") & (rnull.r == r)].random_nldr_p95
    out.append(dict(cell=cell, est=est, r=r, full_flip=g.loc["full_along_U", "flip_rate"], full_lo=g.loc["full_along_U", "flip_rate_lo"],
                    full_p95=g.loc["full_along_U", "random_p95"], full_eff=g.loc["full_along_U", "effective"],
                    feat_flip=g.loc["feat_A_along_U", "flip_rate"], full_nldr=g.loc["full_along_U", "nldr"],
                    full_nldr_rand95=float(fn.iloc[0]) if len(fn) else np.nan, feat_nldr=g.loc["feat_A_along_U", "nldr"],
                    feat_over_full_nldr=g.loc["feat_A_along_U", "nldr"] / g.loc["full_along_U", "nldr"] if g.loc["full_along_U", "nldr"] > .05 else np.nan,
                    illusion=g.loc["full_along_U", "illusion"], gap=g.loc["full_along_U", "illusion_diff"],
                    gap_lo=g.loc["full_along_U", "illusion_diff_lo"], train_acc=g.loc["full_along_U", "das_train_acc"],
                    valid=g.loc["full_along_U", "das_fit_valid"], featAerrB_nldr=fe.nldr, repair_nldr=rep.nldr,
                    repair_nldr_lo=rep.nldr_lo, repair_rand95=float(nn.iloc[0]) if len(nn) else np.nan,
                    repair_gap_closed=(rep.nldr - fe.nldr) / (1 - fe.nldr),
                    repair_gap_closed_rand95=((float(nn.iloc[0]) - fe.nldr) / (1 - fe.nldr)) if len(nn) else np.nan))
d = pd.DataFrame(out).sort_values(["cell", "est", "r"]); d.to_csv(f"{OUT}/e18_rank.csv", index=False)
print("== E18 rank curves (f_pilot; NLDR null = p95 of 20 random U)")
print(d.drop(columns=["full_lo"]).round(3).to_string(index=False))
print("random draws per (cell,r):", rnd.groupby(["cell", "condition", "r"]).size().describe()[["min", "max"]].to_dict())

# ---- E20 paired comparisons: same DAS rotation, 16k vs wider SAE, feat_A_along_U per pair
print("== E20: feat_A_along_U with identical DAS rotation, 16k (E18/E9) vs wider SAE (E20), paired exact McNemar")
p20 = []
f16 = pd.concat([flips("E18", ["das4", "das8"], "feat_A_along_U"), flips("E18", ["das4", "das8"], "full_along_U")])
f20 = pd.concat([flips("E20", ["das4", "das8"], "feat_A_along_U"), flips("E20", ["das4", "das8"], "full_along_U")])
for task, L in (("add_chain", 18), ("entity_track", 12)):
    a = f16[(f16.task == task) & (f16.layer == L)]
    for sae in f20[f20.task == task].sae.unique():
        b = f20[(f20.task == task) & (f20.sae == sae)]
        for est in ("das4", "das8"):
            for cond in ("feat_A_along_U", "full_along_U"):
                x = a[(a.estimator == est) & (a.condition == cond)].set_index("pair_id").flip
                y = b[(b.estimator == est) & (b.condition == cond)].set_index("pair_id").flip
                j = x.index.intersection(y.index); x, y = x.loc[j], y.loc[j]
                n01 = int(((x == 0) & (y == 1)).sum()); n10 = int(((x == 1) & (y == 0)).sum())
                p20.append(dict(task=task, sae_wide=sae, est=est, cond=cond, n=len(j), flip16k=x.mean(), flip_wide=y.mean(),
                                n01=n01, n10=n10, p=binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0))
p20 = pd.DataFrame(p20); p20["p_bh"] = bh(p20.p.values); p20.to_csv(f"{OUT}/e20_paired.csv", index=False)
print(p20.round(4).to_string(index=False))

# ---- E20 geometry: DAS learned subspace inside 16k vs wide error-U
g = []
for task, L, s16, wides in (("RS", 18, "add_chain_L18_L18_16k_l074", ["add_chain_L18_L18_65k_l064"]),
                            ("OT", 12, "entity_track_L12_L12_16k_l082", ["entity_track_L12_L12_65k_l072", "entity_track_L12_L12_262k_l067"])):
    Z16, _ = load("E18", s16)
    for w in [s16] + wides:
        Zw, _ = load("E20" if w != s16 else "E18", w)
        for k in ("das4", "das8"):
            Lr, _, _ = learned(Z16[k], k)
            g.append(dict(task=task, sae=w.split("_L")[-1].split("_", 1)[-1], est=k,
                          **{f"Lrn_in_pca{r}": capture(Lr, Zw["pca_pooled"][:, :r]) for r in (8, 32)},
                          **{f"Lrn_in_as{r}": capture(Lr, Zw["answer_sep"][:, :r]) for r in (8, 32)},
                          pca16k_vs_wide_r8_top=top(Z16["pca_pooled"][:, :8], Zw["pca_pooled"][:, :8]),
                          pca16k_vs_wide_r8_mean=float(pcos(Z16["pca_pooled"][:, :8], Zw["pca_pooled"][:, :8]).mean())))
g = pd.DataFrame(g); g.to_csv(f"{OUT}/e20_geom.csv", index=False)
print("== E20 geometry: learned DAS (identical rotation) captured by each SAE's error-U (Haar: 8/2304=.0035, 32/2304=.014)")
print(g.round(3).to_string(index=False))

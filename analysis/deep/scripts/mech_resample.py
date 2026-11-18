"""Is the SAE error answer-specific? (exploratory, post hoc) x_hat_A + eps from ANOTHER test context that has the same answer as A
(resampC_sameAns) vs a different answer (resampC_diffAns). Paired per pair (same pairs, same features): difference in flip and in
normalised D (per-pair D_cond - D_B over mean(D_A - D_B)); bootstrap 95% CI. E3 (fresh fit) and E2/E2r (pilot)."""
from mech_common import *
pd.set_option("display.width", 250)
rows = []
for exp in ["E3", "E2", "E2r"]:
    pp = load(exp, cols=["pair_id", "task", "layer", "family", "fit_families", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B", "donor_status"])
    pp = pp[pp.estimator.fillna("") == ""]
    if exp == "E3": pp = pp[pp.fit_families == "fresh"]
    pp = wide(pp).drop_duplicates(["pair_id", "condition"])
    for task, g in pp.groupby("task"):
        D = g.pivot_table(index="pair_id", columns="condition", values="D", aggfunc="first")
        A = g.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="first")
        B = g.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="first")
        el = D.index[(A.clean_A == 1) & (B.clean_B == 1)]
        el = el[D.loc[el, ["feat_A_resampC_sameAns", "feat_A_resampC_diffAns"]].notna().all(1)]
        den = (D.full_A - D.clean_B).loc[el].mean()
        for fam_sel, sub in [("all", el)] + [(f, el[g.drop_duplicates("pair_id").set_index("pair_id").loc[el, "family"].values == f]) for f in sorted(g.family.unique())]:
            if len(sub) < 40: continue
            dn = ((D.loc[sub, "feat_A_resampC_sameAns"] - D.loc[sub, "feat_A_resampC_diffAns"]) / den).values
            df_ = (A.loc[sub, "feat_A_resampC_sameAns"].astype(float) - A.loc[sub, "feat_A_resampC_diffAns"].astype(float)).values
            m1, l1, h1 = boot_ci(dn); m2, l2, h2 = boot_ci(df_)
            rows.append(dict(exp=exp, task=task, family=fam_sel, n=len(sub), dnorm_same_minus_diff=m1, lo=l1, hi=h1,
                             dflip=m2, flo=l2, fhi=h2, flip_same=A.loc[sub, "feat_A_resampC_sameAns"].mean(), flip_diff=A.loc[sub, "feat_A_resampC_diffAns"].mean(),
                             flip_errB=A.loc[sub, "feat_A_errB"].mean(), flip_meanPos=A.loc[sub, "feat_A_meanPos"].mean()))
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "resample_same_vs_diff.csv", index=False)
print(t.round(3).to_string(index=False))

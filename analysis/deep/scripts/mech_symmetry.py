"""Feature sufficiency in both directions (exploratory, post hoc). For eligible pairs (clean A -> A, clean B -> B):
  keepA_featA = P(top1 = A | x_hat_A + eps') for eps' in {0, mean, B's error}    (A's features alone give A?)
  keepB_featB = P(top1 = B | x_hat_B + noise)                                   (B's features alone give B?)
  AfromB      = P(top1 = A | x_hat_B + noise)                                   (B's features with its error removed fall back to A)
If features carried the state, keepA and keepB would both be high. A large A/B asymmetry means the 'A' answer is a fallback the
model reaches whenever B-specific information is removed, so feature-only flips to A do not show the features encode A."""
from mech_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
rows = []
for exp in ["E8", "E11", "E1", "E14"]:
    for d in run_dirs(exp):
        pp = pd.read_parquet(d / "per_pair.parquet", columns=["pair_id", "task", "layer", "sae", "fit_families", "family", "condition", "estimator", "seed", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
        pp = pp[pp.estimator.fillna("") == ""]
        if exp == "E14":
            pp = pp[pp.fit_families == pp.family]
        pp = wide(pp)
        for (task, layer, sae, fam), g in pp.groupby(["task", "layer", "sae", "family"]):
            piv = g.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="mean")
            pivB = g.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="mean")
            if "feat_B_noise" not in piv or "clean_A" not in piv: continue
            el = piv.index[(piv.clean_A == 1) & (pivB.clean_B == 1)]
            if len(el) < 20: continue
            r = dict(exp=exp, run=d.name[:6], task=task, layer=layer, sae=sae, family=fam, n=len(el))
            for c in ["feat_A_errB", "feat_A_zero", "feat_A_meanPos", "feat_A_noise", "feat_B_errA"]:
                if c in piv: r[f"A|{c}"] = piv.loc[el, c].mean()
            r["B|feat_B_noise"] = pivB.loc[el, "feat_B_noise"].mean(); r["A|feat_B_noise"] = piv.loc[el, "feat_B_noise"].mean()
            rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(SP / "deep" / "symmetry.csv", index=False)
print(t.round(3).to_string(index=False))

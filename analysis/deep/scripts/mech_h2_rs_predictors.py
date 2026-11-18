"""H2 (exploratory, post hoc): which RS L18 pairs lose A under the feature-only patch? E3 fresh-fit run (737 eligible pairs, all
5 families). Predictors from the prompts: A, B, A - B (sign, |.|), corrupt step (0..2), distance from the first changed token to
the end (T - first_diff), template family, clean margin D_A - D_B. Outcomes: fail = top-1 != A under feat_A_errB / feat_A_meanPos;
continuous r_i. Logistic regression (statsmodels) with cluster-robust SEs by family is not possible with 5 clusters, so plain
SEs + a per-predictor bootstrap of the marginal effect."""
from mech_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
pp = load("E3", cols=["pair_id", "task", "fit_families", "family", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
pp = wide(pp[(pp.task == "add_chain") & (pp.fit_families == "fresh") & (pp.estimator.fillna("") == "")]).drop_duplicates(["pair_id", "condition"])
D = pp.pivot_table(index="pair_id", columns="condition", values="D", aggfunc="first")
A = pp.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="first")
B = pp.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="first")
el = D.index[(A.clean_A == 1) & (B.clean_B == 1)]
pr = pd.concat([prompts("add_chain_e3"), prompts("add_chain_pilot")]).drop_duplicates("pair_id").set_index("pair_id").loc[el]
df = pd.DataFrame(index=el)
df["family"] = pr.family; df["A"] = pr.state_clean.astype(int); df["B"] = pr.state_corrupt.astype(int)
df["AmB"] = df.A - df.B; df["absd"] = df.AmB.abs(); df["Agt"] = (df.AmB > 0).astype(int)
df["cstep"] = pr.meta.apply(lambda m: m["corrupt_step"]); df["dist"] = pr["T"].astype(int) - pr.first_diff.astype(int)
df["margin"] = D.loc[el, "full_A"] - D.loc[el, "clean_B"]
for c in ["feat_A_errB", "feat_A_meanPos"]:
    df[f"fail_{c}"] = (A.loc[el, c] == 0).astype(int)
    df[f"r_{c}"] = (D.loc[el, c] - D.loc[el, "clean_B"]) / df.margin
print("n", len(df)); print(df.describe().round(2).to_string())
print(df.groupby("cstep")[["fail_feat_A_errB", "fail_feat_A_meanPos", "r_feat_A_errB", "dist"]].agg(["mean", "count"]).round(3))
print(df.groupby("A")[["fail_feat_A_errB", "fail_feat_A_meanPos", "r_feat_A_errB"]].agg(["mean", "count"]).round(3))
print(df.groupby("AmB")[["fail_feat_A_errB", "fail_feat_A_meanPos", "r_feat_A_errB"]].agg(["mean", "count"]).round(3))
print(df.groupby("family")[["fail_feat_A_errB", "fail_feat_A_meanPos", "r_feat_A_errB"]].agg(["mean", "count"]).round(3))
Xb = pd.get_dummies(df[["A", "Agt", "absd", "cstep", "family"]].assign(cstep=df.cstep.astype(str)), columns=["cstep", "family"], drop_first=True).astype(float)
Xb["log_margin"] = np.log(df.margin)
for c in ["feat_A_errB", "feat_A_meanPos"]:
    print("== logit fail", c); print(logit_irls(Xb, df[f"fail_{c}"]).round(3).to_string())
    t, r2 = ols(Xb, df[f"r_{c}"]); print("== OLS r_i", c, "R2", round(r2, 3)); print(t.round(4).to_string())
df.to_csv(SP / "deep" / "h2_rs_pairs.csv")

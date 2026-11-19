"""E20 (65k/262k) vs E9 (16k): E9 outcomes by SAE width, from the make_final tables (no recomputation). Post hoc, exploratory.
Out: team/geometry_out/e20_vs_e9_alongU.csv, e20_vs_e9_gaps.csv"""
import pandas as pd, numpy as np
SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
OUT = f"{SP}/team/geometry_out"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
ill = pd.concat([pd.read_csv(f"{SP}/hw/tables/final/E9_illusion.csv"), pd.read_csv(f"{SP}/r5/tables/E20_illusion.csv")])
con = pd.concat([pd.read_csv(f"{SP}/hw/tables/final/E9_conditions.csv"), pd.read_csv(f"{SP}/r5/tables/E20_conditions.csv")])
con = con[con.family.astype(str) == "pilot"]
# 1. gaps
g = con[con.condition.isin(["feat_A_errB", "feat_A_meanPos", "feat_A_zero", "full_A"])]
g = g[["exp_id", "task", "sae", "filter", "condition", "n_pairs", "flip_n_eligible", "flip_rate", "gap_flip_rate", "gap_flip_rate_lo",
       "gap_flip_rate_hi", "gap_nldr", "gap_nldr_lo", "gap_nldr_hi"]].drop_duplicates()
g.to_csv(f"{OUT}/e20_vs_e9_gaps.csv", index=False)
print(g[g["filter"].isin(["f_pilot", "f_none"])].sort_values(["task", "filter", "condition", "sae"]).round(3).to_string(index=False))
# 2. along-U / repair
cols = ["exp_id", "task", "sae", "filter", "condition", "estimator", "r", "n_pairs", "flip_rate", "flip_rate_lo", "flip_rate_hi", "nldr",
        "random_p95", "effective", "das_train_acc", "illusion_diff", "illusion_diff_lo", "illusion_diff_hi", "full_along_U_effective",
        "feature_mediated", "illusion", "cos_top_das"]
x = ill[(ill.r > 0)][cols]
x.to_csv(f"{OUT}/e20_vs_e9_alongU.csv", index=False)
def cellstr(r):
    return f"{r.flip_rate:.3f}{'E' if r.effective is True else ''}|{r.nldr:.3f}|p95 {r.random_p95:.3f}" if pd.notna(r.random_p95) else f"{r.flip_rate:.3f}|{r.nldr:.3f}"
for flt in ["f_pilot", "f_none"]:
    y = x[x["filter"] == flt].copy(); y["s"] = [cellstr(r) for r in y.itertuples()]
    for t, yt in y.groupby("task"):
        print("==", flt, t, "cells: flip[E=effective]|nldr|random p95 flip")
        print(yt.pivot_table(index=["estimator", "r", "condition"], columns="sae", values="s", aggfunc="first").to_string())
        fl = yt[yt.condition == "full_along_U"].copy()
        fl["lab"] = [f"ill={r.illusion} fm={r.feature_mediated} diff={r.illusion_diff:.3f}[{r.illusion_diff_lo:.3f},{r.illusion_diff_hi:.3f}]" for r in fl.itertuples()]
        print(fl.pivot_table(index=["estimator", "r"], columns="sae", values="lab", aggfunc="first").to_string())

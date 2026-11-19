"""E20 standard readout: E9 rule outcomes with wider SAEs at matched L0 (RS L18 65k l064; OT L12 65k l072 and 262k l067) side
by side with E9's 16k outcomes (same cell, filter, estimator, rank). Inputs: r5/tables/E20_illusion.csv, E20_conditions.csv,
hw/tables/final/E9_illusion.csv, E9_conditions.csv. Outputs: team/standard_out/E20_*.csv."""
from standard_common import *

e20 = pd.read_csv(R5 / "E20_illusion.csv")
e9 = pd.read_csv(FINAL / "E9_illusion.csv")
mism, nchk = recheck_rules(e20)
print(f"rule recheck (effective) E20: {mism} disagreements of {nchk}")
e20 = add_p_eff(e20)
pp = load_pp("E20", ["clean_B", "full_along_U", "feat_A_along_U"])
mc = mcnemar_rows(pp)
x = e20[e20.r > 0]
tests = pd.concat([
    x.assign(test="eff:" + x.condition)[["sae", "filter", "estimator", "r", "test", "p_eff"]].rename(columns={"p_eff": "p"}),
    mc.assign(test="illusion_mcnemar")[["sae", "filter", "estimator", "r", "test", "p_ill"]].rename(columns={"p_ill": "p"})])
tests["q_bh"] = bh(tests.p)
tests.to_csv(OUT / "E20_tests.csv", index=False)
print(f"E20 tests: {len(tests)} (BH over all); q<.05: {(tests.q_bh < .05).sum()}")

# baselines
c20 = pd.read_csv(R5 / "E20_conditions.csv")
c9 = pd.read_csv(FINAL / "E9_conditions.csv")
bc = ["full_A", "feat_A_errB", "feat_A_meanPos", "feat_A_zero"]
b = pd.concat([c9[c9.condition.isin(bc) & c9.estimator.isna()], c20[c20.condition.isin(bc) & c20.estimator.isna()]])
print("\nBaselines flip rate [lo,hi] (feat_A_errB gap = full_A - feat_A_errB):")
b["v"] = [f"{f:.3f}[{l:.3f},{h:.3f}]" for f, l, h in zip(b.flip_rate, b.flip_rate_lo, b.flip_rate_hi)]
print(b.pivot_table(index=["task", "sae", "filter"], columns="condition", values="v", aggfunc="first").to_string())
g = b[b.condition == "feat_A_errB"][["task", "sae", "filter", "gap_flip_rate", "gap_flip_rate_lo", "gap_flip_rate_hi"]]
print(g.round(3).to_string())

# rule outcomes side by side
def outcomes(t, tag):
    y = t[(t.condition == "full_along_U") & (t.r > 0)].copy()
    feat = t[(t.condition == "feat_A_along_U") & (t.r > 0)].set_index(["task", "filter", "estimator", "r"])
    rep = t[(t.condition == "repair") & (t.r > 0)].set_index(["task", "filter", "estimator", "r"])
    y = y.set_index(["task", "filter", "estimator", "r"])
    o = pd.DataFrame({f"sae": y.sae, "full": y.flip_rate, "full_lo": y.flip_rate_lo, "p95_full": y.random_p95,
                      "feat": feat.flip_rate, "feat_lo": feat.flip_rate_lo, "p95_feat": feat.random_p95,
                      "repair": rep.flip_rate, "p95_rep": rep.random_p95, "rep_eff": rep.effective.map(flag),
                      "diff": y.illusion_diff, "diff_lo": y.illusion_diff_lo, "acc": y.das_train_acc,
                      "eff_full": y.full_along_U_effective.map(flag), "feat_med": y.feature_mediated.map(flag),
                      "illusion": y.illusion.map(flag)}).reset_index()
    o["tag"] = tag
    return o

O = pd.concat([outcomes(e9, "16k(E9)")] + [outcomes(e20[e20.sae == s], s) for s in e20.sae.unique()])
O = O.sort_values(["task", "filter", "estimator", "r", "tag"])
O.to_csv(OUT / "E20_vs_E9.csv", index=False)
for flt in FILTERS:
    print(f"\n==== {flt}: full / feat (p95) | eff_full feat_med illusion | repair eff")
    s = O[O["filter"] == flt].copy()
    s["v"] = [f"{a:.3f}/{f:.3f} ({p:.3f}) {e}{m}{i} rep{r:.3f}/{rp:.3f}{re}" for a, f, p, e, m, i, r, rp, re in
              zip(s.full, s.feat, s.p95_full, s.eff_full, s.feat_med, s.illusion, s.repair, s.p95_rep, s.rep_eff)]
    print(s.pivot_table(index=["task", "estimator", "r"], columns="tag", values="v", aggfunc="first").to_string())

# change counts vs 16k
key = ["task", "filter", "estimator", "r"]
base = O[O.tag == "16k(E9)"].set_index(key)
print("\n==== outcome changes vs 16k (E9)")
for tag in [t for t in O.tag.unique() if t != "16k(E9)"]:
    w = O[O.tag == tag].set_index(key)
    j = w.join(base, rsuffix="_16k", how="inner")
    for col in ["eff_full", "feat_med", "illusion"]:
        ch = j[j[col] != j[f"{col}_16k"]]
        print(f"{tag} {col}: {len(ch)} changes of {len(j)}")
        if len(ch):
            print(ch[[col, f"{col}_16k", "full", "full_16k", "feat", "feat_16k"]].round(3).to_string())
print("\nDAS fits (train acc):")
print(O[O.estimator.str.startswith("das") & (O["filter"] == "f_pilot")][["task", "tag", "estimator", "acc"]].to_string())

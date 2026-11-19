"""E22 standard readout: multilingual DAS/PCA fitted natively vs on English (RS L18: fr, hinglish, hi; PC L12: fr, hi, hinglish),
E9 rules on each fit, E13 gates (test split) alongside. Paired exact McNemar (two-sided) for English-fit minus native-fit flips on
the same evaluation pairs; one-sided exact binomial H0 flip <= p95 + .10 for each row; BH over all E22 tests.
Inputs: r5/tables/E22_illusion.csv, E22_transfer.csv, hw/tables/final/E13_behaviour_lang.csv, E22 per_pair."""
from standard_common import *

t = pd.read_csv(R5 / "E22_illusion.csv")
mism, nchk = recheck_rules(t)
print(f"rule recheck (effective): {mism} disagreements of {nchk}")
t = add_p_eff(t)
t["fit"] = np.where(t.fit_families == t.family, "native", np.where(t.fit_families.str.endswith("_en"), "english", "other"))
x = t[(t.r > 0) & t.condition.isin(["full_along_U", "feat_A_along_U"])].copy()

# per-pair: native vs English fit on the same pairs, and full-vs-feat McNemar per fit
pp = load_pp("E22", ["clean_B", "full_along_U", "feat_A_along_U"])
mc = mcnemar_rows(pp)
tr_rows = []
for (task, fam), g in pp.groupby(["task", "family"]):
    if fam.endswith("_en"):
        continue
    base = g[(g.condition == "clean_B")].drop_duplicates("pair_id").set_index("pair_id")
    elig = base.top1_id == base.ans_B_id
    for cond in ["full_along_U", "feat_A_along_U"]:
        for est in ["das4", "das8", "pca_pooled"]:
            for r in sorted(g[(g.estimator == est) & (g.r > 0)].r.unique()):
                nat = g[(g.condition == cond) & (g.estimator == est) & (g.r == r) & (g.fit_families == fam)].drop_duplicates("pair_id").set_index("pair_id")
                eng = g[(g.condition == cond) & (g.estimator == est) & (g.r == r) & g.fit_families.str.endswith("_en")].drop_duplicates("pair_id").set_index("pair_id")
                for flt in FILTERS:
                    ids = base.index[base[flt].astype(bool) & elig].intersection(nat.index).intersection(eng.index)
                    a = (eng.loc[ids].top1_id == eng.loc[ids].ans_A_id).to_numpy()
                    b = (nat.loc[ids].top1_id == nat.loc[ids].ans_A_id).to_numpy()
                    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
                    p = float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                    tr_rows.append({"task": task, "family": fam, "filter": flt, "condition": cond, "estimator": est, "r": int(r),
                                    "n_elig": len(ids), "flip_english": a.mean(), "flip_native": b.mean(),
                                    "eng_minus_nat": a.mean() - b.mean(), "n_eng_only": n10, "n_nat_only": n01, "p_transfer": p})
TR = pd.DataFrame(tr_rows)
tests = pd.concat([
    x.assign(test="eff:" + x.condition + ":" + x.fit)[["task", "family", "filter", "estimator", "r", "test", "p_eff"]].rename(columns={"p_eff": "p"}),
    mc.assign(test="illusion_mcnemar:" + np.where(mc.fit_families == mc.family, "native", "english"))[["task", "family", "filter", "estimator", "r", "test", "p_ill"]].rename(columns={"p_ill": "p"}),
    TR.assign(test="transfer:" + TR.condition)[["task", "family", "filter", "estimator", "r", "test", "p_transfer"]].rename(columns={"p_transfer": "p"})])
tests["q_bh"] = bh(tests.p)
tests.to_csv(OUT / "E22_tests.csv", index=False)
print(f"E22 tests: {len(tests)} (BH over all); q<.05: {(tests.q_bh < .05).sum()}")
q = tests.set_index(["task", "family", "filter", "estimator", "r", "test"]).q_bh
TR["q_transfer"] = [q.get((a, f, fl, e, r, "transfer:" + c), np.nan) for a, f, fl, e, r, c in
                    zip(TR.task, TR.family, TR["filter"], TR.estimator, TR.r, TR.condition)]
# cross-check against shipped E22_transfer
ship = pd.read_csv(R5 / "E22_transfer.csv")
m = TR.merge(ship, on=["task", "family", "filter", "condition", "estimator", "r"])
print("max |my native flip - shipped|:", np.nanmax(np.abs(m.flip_native - m.flip_rate_native)),
      " max |my english flip - shipped|:", np.nanmax(np.abs(m.flip_english - m.flip_rate_english)), "rows", len(m))

# E13 gates (test split governs)
g13 = pd.read_csv(FINAL / "E13_behaviour_lang.csv")
g13 = g13[g13.split == "test"].set_index("family")
TR["gate_A"] = TR.family.map(g13.gate_A_viable)
TR["gate_B"] = TR.family.map(g13.gate_B_within_10pts)
TR["acc_clean"] = TR.family.map(g13.acc_clean)
TR["readable"] = TR.gate_A.astype(str).eq("True") & TR.gate_B.astype(str).eq("True") & TR.family.map(g13.validated).astype(bool)

# rule outcomes per fit
ro = x[x.condition == "full_along_U"][["task", "family", "filter", "estimator", "r", "fit", "flip_rate", "das_train_acc", "das_fit_valid",
                                        "full_along_U_effective", "feature_mediated", "illusion", "illusion_diff", "illusion_diff_lo", "random_p95"]]
ro = ro.assign(rules=[f"{flag(a)}{flag(b)}{flag(c)}" for a, b, c in zip(ro.full_along_U_effective, ro.feature_mediated, ro.illusion)])
piv = ro.pivot_table(index=["task", "family", "filter", "estimator", "r"], columns="fit",
                     values=["rules", "das_train_acc"], aggfunc="first")
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index()
F = TR[TR.condition == "full_along_U"].merge(
    TR[TR.condition == "feat_A_along_U"][["task", "family", "filter", "estimator", "r", "flip_english", "flip_native", "eng_minus_nat", "q_transfer"]],
    on=["task", "family", "filter", "estimator", "r"], suffixes=("_full", "_feat")).merge(piv, on=["task", "family", "filter", "estimator", "r"], how="left")
F.to_csv(OUT / "E22_readout.csv", index=False)
cols = ["task", "family", "estimator", "r", "acc_clean", "gate_A", "gate_B", "n_elig", "das_train_acc_native", "das_train_acc_english",
        "flip_native_full", "flip_english_full", "eng_minus_nat_full", "q_transfer_full", "flip_native_feat", "flip_english_feat",
        "eng_minus_nat_feat", "q_transfer_feat", "rules_native", "rules_english"]
for flt in FILTERS:
    print(f"\n==== {flt} (rules = eff_full / feature_mediated / illusion)")
    print(F[F["filter"] == flt][cols].round(3).to_string())

# English rows on the multilingual English pack (reference for each task)
en = ro[ro.family.str.endswith("_en")]
print("\n==== English-pack reference rows")
print(en[["task", "family", "filter", "estimator", "r", "flip_rate", "random_p95", "das_train_acc", "rules", "illusion_diff", "illusion_diff_lo"]].round(3).to_string())

# filter disagreement on rules
for fit in ("native", "english"):
    pv = ro[ro.fit == fit].pivot_table(index=["task", "family", "estimator", "r"], columns="filter", values="rules", aggfunc="first")
    d = pv[pv.nunique(axis=1) > 1]
    print(f"\nfilter disagreements on rules ({fit} fit):\n", d.to_string() if len(d) else "none")

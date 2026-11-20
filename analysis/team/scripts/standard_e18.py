"""E18 standard readout: rank curves (r = 1..64) of full_along_U / feat_A_along_U / repair for pca_pooled and answer_sep vs
random p95, and DAS rank 1..16, per cell (RS L18, OT L12, PC L12, PC L16), E9 rules. Inputs: r5/tables/E18_illusion.csv,
E18_conditions.csv (baselines), E18 per_pair (McNemar). Outputs: team/standard_out/E18_*.csv + printed readout."""
from standard_common import *

t = pd.read_csv(R5 / "E18_illusion.csv")
c = pd.read_csv(R5 / "E18_conditions.csv")
mism, nchk = recheck_rules(t)
print(f"rule recheck (effective): {mism} disagreements of {nchk} evaluable rows")
t = add_p_eff(t)
t["cell"] = t.task.str.replace("add_chain", "RS").str.replace("entity_track", "OT").str.replace("people_cities", "PC") + " L" + t.layer.astype(str)
c["cell"] = c.task.str.replace("add_chain", "RS").str.replace("entity_track", "OT").str.replace("people_cities", "PC") + " L" + c.layer.astype(str)

# baselines per cell/filter
b = c[c.condition.isin(["full_A", "feat_A_errB", "feat_A_meanPos", "clean_B"]) & c.estimator.isna()]
print("\nBaselines (flip rate):")
print(b.pivot_table(index=["cell", "filter"], columns="condition", values="flip_rate").round(3).to_string())
print(b.groupby(["cell", "filter"])[["n_pairs", "flip_n_eligible"]].first().to_string())

pp = load_pp("E18", ["clean_B", "full_along_U", "feat_A_along_U"])
mc = mcnemar_rows(pp)
mc["cell"] = mc.task.str.replace("add_chain", "RS").str.replace("entity_track", "OT").str.replace("people_cities", "PC") + " L" + mc.layer.astype(str)
x = t[(t.r > 0)].copy()
chk = x[x.condition != "repair"].merge(mc, on=["cell", "filter", "estimator", "r"])
dd = np.where(chk.condition == "full_along_U", chk.flip_full_mine, chk.flip_feat_mine) - chk.flip_rate
print("max |my flip - table flip|:", np.nanmax(np.abs(dd)), "rows", len(chk))

tests = pd.concat([
    x.assign(test="eff:" + x.condition)[["cell", "filter", "estimator", "r", "test", "p_eff"]].rename(columns={"p_eff": "p"}),
    mc.assign(test="illusion_mcnemar")[["cell", "filter", "estimator", "r", "test", "p_ill"]].rename(columns={"p_ill": "p"})])
tests["q_bh"] = bh(tests.p)
tests.to_csv(OUT / "E18_tests.csv", index=False)
print(f"E18 tests: {len(tests)} (BH over all); rejections q<.05: {(tests.q_bh < .05).sum()}")
q = tests.set_index(["cell", "filter", "estimator", "r", "test"]).q_bh
x["q_eff"] = [q.get((a, f, e, r, "eff:" + cd), np.nan) for a, f, e, r, cd in zip(x.cell, x["filter"], x.estimator, x.r, x.condition)]
x["eff_flag"] = x.effective.map(flag)
x.to_csv(OUT / "E18_rows.csv", index=False)

# rank-curve tables (f_pilot first): "flip / p95 / flag"
for flt in FILTERS:
    print(f"\n==== rank curves, filter {flt}  (cell: flip [lo] p95 eff)")
    for est in ["pca_pooled", "answer_sep"]:
        for cond in ["full_along_U", "feat_A_along_U", "repair"]:
            s = x[(x["filter"] == flt) & (x.estimator == est) & (x.condition == cond)]
            s = s.assign(v=[f"{f:.3f}[{lo:.3f}]/{p:.3f}{'*' if e == 'T' else ''}" for f, lo, p, e in
                            zip(s.flip_rate, s.flip_rate_lo, s.random_p95, s.eff_flag)])
            print(f"-- {est} {cond}")
            print(s.pivot_table(index="cell", columns="r", values="v", aggfunc="first").to_string())

# summary: first rank at which error-U is effective, per cell/filter/estimator/condition
summ = []
for (cell, flt, est, cond), s in x[x.estimator.isin(["pca_pooled", "answer_sep"])].groupby(["cell", "filter", "estimator", "condition"]):
    s = s.sort_values("r")
    effr = s[s.eff_flag == "T"].r.tolist()
    summ.append({"cell": cell, "filter": flt, "estimator": est, "condition": cond, "effective_ranks": effr,
                 "max_flip": s.flip_rate.max(), "r_at_max": int(s.loc[s.flip_rate.idxmax(), "r"]),
                 "max_flip_minus_p95": (s.flip_rate - s.random_p95).max()})
S = pd.DataFrame(summ)
S.to_csv(OUT / "E18_error_U_summary.csv", index=False)
print("\n==== error-U: ranks where effective (E9 rule)")
print(S.round(3).to_string())

# DAS per cell
d = x[x.estimator.str.startswith("das")].copy()
fu = d[d.condition == "full_along_U"].set_index(["cell", "filter", "estimator"])
fe = d[d.condition == "feat_A_along_U"].set_index(["cell", "filter", "estimator"])
rp = d[d.condition == "repair"].set_index(["cell", "filter", "estimator"])
mi = mc[mc.estimator.str.startswith("das")].set_index(["cell", "filter", "estimator"])
D = pd.DataFrame({"r": fu.r, "train_acc": fu.das_train_acc, "fit_valid": fu.das_fit_valid.map(flag),
                  "full": fu.flip_rate, "full_lo": fu.flip_rate_lo, "p95_full": fu.random_p95,
                  "feat": fe.flip_rate, "feat_lo": fe.flip_rate_lo, "p95_feat": fe.random_p95,
                  "repair": rp.flip_rate, "p95_repair": rp.random_p95,
                  "diff": fu.illusion_diff, "diff_lo": fu.illusion_diff_lo, "diff_hi": fu.illusion_diff_hi,
                  "eff_full": fu.full_along_U_effective.map(flag), "feat_med": fu.feature_mediated.map(flag),
                  "illusion": fu.illusion.map(flag), "n10": mi.n10, "n01": mi.n01, "q_eff_full": fu.q_eff,
                  "q_eff_feat": fe.q_eff}).reset_index()
D["q_ill"] = [q.get((a, f, e, r, "illusion_mcnemar"), np.nan) for a, f, e, r in zip(D.cell, D["filter"], D.estimator, D.r)]
D = D.sort_values(["cell", "filter", "r"])
D.to_csv(OUT / "E18_das.csv", index=False)
print("\n==== DAS rank 1..16 per cell")
for flt in FILTERS:
    print(f"-- {flt}")
    print(D[D["filter"] == flt].drop(columns="filter").round(3).to_string())

# filter disagreement on the rule outcomes
y = x[x.condition != "repair"]
piv = y.pivot_table(index=["cell", "estimator", "r", "condition"], columns="filter", values="eff_flag", aggfunc="first")
dis = piv[piv.nunique(axis=1) > 1]
print("\n==== filter disagreements on effective (cell, estimator, r, condition):")
print(dis.to_string() if len(dis) else "none")
ill = t[t.condition == "full_along_U"].assign(ill=lambda z: z.illusion.map(flag)).pivot_table(index=["cell", "estimator", "r"], columns="filter", values="ill", aggfunc="first")
print("\n==== filter disagreements on illusion:")
print(ill[ill.nunique(axis=1) > 1].to_string())

# DAS alignment with error-U (cos_top_das on the pca/answer_sep rows at rank k)
al = t[t.estimator.isin(["pca_pooled", "answer_sep"]) & (t.condition == "full_along_U") & (t["filter"] == "f_pilot") & t.cos_top_das.notna()]
print("\n==== error-U vs DAS_k alignment (top principal cosine; addendum: aligned >= .5, unrelated <= Haar p99)")
print(al.pivot_table(index=["cell", "r"], columns="estimator", values="cos_top_das").round(3).to_string())
print(al.pivot_table(index=["cell", "r"], columns="estimator", values="das_alignment", aggfunc="first").to_string())

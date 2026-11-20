"""H5: when the SAE-only patch fails in running sum, which answer does the model give? (exploratory, post hoc)"""
from mech_common import *
import sys
CANDS = [235276, 235274, 235284, 235304, 235310, 235308, 235318, 235324, 235321, 235315]
DIG = {t: i for i, t in enumerate(CANDS)}
out = []
for exp in ["E2", "E3", "E9", "E16", "E8"]:
    for d in run_dirs(exp):
        pp = pd.read_parquet(d / "per_pair.parquet")
        pp = pp[(pp.task == "add_chain") & (pp.estimator.isin(["", None]) | pp.estimator.isna()) ]
        if pp.empty: continue
        pp = wide(pp)
        for (layer, sae, fit, fam), g in pp.groupby(["layer", "sae", "fit_families", "family"]):
            cb = g[g.condition == "clean_B"].set_index("pair_id")
            elig = cb.index[cb.isB & g[g.condition == "clean_B"].set_index("pair_id").f_A_correct]
            for cond in ["feat_A_errB", "feat_A_meanPos", "feat_A_zero", "feat_A_noise", "full_A"]:
                c = g[(g.condition == cond)]
                if c.empty: continue
                c = c.drop_duplicates("pair_id").set_index("pair_id").loc[lambda x: x.index.isin(elig)]
                a = c.ans_A_id.map(DIG); b = c.ans_B_id.map(DIG); t = c.top1_id.map(DIG)
                fail = ~c.isA
                kind = np.where(c.isA, "A", np.where(c.isB, "B", np.where(t.isna(), "nondigit",
                        np.where((t - a).abs() == 1, "A+-1", "other_digit"))))
                vc = pd.Series(kind).value_counts()
                # among non-A failures with digit answers: is the wrong digit between A and B (interpolation)?
                o = (kind == "other_digit") | (kind == "A+-1")
                between = ((t > np.minimum(a, b)) & (t < np.maximum(a, b)))[o].mean() if o.sum() else np.nan
                towardB = (np.sign(t - a) == np.sign(b - a))[o].mean() if o.sum() else np.nan
                out.append(dict(exp=exp, run=d.name, layer=layer, sae=sae, fit=fit, family=fam, cond=cond, n=len(c),
                                A=vc.get("A", 0), B=vc.get("B", 0), Apm1=vc.get("A+-1", 0), other=vc.get("other_digit", 0),
                                nondigit=vc.get("nondigit", 0), frac_between=between, frac_towardB=towardB))
t = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
t.to_csv(SP / "deep" / "h5_rs_candidates.csv", index=False)
print(t[t.cond.isin(["feat_A_errB", "feat_A_meanPos"])].to_string(index=False))

"""Uniform-shrink model (exploratory, post hoc). Hypothesis: the feature-only patch behaves like a linear interpolation of the
candidate logits, l' = l_B + a (l_A - l_B), with ONE scalar a per cell (a = the cell's NLDR, ratio of means, fitted on the same
pairs: in-sample; a cross-fitted variant uses a from the other half). Predicted top-1 = argmax over candidates of l'.
Report: observed vs predicted flip rate, per-pair agreement of top-1 class, and for RS whether predicted wrong answers match
observed wrong answers. Baseline: predict 'A' for every pair."""
from mech_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
rows = []
for exp in ["E8", "E11", "E3"]:
    pp = load(exp, cols=["pair_id", "task", "layer", "family", "fit_families", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B", "cand_logits"])
    pp = pp[pp.estimator.fillna("") == ""]
    if exp == "E3": pp = pp[pp.fit_families == "fresh"]
    pp = wide(pp).drop_duplicates(["pair_id", "layer", "condition"])
    for (task, layer), g in pp.groupby(["task", "layer"]):
        if exp == "E11" and task != "people_cities": continue
        G = {c: x.set_index("pair_id") for c, x in g.groupby("condition")}
        el = G["clean_A"].index[(G["clean_A"].isA) & (G["clean_B"].isB.reindex(G["clean_A"].index) == True)]
        for cond in ["feat_A_errB", "feat_A_meanPos", "feat_A_zero"]:
            if cond not in G: continue
            P = G[cond].loc[el]; LA = np.stack(G["full_A"].loc[el].cand_logits.values); LB = np.stack(G["clean_B"].loc[el].cand_logits.values)
            Lc = np.stack(P.cand_logits.values)
            DA = G["full_A"].loc[el].D.values; DB = G["clean_B"].loc[el].D.values; Dc = P.D.values
            ok = np.isfinite(LA).all(0) & np.isfinite(LB).all(0)
            LA, LB, Lc = LA[:, ok], LB[:, ok], Lc[:, ok]
            half = np.arange(len(el)) % 2 == 0
            def a_of(m): return (Dc[m] - DB[m]).mean() / (DA[m] - DB[m]).mean()
            a_in = a_of(np.ones(len(el), bool))
            a_cf = np.where(half, a_of(~half), a_of(half))
            pred = np.argmax(LB + a_cf[:, None] * (LA - LB), 1)
            obs = np.argmax(Lc, 1)
            iA = np.argmax(LA, 1); iB = np.argmax(LB, 1)
            cls = lambda x: np.where(x == iA, "A", np.where(x == iB, "B", "other"))
            pc, oc = cls(pred), cls(obs)
            rows.append(dict(exp=exp, task=task, layer=layer, cond=cond, n=len(el), a=a_in, flip_obs=(oc == "A").mean(), flip_pred=(pc == "A").mean(),
                             other_obs=(oc == "other").mean(), other_pred=(pc == "other").mean(), agree_class=(pc == oc).mean(),
                             agree_token=(pred == obs).mean(), base_agree=(oc == "A").mean(),
                             fail_recall=((pc != "A") & (oc != "A")).sum() / max(1, (oc != "A").sum()),
                             fail_precision=((pc != "A") & (oc != "A")).sum() / max(1, (pc != "A").sum())))
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "interp_model.csv", index=False)
print(t[t.cond == "feat_A_errB"].round(3).to_string(index=False))
print(t[(t.cond != "feat_A_errB") & (t.task == "add_chain") & (t.layer == 18)].round(3).to_string(index=False))

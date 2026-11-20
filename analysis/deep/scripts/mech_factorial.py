"""2x2 factorial decomposition of the A-vs-B logit difference D into a features part, an error part and their interaction
(exploratory, post hoc). Cells per pair: (x_hat_A, eps_A) = full_A, (x_hat_A, eps_B) = feat_A_errB, (x_hat_B, eps_A) = feat_B_errA,
(x_hat_B, eps_B) = full_B. With T = D(AA) - D(BB):
  feature share = [(D(AA) - D(BA)) + (D(AB) - D(BB))] / 2T      error share = [(D(AA) - D(AB)) + (D(BA) - D(BB))] / 2T
  interaction   = [D(AA) - D(AB) - D(BA) + D(BB)] / T  (shares are ratios of means; feature + error = 1).
Also top-1 'conjunctive' rates: P(B | x_hat_B + eps_A) and P(A | x_hat_A + eps_B)."""
from mech_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
C = {"AA": "full_A", "AB": "feat_A_errB", "BA": "feat_B_errA", "BB": "full_B"}


def decomp(piv, n=2000, seed=0):
    D = {k: piv[v].values for k, v in C.items()}
    T = D["AA"] - D["BB"]
    f = (D["AA"] - D["BA"] + D["AB"] - D["BB"]) / 2
    e = (D["AA"] - D["AB"] + D["BA"] - D["BB"]) / 2
    i = D["AA"] - D["AB"] - D["BA"] + D["BB"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(T), (n, len(T)))
    out = {}
    for name, x in [("feat_share", f), ("err_share", e), ("interaction", i)]:
        v = x[idx].mean(1) / T[idx].mean(1)
        out[name] = x.mean() / T.mean(); out[name + "_lo"] = np.percentile(v, 2.5); out[name + "_hi"] = np.percentile(v, 97.5)
    return out


rows = []
for exp in ["E1", "E3", "E2", "E14"]:
    for d in run_dirs(exp):
        pp = pd.read_parquet(d / "per_pair.parquet", columns=["pair_id", "task", "layer", "sae", "fit_families", "family", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
        pp = pp[(pp.estimator.fillna("") == "") & pp.condition.isin(list(C.values()) + ["clean_A", "clean_B"])]
        if exp == "E14":
            pp = pp[pp.fit_families == pp.family]
        if exp == "E3":
            pp = pp[pp.fit_families == "fresh"]
        if pp.condition.nunique() < 6: continue
        pp = wide(pp).drop_duplicates(["pair_id", "layer", "sae", "condition"])
        for (task, layer, sae, fam), g in pp.groupby(["task", "layer", "sae", "family"]):
            piv = g.pivot_table(index="pair_id", columns="condition", values="D", aggfunc="first").dropna()
            iA = g.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="first")
            iB = g.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="first")
            el = piv.index[(iA.loc[piv.index, "clean_A"] == 1) & (iB.loc[piv.index, "clean_B"] == 1)]
            if len(el) < 30: continue
            r = dict(exp=exp, task=task, layer=layer, sae=sae, family=fam, n=len(el), T=float((piv.full_A - piv.full_B).loc[el].mean()))
            r.update(decomp(piv.loc[el]))
            r["P(A|xA+eB)"] = iA.loc[el, "feat_A_errB"].mean(); r["P(B|xB+eA)"] = iB.loc[el, "feat_B_errA"].mean()
            r["P(A|xB+eA)"] = iA.loc[el, "feat_B_errA"].mean()
            rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(SP / "deep" / "factorial.csv", index=False)
cols = ["exp", "task", "layer", "family", "n", "T", "feat_share", "feat_share_lo", "feat_share_hi", "err_share", "interaction", "interaction_lo", "interaction_hi", "P(A|xA+eB)", "P(B|xB+eA)", "P(A|xB+eA)"]
print(t[cols].round(3).to_string(index=False))

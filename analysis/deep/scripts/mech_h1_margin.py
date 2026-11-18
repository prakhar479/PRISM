"""H1 (exploratory, post hoc): is the SAE-only loss a uniform shrink of the logit difference or a tail of failing pairs?
Per pair: r_i = (D(x_hat_A + eps_B) - D_B) / (D_A - D_B). Report quantiles, share of pairs with r_i < 0.5, the ratio-of-means NLDR,
the flip gap, and the Spearman correlation of r_i with the clean margin D_A - D_B. E8 grid (4 tasks x layers) + E11 PC."""
from mech_common import *
from scipy.stats import spearmanr
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
rows = []
for exp in ["E8", "E11"]:
    pp = load(exp, cols=["pair_id", "task", "layer", "family", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
    pp = wide(pp[pp.estimator.fillna("") == ""]).drop_duplicates(["pair_id", "layer", "condition"])
    for (task, layer), g in pp.groupby(["task", "layer"]):
        if exp == "E11" and task != "people_cities": continue
        D = g.pivot_table(index="pair_id", columns="condition", values="D", aggfunc="first")
        A = g.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="first")
        B = g.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="first")
        el = D.index[(A.clean_A == 1) & (B.clean_B == 1)]
        den = D.loc[el, "full_A"] - D.loc[el, "clean_B"]
        for cond in ["feat_A_errB", "feat_A_meanPos"]:
            ri = (D.loc[el, cond] - D.loc[el, "clean_B"]) / den
            rho = spearmanr(ri, den).statistic
            rows.append(dict(task=task, layer=layer, cond=cond, n=len(el), nldr=float((D.loc[el, cond] - D.loc[el, "clean_B"]).mean() / den.mean()),
                             q10=ri.quantile(.1), q50=ri.median(), q90=ri.quantile(.9), frac_lt_half=(ri < 0.5).mean(),
                             flip_gap=1 - A.loc[el, cond].mean(), rho_margin=rho))
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "h1_margin.csv", index=False)
print(t[t.cond == "feat_A_errB"].round(3).to_string(index=False))

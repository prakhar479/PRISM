"""STATS priority 2 (exploratory): which E17/E20/E22 nulls are informative?
Inputs: team/stats_out/power_contrasts.csv (E17 seed/epoch/lr and E20 width paired contrasts, written by stats_power.py from
per_pair data: B = 2000 pair bootstrap, MDE80 = 2.8 x bootstrap SE), team/stats_out/power_e22_transfer.csv (E22 English-fit vs
native-fit, paired), r5/tables/E22_illusion.csv (E22 language vs English home, UNPAIRED: different prompts).
Classification of a non-significant contrast (95% CI includes 0):
  degenerate     both arms at the floor/ceiling (no discordant pairs / CI [0, 0]): says nothing about the effect of interest
  informative    90% CI inside +-10 points (TOST equivalence at the project's 10-point bar)
  underpowered   otherwise
MDE (alpha .05 two-sided, power .80): paired = 2.8 x bootstrap SE; unpaired (E22 language vs English) = 2.8 x sqrt(p1 q1/n1 +
p2 q2/n2) evaluated at the observed rates, with n = eligible pairs.
Output: team/stats_out/power2_classified.csv, power2_e22_vs_english.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/team/scripts")
from stats_common import OUT, SP, bh  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
Z = sst.norm.ppf(0.975) + sst.norm.ppf(0.8)


def classify(r):
    if r.lo95 > 0 or r.hi95 < 0:
        return "significant"
    if (r.lo95 == 0 and r.hi95 == 0) or (pd.notna(r.get("psi_discordant")) and r.psi_discordant == 0):
        return "degenerate (floor/ceiling)"
    if r.lo90 > -0.10 and r.hi90 < 0.10:
        return "informative null (within +-10 pts)"
    return "underpowered null"


P = pd.read_csv(f"{OUT}/power_contrasts.csv")
P["mde80"] = P.MDE_from_SE
P["verdict"] = P.apply(classify, axis=1)
P["q_BH_within_exp"] = np.nan
for e, g in P.groupby("exp"):
    P.loc[g.index, "q_BH_within_exp"] = bh(g.p.to_numpy())
T = pd.read_csv(f"{OUT}/power_e22_transfer.csv")
T = T.rename(columns={"est": "diff"})
T["exp"], T["kind"] = "E22", "English-fit minus native-fit (paired)"
T["contrast"] = T.family + " " + T.estimator + " r" + T.r.astype(str)
T["mde80"] = (T.hi95 - T.lo95) / (2 * 1.96) * Z
T["verdict"] = T.apply(classify, axis=1)
T["q_BH_within_exp"] = T.q_BH_exp_all
cols = ["exp", "kind", "contrast", "condition", "diff", "lo95", "hi95", "lo90", "hi90", "mde80", "verdict", "q_BH_within_exp"]
A = pd.concat([P[cols + ["n_elig"]], T[cols + ["n_pairs"]].rename(columns={"n_pairs": "n_elig"})], ignore_index=True)
A.to_csv(f"{OUT}/power2_classified.csv", index=False)
print("[A] verdict counts by experiment / kind")
print(A.groupby(["exp", "kind", "verdict"]).size().unstack(fill_value=0).to_string())
print("\n[A] MDE80 (points) by experiment / kind: median [min, max] over contrasts that are not degenerate")
nd = A[~A.verdict.str.startswith("degenerate")]
print(nd.groupby(["exp", "kind"]).mde80.agg(["median", "min", "max", "count"]).round(3).to_string())
print("\n[A] all non-significant, non-degenerate contrasts")
print(A[A.verdict.str.contains("null")][cols + ["n_elig"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- E22 language vs English home (unpaired)
t = pd.read_csv(f"{SP}/r5/tables/E22_illusion.csv")
t = t[(t["filter"] == "f_A_correct") & t.condition.isin(["full_along_U", "feat_A_along_U"]) & (t.r > 0)]
rows = []
for (task, cond, est, r), g in t.groupby(["task", "condition", "estimator", "r"]):
    home = g[g.family.str.endswith("_en") & g.fit_families.str.endswith("_en")]
    if home.empty:
        continue
    h = home.iloc[0]
    for _, x in g[~g.family.str.endswith("_en")].iterrows():
        p1, n1, p2, n2 = x.flip_rate, x.flip_n_eligible, h.flip_rate, h.flip_n_eligible
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        d = p1 - p2
        pv = 2 * sst.norm.sf(abs(d) / se) if se > 0 else (1.0 if d == 0 else 0.0)
        rows.append(dict(task=task, condition=cond, estimator=est, r=r, family=x.family,
                         fit="english" if x.fit_families.endswith("_en") else "native", flip_lang=p1, n_lang=n1, flip_en_home=p2,
                         n_en=n2, diff=d, lo95=d - 1.96 * se, hi95=d + 1.96 * se, lo90=d - 1.645 * se, hi90=d + 1.645 * se, p=pv,
                         mde80=Z * se if se > 0 else np.nan))
E = pd.DataFrame(rows)
E["q_BH"] = bh(E.p.to_numpy())
E["verdict"] = [("degenerate (floor/ceiling)" if (r.flip_lang in (0, 1) and r.flip_en_home in (0, 1)) else
                 "significant" if (r.lo95 > 0 or r.hi95 < 0) else
                 "informative null (within +-10 pts)" if (r.lo90 > -0.1 and r.hi90 < 0.1) else "underpowered null")
                for r in E.itertuples()]
E.to_csv(f"{OUT}/power2_e22_vs_english.csv", index=False)
print("\n[B] E22: language minus English home flip rate (unpaired normal approx.), f_A_correct, BH over", len(E), "tests")
print(E[["task", "condition", "estimator", "r", "family", "fit", "flip_lang", "n_lang", "flip_en_home", "n_en", "diff", "lo95",
         "hi95", "mde80", "q_BH", "verdict"]].round(3).to_string(index=False))
print(E.groupby(["task", "verdict"]).size().unstack(fill_value=0).to_string())

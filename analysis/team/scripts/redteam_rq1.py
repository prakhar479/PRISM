"""RED TEAM (exploratory). Independent recomputation of the RQ1 gate (E3, RS L18, fit fresh, eval fresh, f_pilot) and
sensitivity of the shortfall to the clean margin: shortfall in pairs split at the median clean-A margin over the best other
candidate (A's lead), and per family. Own code (not harness/e8.py)."""
from pathlib import Path
import numpy as np, pandas as pd
SP = __import__("layout_shim").root()
rng = np.random.default_rng(0)
pp = pd.read_parquet(SP / "ana_runs/E3/7bf755d092fa/per_pair.parquet",
                     columns=["pair_id", "task", "family", "fit_families", "condition", "estimator", "seed", "f_pilot", "ans_A_id",
                              "ans_B_id", "top1_id", "logit_A", "logit_B", "cand_logits"])
g = pp[(pp.task == "add_chain") & (pp.fit_families == "fresh") & ~pp.family.str.startswith("pilot") & pp.f_pilot & (pp.estimator.fillna("") == "")]
g = g[g.seed.fillna(-1) <= 0].drop_duplicates(["pair_id", "condition"])
T = g.pivot(index="pair_id", columns="condition", values="top1_id")
a = g.drop_duplicates("pair_id").set_index("pair_id").reindex(T.index)
el = T.index[T.clean_B == a.ans_B_id]
hitF = (T.loc[el, "full_A"] == a.ans_A_id[el]).astype(float)
hitM = (T.loc[el, "feat_A_meanPos"] == a.ans_A_id[el]).astype(float)
hitE = (T.loc[el, "feat_A_errB"] == a.ans_A_id[el]).astype(float)
d = (hitF - hitM).to_numpy()
bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
print("n eligible", len(el), "shortfall meanPos %.3f [%.3f, %.3f]" % (d.mean(), *np.percentile(bs, [2.5, 97.5])), "errB shortfall %.3f" % (hitF - hitE).mean())
# clean-A lead over best other candidate
ca = g[g.condition == "clean_A"].set_index("pair_id").reindex(el)
lead = []
for pid, r in ca.iterrows():
    cl = np.asarray(r.cand_logits, float); la = r.logit_A
    others = np.sort(cl)[::-1]
    lead.append(la - (others[1] if np.isclose(others[0], la) else others[0]))
lead = pd.Series(lead, index=el)
med = lead.median()
for nm, m in (("low lead", lead <= med), ("high lead", lead > med)):
    print(nm, "n", int(m.sum()), "median lead %.2f" % lead[m].median(), "shortfall meanPos %.3f errB %.3f" % ((hitF - hitM)[m].mean(), (hitF - hitE)[m].mean()))
for fam, m in a.loc[el].groupby("family").groups.items():
    print(fam, len(m), "shortfall meanPos %.3f" % (hitF - hitM)[m].mean())

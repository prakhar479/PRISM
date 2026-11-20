"""Replication of the RS start-number effect on independent pairs (E22: rs_en/fr/hi/hinglish, e13 prompt set; E3/E2r not
used because they lack along-U conditions). Spearman of per-pair full_along_U NLDR with start, per family x key."""
import numpy as np, pandas as pd, re
from scipy import stats
import patterns_common as C

d = C.add_refs(C.load("E22", filt=lambda x: (x.task == "add_chain") & x.condition.isin(["clean_A", "clean_B", "full_along_U"]) & (x.estimator != "random")))
d = d[d.A_ok & d.B_ok & (d.condition == "full_along_U")]
d["key"] = C.ckey(d)
sn = C.set_name_for(C.run_dirs("E22")[0], "add_chain")
P = C.prompts(sn)
def start(pid):
    r = P[pid]
    # states field may be absent; parse first integer of the final line (latin digits) else use meta
    nums = re.findall(r"\d+", r["clean_prompt"].rstrip().split("\n")[-1])
    return int(nums[0]) if nums else np.nan
d["start"] = d.pair_id.map(start)
rows = []
for (fam, k), g in d.groupby(["family", "key"]):
    g = g.groupby("pair_id").agg(nldr=("nldr", "mean"), start=("start", "first"), den=("den", "first")).dropna()
    r, p = stats.spearmanr(g.nldr, g.start)
    rows.append(dict(family=fam, key=k, n=len(g), rho_start=r, p=p, mean_start0=g[g.start == 0].nldr.mean(), mean_start_ge2=g[g.start >= 2].nldr.mean(),
                     n_start0=(g.start == 0).sum()))
R = pd.DataFrame(rows); R = R.dropna(subset=["p"]); R["q_bh"] = C.bh(R.p.values); R.to_csv(C.OUT / "rs_start_e22.csv", index=False)
print(R.round(4).to_string())

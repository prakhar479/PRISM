"""OT L12: per-pair along-U responses by corrupt_step (0 = first move corrupted, ~18 patched positions; 1 = second move,
~11 positions), across E9 / E18 (16k) and E20 (65k, 262k). Mann-Whitney per (run, sae, key); BH over all. Exploratory."""
import numpy as np, pandas as pd
from scipy import stats
import patterns_common as C

out = []
for exp in ["E9", "E18", "E20"]:
    d = C.add_refs(C.load(exp, filt=lambda x: (x.task == "entity_track") & x.condition.isin(["clean_A", "clean_B", "full_along_U", "feat_A_along_U", "feat_A_errB", "feat_A_meanPos"]) & (x.estimator != "random")))
    d = d[d.A_ok & d.B_ok & ~d.condition.isin(["clean_A", "clean_B"])]
    F = C.features("entity_track", d.pair_id.unique(), C.set_name_for(C.run_dirs(exp)[0], "entity_track")).set_index("pair_id")
    d["step"] = d.pair_id.map(F.corrupt_step); d["npatch"] = d.pair_id.map(F.n_patched)
    d["key"] = C.ckey(d)
    for (s, k), g in d.groupby(["sae", "key"]):
        a, b = g[g.step == 0].nldr, g[g.step == 1].nldr
        if len(a) < 5 or len(b) < 5 or g.nldr.std() < 1e-6:
            continue
        u, p = stats.mannwhitneyu(a, b)
        out.append(dict(exp=exp, sae=s, key=k, n0=len(a), n1=len(b), med0=a.median(), med1=b.median(), p=p,
                        npatch0=g[g.step == 0].npatch.median(), npatch1=g[g.step == 1].npatch.median()))
R = pd.DataFrame(out); R["q_bh"] = C.bh(R.p.values); R.to_csv(C.OUT / "ot_step.csv", index=False)
R["est"] = R.key.str.split("|").str[1]; R["cond"] = R.key.str.split("|").str[0]
R["kind"] = np.where(R.est.str.startswith("das"), "das", R.est)
pd.set_option("display.width", 250)
print(f"{len(R)} tests")
print(R.groupby(["exp", "sae", "cond", "kind"]).agg(n=("p", "size"), med0=("med0", "median"), med1=("med1", "median"),
      share_step1_higher=("med1", lambda x: np.nan), sig=("q_bh", lambda q: (q < .05).mean())).drop(columns="share_step1_higher").round(3).to_string())

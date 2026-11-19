"""Layer shape of the feature-only gap (E8 atlas, 16k canonical SAEs): gap(feat_A_errB), gap(feat_A_meanPos) and the
'B-pull' = gap(errB) - gap(meanPos) (= the part of the gap caused by B's own error rather than by losing A's error).
Ratio-of-means NLDR gaps on eligible pairs; split-half replication of the curve shape. Exploratory/post hoc."""
import numpy as np, pandas as pd
import patterns_common as C

d = C.add_refs(C.load("E8", filt=lambda x: x.condition.isin(["clean_A", "clean_B", "feat_A_errB", "feat_A_meanPos"])))
d = d[d.A_ok & d.B_ok].sort_values("run").drop_duplicates(["task", "layer", "pair_id", "condition", "family"])
rng = np.random.default_rng(5)
def gap(g):
    return 1 - (g.D.mean() - g.D_B.mean()) / (g.D_A.mean() - g.D_B.mean())
rows = []
for t, gt in d.groupby("task"):
    pids = np.array(sorted(gt.pair_id.unique())); rng.shuffle(pids); H = set(pids[: len(pids) // 2])
    for (L, c), g in gt[gt.condition.isin(["feat_A_errB", "feat_A_meanPos"])].groupby(["layer", "condition"]):
        bs = [gap(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(1000)]
        rows.append(dict(task=t, layer=L, cond=c, n=len(g), gap=gap(g), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         gap_h1=gap(g[g.pair_id.isin(H)]), gap_h2=gap(g[~g.pair_id.isin(H)])))
R = pd.DataFrame(rows)
W = R.pivot_table(index=["task", "layer"], columns="cond", values=["gap", "lo", "hi", "gap_h1", "gap_h2"])
W.columns = [f"{a}_{b}" for a, b in W.columns]
for s in ["", "_h1", "_h2"]:
    W[f"bpull{s}"] = W[f"gap{s}_feat_A_errB"] - W[f"gap{s}_feat_A_meanPos"]
W.to_csv(C.OUT / "layer_shape_E8.csv")
pd.set_option("display.width", 250)
print(W[["gap_feat_A_errB", "lo_feat_A_errB", "hi_feat_A_errB", "gap_feat_A_meanPos", "bpull", "bpull_h1", "bpull_h2"]].round(3).to_string())
from scipy import stats
for t, g in W.reset_index().groupby("task"):
    print(t, "split-half shape corr errB gap:", round(stats.pearsonr(g.gap_h1_feat_A_errB, g.gap_h2_feat_A_errB)[0], 3),
          "bpull:", round(stats.pearsonr(g.bpull_h1, g.bpull_h2)[0], 3), "argmax bpull h1/h2:", g.layer.values[g.bpull_h1.argmax()], g.layer.values[g.bpull_h2.argmax()])

"""H3 cross-layer stability and H4 cross-task overlap of error directions (E8: 4 tasks x 12 layers; E11: OT, PC L8-16)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
import pandas as pd
out = []
TASKS = ["entity_track", "add_chain", "var_binding", "ioi_control"]
LAYERS = [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
import glob as g
def stem(task, L):
    f = g.glob(f"{R}/E8/*/directions/{task}_L{L}_L{L}_16k_*.npz")
    return f[0].split('/')[-1][:-4] if f else None
D = {}
for t in TASKS:
    for L in LAYERS:
        s = stem(t, L)
        if s: D[(t, L)] = load("E8", s)[0]
# H3: cross-layer (same task)
rows = []
for est in ["pca_pooled", "answer_sep"]:
    for t in TASKS:
        for i, L1 in enumerate(LAYERS):
            for L2 in LAYERS[i+1:]:
                if (t, L1) in D and (t, L2) in D:
                    for r in (1, 2, 4):
                        rows.append(dict(est=est, task=t, L1=L1, L2=L2, r=r, top=top(D[(t, L1)][est][:, :r], D[(t, L2)][est][:, :r])))
cl = pd.DataFrame(rows); cl.to_csv(f"{SP}/deep/geo_crosslayer_E8.csv", index=False)
print("== H3 adjacent-layer top cosine (pca_pooled), r=1 / r=4")
adj = cl[(cl.est == "pca_pooled")]
adj = adj[adj.apply(lambda x: LAYERS.index(x.L2) == LAYERS.index(x.L1) + 1, axis=1)]
print(adj.pivot_table(index=["task", "L1", "L2"], columns="r", values="top").round(2).to_string())
print("== answer_sep adjacent r=1")
a2 = cl[(cl.est == "answer_sep") & (cl.r == 1)]
a2 = a2[a2.apply(lambda x: LAYERS.index(x.L2) == LAYERS.index(x.L1) + 1, axis=1)]
print(a2.pivot_table(index=["L1", "L2"], columns="task", values="top").round(2).to_string())
# H4 cross-task at same layer
rows = []
for est in ["pca_pooled", "answer_sep"]:
    for L in LAYERS:
        for i, t1 in enumerate(TASKS):
            for t2 in TASKS[i+1:]:
                if (t1, L) in D and (t2, L) in D:
                    for r in (1, 2, 4, 8):
                        rows.append(dict(est=est, L=L, t1=t1, t2=t2, r=r, top=top(D[(t1, L)][est][:, :r], D[(t2, L)][est][:, :r]),
                                         cap=capture(D[(t1, L)][est][:, :r], D[(t2, L)][est][:, :r])))
ct = pd.DataFrame(rows); ct.to_csv(f"{SP}/deep/geo_crosstask_E8.csv", index=False)
print("== H4 cross-task top cosine pca_pooled r=1 (Haar p99 0.052) and r=4 (0.097)")
for r in (1, 4):
    print("r=", r); print(ct[(ct.est == "pca_pooled") & (ct.r == r)].pivot_table(index="L", columns=["t1", "t2"], values="top").round(2).to_string())
print("== answer_sep r=1"); print(ct[(ct.est == "answer_sep") & (ct.r == 1)].pivot_table(index="L", columns=["t1", "t2"], values="top").round(2).to_string())

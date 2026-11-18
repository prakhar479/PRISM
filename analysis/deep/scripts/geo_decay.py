"""H3b: decay of error-direction similarity with layer distance (E8, E11 incl. people-cities)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
import pandas as pd, glob as g
cl = pd.read_csv(f"{SP}/deep/geo_crosslayer_E8.csv")
cl["dist"] = cl.L2 - cl.L1
x = cl[(cl.est == "pca_pooled") & (cl.r == 1) & (cl.L1 >= 8) & (cl.L2 <= 16)]
print("== E8 pca r=1, mean top cosine by layer distance (L8-16)")
print(x.pivot_table(index="dist", columns="task", values="top", aggfunc="mean").round(2).to_string())
# E11: OT and PC (pc_en fit) L8-16
D = {}
for t, suf in [("entity_track", ""), ("people_cities", "__fit-pc_en")]:
    for L in range(8, 17):
        f = g.glob(f"{R}/E11/*/directions/{t}_L{L}_L{L}_16k_*{suf}.npz")
        f = [p for p in f if ("__fit" in p) == bool(suf)]
        if f: D[(t, L)] = dict(np.load(f[0]))
rows = []
for t in ["entity_track", "people_cities"]:
    for L1 in range(8, 17):
        for L2 in range(L1 + 1, 17):
            for est in ["pca_pooled", "answer_sep"]:
                rows.append(dict(task=t, L1=L1, L2=L2, dist=L2 - L1, est=est, top1=top(D[(t, L1)][est][:, :1], D[(t, L2)][est][:, :1]),
                                 top4=top(D[(t, L1)][est][:, :4], D[(t, L2)][est][:, :4])))
e11 = pd.DataFrame(rows); e11.to_csv(f"{SP}/deep/geo_crosslayer_E11.csv", index=False)
print("== E11 mean top cosine r=1 by distance"); print(e11.pivot_table(index="dist", columns=["task", "est"], values="top1", aggfunc="mean").round(2).to_string())
print("== E11 OT vs PC same layer (binding tasks)")
for L in range(8, 17):
    a, b = D[("entity_track", L)], D[("people_cities", L)]
    print(L, {est: [round(top(a[est][:, :r], b[est][:, :r]), 3) for r in (1, 2, 4, 8)] for est in ["pca_pooled", "answer_sep"]})
# E11 vs E8 OT same layer: reproducibility of direction across runs (same templates, different sampled pairs?)
print("== reproducibility: E11 OT vs E8 OT, E6 OT, E9 OT at L12 (pca r1 / r4)")
for e in ["E8", "E6", "E9", "E2", "E2r", "E0", "E10", "E4", "E7"]:
    f = g.glob(f"{R}/{e}/*/directions/entity_track_L12_L12_16k_l082.npz")
    for p in f:
        z = np.load(p); print(e, p.split('/')[-3][:6], [round(top(D[("entity_track", 12)]["pca_pooled"][:, :r], z["pca_pooled"][:, :r]), 3) for r in (1, 4)],
                              "answer_sep r1", round(top(D[("entity_track", 12)]["answer_sep"][:, :1], z["answer_sep"][:, :1]), 3))

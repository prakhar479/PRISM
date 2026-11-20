"""DAS 'learned component': project each rotation's columns off the seed-0 init span (shared by every DAS fit), orthonormalize,
keep directions with non-negligible norm. Redo DAS-vs-DAS (across layers / tasks) and DAS-vs-error comparisons on it."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, f"{__import__('geo_lib').SP}/hw")
from geo_lib import *
from harness.model.das import Rotation
import pandas as pd
inits = {k: Rotation(2304, k, 0).R.detach().double().numpy() for k in (1, 2, 4, 8)}
def learned(D):
    k = D.shape[1]; I = inits[k]; R_ = D - I @ (I.T @ D)
    u, s, vt = np.linalg.svd(R_, full_matrices=False)
    return u[:, s > 0.2], s
cells = {"RS18": ("E9", "add_chain_L18_L18_16k_l074"), "OT12": ("E9", "entity_track_L12_L12_16k_l082"),
         "PC12": ("E10", "people_cities_L12_L12_16k_l082__fit-pc_en"), "RS12": ("E16", "add_chain_L12_L12_16k_l082"),
         "RS14": ("E16", "add_chain_L14_L14_16k_l084"), "RS16": ("E16", "add_chain_L16_L16_16k_l078"), "RS20": ("E16", "add_chain_L20_L20_16k_l071")}
Z = {c: load(*v)[0] for c, v in cells.items()}
Lr = {}
for c in Z:
    for k in ("das1", "das2", "das4", "das8"):
        if k in Z[c]:
            Lr[(c, k)], s = learned(Z[c][k].astype(np.float64))
            print(c, k, "learned singular values", np.round(s, 2), "kept", Lr[(c, k)].shape[1])
print("== learned das4 across RS layers (top cos; Haar p99 for 4-dim: .097)")
ls = ["RS12", "RS14", "RS16", "RS18", "RS20"]
for i, a in enumerate(ls):
    print(a, {b: round(top(Lr[(a, "das4")], Lr[(b, "das4")]), 3) for b in ls[i+1:]})
print("== learned das (OT12 vs PC12)", {k: round(top(Lr[("OT12", k)], Lr[("PC12", k)]), 3) for k in ("das1", "das2", "das4")})
print("== learned das vs error subspaces: capture in pca_pooled 32 / answer_sep 32 (Haar 0.014) and top cos with pca r=k")
rows = []
for (c, k), Ld in Lr.items():
    rows.append(dict(cell=c, das=k, dims=Ld.shape[1], cap_pca32=capture(Ld, Z[c]["pca_pooled"]), cap_as32=capture(Ld, Z[c]["answer_sep"]),
                     top_pca_k=top(Ld, Z[c]["pca_pooled"][:, :int(k[3:])]), top_as_k=top(Ld, Z[c]["answer_sep"][:, :int(k[3:])]),
                     top_full_das_vs_pca_k=top(Z[c][k], Z[c]["pca_pooled"][:, :int(k[3:])])))
d = pd.DataFrame(rows); d.to_csv(f"{SP}/deep/geo_das_learned.csv", index=False); print(d.round(3).to_string(index=False))
# RS pca across layers for comparison was 0.13-0.56; also learned das4 vs pca r4 of the other layers

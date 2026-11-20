"""H5: DAS rotations vs error subspaces (how much of the causal DAS subspace lies in the 32-dim error PCA / answer_sep span),
DAS across ranks (nested?), DAS across layers (E16 + E9), people-cities DAS (E10)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
import pandas as pd
nullcap = lambda k, m: m / 2304
cells = [("E9", "add_chain_L18_L18_16k_l074"), ("E9", "entity_track_L12_L12_16k_l082"), ("E10", "people_cities_L12_L12_16k_l082__fit-pc_en"),
         ("E16", "add_chain_L12_L12_16k_l082"), ("E16", "add_chain_L14_L14_16k_l084"), ("E16", "add_chain_L16_L16_16k_l078"), ("E16", "add_chain_L20_L20_16k_l071")]
rows = []
for e, s in cells:
    Z, J = load(e, s)
    for k in [c for c in Z if c.startswith("das")]:
        D = Z[k]; kk = D.shape[1]; acc = J[k]["meta"].get("train_interchange_acc")
        r = dict(exp=e, cell=s.split("_L")[0] + "_L" + s.split("_L")[1], das=k, acc=acc)
        for est in ["pca_pooled", "answer_sep"]:
            for m in (1, 4, 8, 32):
                r[f"cap_{est}_{m}"] = capture(D, Z[est][:, :m])
        both = np.concatenate([Z["pca_pooled"], Z["answer_sep"]], 1)
        r["cap_union64"] = capture(D, both); r["null_union64"] = np.linalg.matrix_rank(both) / 2304
        rows.append(r)
d = pd.DataFrame(rows); d.to_csv(f"{SP}/deep/geo_das_capture.csv", index=False)
pd.set_option("display.width", 250)
print("capture = mean squared projection of DAS columns onto the error subspace; Haar expectation m/2304 (m=32: 0.014, union ~64: 0.028)")
print(d.round(3).to_string(index=False))
# nestedness of DAS ranks
print("== DAS nestedness: capture of das_k in das_2k")
for e, s in cells[:3]:
    Z, _ = load(e, s)
    print(s[:30], {f"{a}->{b}": round(capture(Z[a], Z[b]), 3) for a, b in [("das1", "das2"), ("das2", "das4"), ("das4", "das8")] if a in Z and b in Z})
# DAS across layers (RS): E16 L12,14,16,20 + E9 L18
L = {12: ("E16", "add_chain_L12_L12_16k_l082"), 14: ("E16", "add_chain_L14_L14_16k_l084"), 16: ("E16", "add_chain_L16_L16_16k_l078"),
     18: ("E9", "add_chain_L18_L18_16k_l074"), 20: ("E16", "add_chain_L20_L20_16k_l071")}
Zs = {l: load(*v)[0] for l, v in L.items()}
print("== RS das4 across layers: top cosine / capture  (Haar p99 top r4 0.097)")
ls = sorted(Zs)
for i, a in enumerate(ls):
    print(a, {b: (round(top(Zs[a]["das4"], Zs[b]["das4"]), 2), round(capture(Zs[a]["das4"], Zs[b]["das4"]), 2)) for b in ls[i+1:]})
print("== RS pca_pooled r4 across the same layers")
for i, a in enumerate(ls):
    print(a, {b: round(top(Zs[a]["pca_pooled"][:, :4], Zs[b]["pca_pooled"][:, :4]), 2) for b in ls[i+1:]})
# cross-task DAS: OT das4 vs RS das4 vs PC das4 (different layers: OT/PC L12 vs RS L18; PC vs OT same layer)
ot, pc = load(*cells[1])[0], load(*cells[2])[0]
print("== OT das vs PC das (both L12):", {k: round(top(ot[k], pc[k]), 3) for k in ("das1", "das2", "das4")}, "Haar p99 .052/.072/.097")

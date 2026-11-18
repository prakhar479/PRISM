"""Which 16k atoms align with the DAS learned directions (RS L18, OT L12)? Do they recur across DAS ranks? Do error directions
point at the same atoms? Atom ids for Neuronpedia lookup (gemma-2-2b, <layer>-gemmascope-res-16k)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, f"{__import__('geo_lib').SP}/hw")
from geo_lib import *
from harness.model.das import Rotation
HF = f"{SP}/deep/hf/models--google--gemma-scope-2b-pt-res/snapshots/fd571b47c1c64851e9b1989792367b9babb4af63"
P = {12: f"{HF}/layer_12/width_16k/average_l0_82/params.npz", 18: f"{HF}/layer_18/width_16k/average_l0_74/params.npz"}
inits = {k: Rotation(2304, k, 0).R.detach().double().numpy() for k in (1, 2, 4, 8)}
for L, stem in [(18, "add_chain_L18_L18_16k_l074"), (12, "entity_track_L12_L12_16k_l082")]:
    z = np.load(P[L]); W = z["W_dec"].astype(np.float64); Dn = W / np.linalg.norm(W, axis=1, keepdims=True); thr = z["threshold"]
    Z = load("E9", stem)[0]; print(f"== {stem}")
    tops = {}
    for k in ("das1", "das2", "das4", "das8"):
        D = Z[k].astype(np.float64); I = inits[int(k[3:])]; Rr = D - I @ (I.T @ D)
        u, s, _ = np.linalg.svd(Rr, full_matrices=False); U = u[:, s > .5]
        # energy of each atom in the learned subspace
        e = ((Dn @ U) ** 2).sum(1); idx = np.argsort(-e)[:8]; tops[k] = set(idx[:5].tolist())
        print(k, "top atoms (id: cos^2 in learned span, threshold):", [(int(i), round(float(e[i]), 3), round(float(thr[i]), 2)) for i in idx])
    ks = list(tops)
    print("top-5 atom overlap across ranks:", {f"{a}&{b}": len(tops[a] & tops[b]) for i, a in enumerate(ks) for b in ks[i+1:]})
    for est in ("pca_pooled", "answer_sep"):
        e = ((Dn @ Z[est][:, :4]) ** 2).sum(1); idx = np.argsort(-e)[:5]
        print(est, "r4 top atoms:", [(int(i), round(float(e[i]), 3)) for i in idx], "overlap with das8 top5:", len(set(idx.tolist()) & tops["das8"]))
    # null: energy of best atom in a random k-dim subspace
    rng = np.random.default_rng(0)
    nb = [float((((Dn @ np.linalg.qr(rng.standard_normal((2304, 2)))[0]) ** 2).sum(1)).max()) for _ in range(50)]
    print("null best-atom energy in a random 2-dim subspace: p50 %.3f p99 %.3f" % (np.median(nb), np.percentile(nb, 99)))

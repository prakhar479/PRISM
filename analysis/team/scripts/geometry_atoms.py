"""Decoder-atom geometry with the public 16k Gemma Scope decoders (L12 l082, L18 l074). Post hoc, exploratory.
(a) per-column: each E18 error PC (pca_pooled cols 1..64) -> max |cos| with a decoder atom and OMP-20 energy; summarised by rank bin.
(b) subspace atom capture: share of a k-dim subspace's energy inside the span of n greedily chosen atoms (simultaneous OMP:
    each step picks the atom with the largest residual-energy gain for the subspace). Applied to learned DAS (init removed) k=4,8,16,
    error pca_pooled r=k, and Haar-random k-dim subspaces (null, 20 draws). Compared (descriptively) with the behavioural feature
    capture NLDR(feat_A_along_U)/NLDR(full_along_U) at 16k from geometry_width_capture.py.
Out: team/geometry_out/atoms_cols.csv, atoms_subspace.csv"""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from geometry_lib import *
import pandas as pd, functools
print = functools.partial(print, flush=True)
HF = f"{SP}/deep/hf/models--google--gemma-scope-2b-pt-res/snapshots/fd571b47c1c64851e9b1989792367b9babb4af63"
PAR = {12: f"{HF}/layer_12/width_16k/average_l0_82/params.npz", 18: f"{HF}/layer_18/width_16k/average_l0_74/params.npz"}
cells = {"RS18": (18, "add_chain_L18_L18_16k_l074"), "OT12": (12, "entity_track_L12_L12_16k_l082"),
         "PC12": (12, "people_cities_L12_L12_16k_l082__fit-pc_en")}
def unit_rows(W): return W / np.linalg.norm(W, axis=1, keepdims=True)
def omp_cols(u, A, n=20):
    res, sel = u.copy(), []
    for _ in range(n):
        sel.append(int(np.argmax(np.abs(A @ res)))); B = A[sel].T
        res = u - B @ np.linalg.lstsq(B, u, rcond=None)[0]
    return 1 - float(res @ res)
def somp(Q, A, ns=(5, 20, 50)):
    """Q orthonormal [D,k]; returns ({n: energy share of span(Q) in span of n selected atoms}, selected atom ids).
    Simultaneous OMP with exact gains: atoms are deflated by the selected span (incremental Gram-Schmidt)."""
    k = Q.shape[1]; Rr, sel, out = Q.copy(), [], {}
    An = A.copy(); nn = (An ** 2).sum(1)
    for i in range(max(ns)):
        g = ((An @ Rr) ** 2).sum(1) / np.maximum(nn, 1e-12); g[sel] = -1
        j = int(np.argmax(g)); sel.append(j)
        q = An[j] / np.linalg.norm(An[j])
        Rr = Rr - np.outer(q, q @ Rr)
        c = An @ q; An -= np.outer(c, q); nn = nn - c ** 2
        if i + 1 in ns: out[i + 1] = 1 - float((Rr ** 2).sum()) / k
    return out, sel

rows_c, rows_s = [], []
rng = np.random.default_rng(0)
for c, (L, stem) in cells.items():
    z = np.load(PAR[L]); A = unit_rows(z["W_dec"].astype(np.float64))
    Z, J = load("E18", stem)
    nul = [(float(np.abs(A @ (v / np.linalg.norm(v))).max()), omp_cols(v / np.linalg.norm(v), A)) for v in rng.standard_normal((100, D))]
    nul = np.array(nul); print(c, "null maxcos p50/p99", np.percentile(nul[:, 0], [50, 99]).round(3), "omp20 p50/p99", np.percentile(nul[:, 1], [50, 99]).round(3))
    for est in ("pca_pooled", "answer_sep"):
        for j in range(64):
            u = Z[est][:, j] / np.linalg.norm(Z[est][:, j])
            rows_c.append(dict(cell=c, est=est, col=j + 1, maxcos=float(np.abs(A @ u).max()), omp20=omp_cols(u, A),
                               null_maxcos_p99=np.percentile(nul[:, 0], 99), null_omp20_p99=np.percentile(nul[:, 1], 99)))
    for k in (4, 8, 16):
        subs = {"das_learned": das_learned(Z[f"das{k}"], f"das{k}")[0], "pca_pooled": orth(Z["pca_pooled"][:, :k]),
                "answer_sep": orth(Z["answer_sep"][:, :k])}
        for name, Q in subs.items():
            o, sel = somp(Q, A)
            rows_s.append(dict(cell=c, k=k, subspace=name, das_acc=J[f"das{k}"]["meta"]["train_interchange_acc"] if name == "das_learned" else np.nan,
                               **{f"atomcap_n{n}": v for n, v in o.items()}, top_atoms=" ".join(map(str, sel[:8]))))
        nn = [somp(orth(rng.standard_normal((D, k))), A)[0] for _ in range(20)]
        for q in (50, 99):
            rows_s.append(dict(cell=c, k=k, subspace=f"haar_p{q}", **{f"atomcap_n{n}": float(np.percentile([x[n] for x in nn], q)) for n in (5, 20, 50)}))
dc = pd.DataFrame(rows_c); dc.to_csv(f"{OUT}/atoms_cols.csv", index=False)
ds = pd.DataFrame(rows_s); ds.to_csv(f"{OUT}/atoms_subspace.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
dc["bin"] = pd.cut(dc.col, [0, 2, 8, 16, 32, 64])
print(dc.groupby(["cell", "est", "bin"], observed=True)[["maxcos", "omp20"]].agg(["median", "max"]).round(3).to_string())
from scipy.stats import spearmanr
for (c, e), x in dc.groupby(["cell", "est"]):
    print(c, e, "spearman(col rank, omp20)", spearmanr(x.col, x.omp20))
print(ds.round(3).to_string(index=False))

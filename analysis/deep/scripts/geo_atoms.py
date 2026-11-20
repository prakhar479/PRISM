"""H1 with decoders (Gemma Scope 16k, L12 l082 and L18 l074): are task-relevant error directions near dictionary atoms?
Measures per direction u (unit): max |cos(u, atom)| over W_dec rows; fraction of u's energy captured by the best k atoms chosen
greedily (OMP, k = 1, 5, 20); same for W_enc columns (can the encoder 'see' u?). Null: random unit vectors (n=200)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, f"{__import__('geo_lib').SP}/hw")
from geo_lib import *
import pandas as pd, glob as g
from harness.model.das import Rotation
HF = f"{SP}/deep/hf/models--google--gemma-scope-2b-pt-res/snapshots/fd571b47c1c64851e9b1989792367b9babb4af63"
P = {12: f"{HF}/layer_12/width_16k/average_l0_82/params.npz", 18: f"{HF}/layer_18/width_16k/average_l0_74/params.npz"}
def unit_rows(W): return W / np.linalg.norm(W, axis=1, keepdims=True)
def omp_energy(u, A, ks=(1, 5, 20)):
    """A: [n_atoms, d] unit rows; greedy OMP on u (unit); energy captured after k atoms."""
    res, sel, out = u.copy(), [], {}
    for i in range(max(ks)):
        j = int(np.argmax(np.abs(A @ res))); sel.append(j)
        B = A[sel].T; coef, *_ = np.linalg.lstsq(B, u, rcond=None); res = u - B @ coef
        if i + 1 in ks: out[i + 1] = 1 - float(res @ res)
    return out
def stats(u, Dn, En):
    u = u / np.linalg.norm(u); o = omp_energy(u, Dn)
    return dict(maxcos_dec=float(np.abs(Dn @ u).max()), omp1=o[1], omp5=o[5], omp20=o[20], maxcos_enc=float(np.abs(En @ u).max()))
inits = {k: Rotation(2304, k, 0).R.detach().double().numpy() for k in (1, 2, 4, 8)}
rows = []
for L in (12, 18):
    z = np.load(P[L]); Dn = unit_rows(z["W_dec"].astype(np.float64)); En = unit_rows(z["W_enc"].T.astype(np.float64))
    rng = np.random.default_rng(L)
    nul = pd.DataFrame([stats(rng.standard_normal(2304), Dn, En) for _ in range(200)])
    print(f"L{L} null (random unit): ", nul.quantile([.5, .99]).round(3).to_dict())
    rows += [dict(layer=L, src="null_p50", **nul.quantile(.5).to_dict()), dict(layer=L, src="null_p99", **nul.quantile(.99).to_dict())]
    stems = [("E8", f) for f in g.glob(f"{R}/E8/*/directions/*_L{L}_L{L}_16k_*.npz")]
    stems += [("E10", f) for f in g.glob(f"{R}/E10/*/directions/people_cities_L{L}_*.npz")]
    stems += [("E9", f) for f in g.glob(f"{R}/E9/*/directions/*_L{L}_L{L}_16k_*.npz")]
    for e, f in stems:
        Z = dict(np.load(f)); t = f.split("/")[-1].split("_L")[0]
        for est in ["pca_pooled", "answer_sep"] if e != "E9" else []:
            for c in range(2):
                rows.append(dict(layer=L, src=f"{e}:{t}:{est}[{c}]", **stats(Z[est][:, c], Dn, En)))
        if e == "E9":
            for k in ("das1", "das2", "das4", "das8"):
                D = Z[k].astype(np.float64); I = inits[int(k[3:])]; Rr = D - I @ (I.T @ D)
                u_, s_, _ = np.linalg.svd(Rr, full_matrices=False)
                for c in range(int((s_ > .5).sum())):
                    rows.append(dict(layer=L, src=f"E9:{t}:{k}_learned[{c}]", **stats(u_[:, c], Dn, En)))
d = pd.DataFrame(rows); d.to_csv(f"{SP}/deep/geo_atoms.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(d.round(3).to_string(index=False))

"""Shared helpers for team/geometry analyses (post hoc, exploratory)."""
import glob, json, sys
import numpy as np
SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
sys.path.insert(0, f"{SP}/hw")
R = f"{SP}/ana_runs"; OUT = f"{SP}/team/geometry_out"
D = 2304

def load(exp, stem):
    f = glob.glob(f"{R}/{exp}/*/directions/{stem}.npz")
    if not f: return None, None
    z = np.load(f[0]); j = json.load(open(f[0][:-4] + ".json"))
    return {k: z[k].astype(np.float64) for k in z.files}, j

def orth(U):
    q, _ = np.linalg.qr(U); return q

def pcos(A, B):
    return np.linalg.svd(orth(A).T @ orth(B), compute_uv=False)

def top(A, B): return float(pcos(A, B)[0])

def capture(A, B):
    """share of span(A) inside span(B): mean squared projection of A's orthonormal basis onto span(B). Haar mean dim(B)/D."""
    Qa, Qb = orth(A), orth(B)
    return float(((Qb.T @ Qa) ** 2).sum() / Qa.shape[1])

_NULL = {}
def haar_top_p99(k, m, n=1000, seed=0):
    """p99 of the top principal cosine between a Haar-random k-dim subspace and a fixed m-dim subspace of R^D."""
    key = (min(k, m), max(k, m))
    if key not in _NULL:
        rng = np.random.default_rng(seed); a, b = key; v = []
        for _ in range(n):
            q = orth(rng.standard_normal((D, a)))
            v.append(np.linalg.svd(q[:b], compute_uv=False)[0])
        _NULL[key] = float(np.percentile(v, 99))
    return _NULL[key]

_INIT = {}
def das_learned(U, est, seed=0):
    """Same as scripts/make_final.py das_learned: orthonormal basis of U with the shared Rotation(d,k,seed) init span projected
    out (singular values > 1e-6 max), and the init capture (share of U's energy in the init span; Haar k/D)."""
    from harness.config import parse_das
    from harness.model.das import Rotation
    p = parse_das(est); s = seed if p["seed"] is None else p["seed"]
    if (p["k"], s) not in _INIT:
        _INIT[(p["k"], s)] = Rotation(D, p["k"], s).R.detach().double().numpy()
    I = _INIT[(p["k"], s)]
    cap = float(np.sum((I.T @ U) ** 2) / U.shape[1])
    Rr = U - I @ (I.T @ U)
    q, s_, _ = np.linalg.svd(Rr, full_matrices=False)
    return q[:, s_ > 1e-6 * max(s_.max(), 1e-30)], cap, s_

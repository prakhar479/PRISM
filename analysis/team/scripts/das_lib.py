"""Shared helpers for the `das` role (E17/E18/E20 causal-subspace analyses). Post hoc, exploratory. CPU only.
Reads only ana_runs/*/directions and per_pair.parquet; writes to team/das_out/."""
import glob, json, os, sys
import numpy as np
SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
sys.path.insert(0, f"{SP}/hw")
R = f"{SP}/ana_runs"
OUT = f"{SP}/team/das_out"
os.makedirs(OUT, exist_ok=True)
HAAR = {1: .052, 2: .072, 4: .097, 8: .131, 16: .179, 32: .246, 64: .339}
HF = f"{SP}/deep/hf/models--google--gemma-scope-2b-pt-res/snapshots/fd571b47c1c64851e9b1989792367b9babb4af63"
DEC = {12: f"{HF}/layer_12/width_16k/average_l0_82/params.npz", 18: f"{HF}/layer_18/width_16k/average_l0_74/params.npz"}
RECURRENT_L18 = [15376, 7686, 8352, 11219, 13950, 411, 14691]


def load(exp, stem):
    f = glob.glob(f"{R}/{exp}/*/directions/{stem}.npz")
    if not f:
        return None, None
    z = np.load(f[0]); j = json.load(open(f[0][:-4] + ".json"))
    return {k: z[k].astype(np.float64) for k in z.files}, j


def orth(U):
    q, _ = np.linalg.qr(U); return q


def pcos(A, B):
    return np.linalg.svd(orth(A).T @ orth(B), compute_uv=False)


def top(A, B):
    return float(pcos(A, B)[0])


def capture(A, B):
    """share of span(A)'s energy inside span(B) (mean squared projection of A's orthonormal cols); Haar = dim(B)/d."""
    Qa, Qb = orth(A), orth(B)
    return float(((Qb.T @ Qa) ** 2).sum() / Qa.shape[1])


_INIT = {}


def das_init(k, seed, d=2304):
    if (k, seed) not in _INIT:
        from harness.model.das import Rotation
        _INIT[(k, seed)] = Rotation(d, k, seed).R.detach().double().numpy()
    return _INIT[(k, seed)]


def das_parse(name):
    from harness.config import parse_das
    p = parse_das(name); p["seed"] = 0 if p["seed"] is None else p["seed"]
    return p


def learned(U, name, thr=1e-6):
    """(orthonormal basis of U with its own init span projected out, singular values of the residual, init capture)."""
    p = das_parse(name); I = das_init(p["k"], p["seed"])
    cap = float(np.sum((I.T @ U) ** 2) / U.shape[1])
    Rr = U - I @ (I.T @ U)
    q, s, _ = np.linalg.svd(Rr, full_matrices=False)
    return q[:, s > thr * max(s.max(), 1e-30)], s, cap


def learned_weighted(U, name):
    """learned component without renormalisation: U - P_init U (columns keep their residual norm). Used for energy shares."""
    p = das_parse(name); I = das_init(p["k"], p["seed"])
    return U - I @ (I.T @ U)


_D = {}


def decoder(layer):
    if layer not in _D:
        z = np.load(DEC[layer]); W = z["W_dec"].astype(np.float64)
        _D[layer] = W / np.linalg.norm(W, axis=1, keepdims=True)
    return _D[layer]


def subspace_omp(Q, D, m_max=20, ms=(1, 2, 5, 10, 20)):
    """Greedy atom selection for a subspace. Q [d,k] orthonormal (or weighted columns), D [n,d] unit atoms.
    At each step pick the atom with the largest squared projection of the current residual (summed over Q's columns), then
    refit (least squares on the selected atoms). Returns energy share captured after m atoms (relative to ||Q||_F^2) and atoms."""
    tot = float((Q ** 2).sum()); res = Q.copy(); sel = []; out = {}
    for i in range(m_max):
        sc = ((D @ res) ** 2).sum(1); sc[sel] = -1
        j = int(np.argmax(sc)); sel.append(j)
        B = D[sel].T
        coef, *_ = np.linalg.lstsq(B, Q, rcond=None); res = Q - B @ coef
        if i + 1 in ms:
            out[i + 1] = 1 - float((res ** 2).sum()) / tot
    return out, sel


def atom_share(Q, D, ids):
    """per-atom cos^2 with span(Q) (Q orthonormal): ||Q^T d||^2."""
    return {int(i): float(((Q.T @ D[i]) ** 2).sum()) for i in ids}


def flips(exp, estimators, condition="full_along_U", filt="f_pilot", extra_filter=None):
    """per-pair flip indicator (top1 == ans_A among eligible: clean_B top1 == ans_B), test split, given filter.
    Returns DataFrame pair_id x (cell, estimator)."""
    import pandas as pd
    f = glob.glob(f"{R}/{exp}/*/per_pair.parquet")[0]
    cols = ["pair_id", "split", "task", "layer", "sae", "condition", "estimator", "r", "fit_families", filt, "top1_id", "ans_A_id",
            "ans_B_id"]
    d = pd.read_parquet(f, columns=cols)
    d = d[(d.split == "test") & d[filt]]
    cb = d[d.condition == "clean_B"][["pair_id", "task", "layer", "sae", "fit_families", "top1_id", "ans_B_id"]]
    cb = cb[cb.top1_id == cb.ans_B_id][["pair_id", "task", "layer", "sae", "fit_families"]]
    x = d[(d.condition == condition) & d.estimator.isin(estimators)]
    x = x.merge(cb, on=["pair_id", "task", "layer", "sae", "fit_families"])
    x["flip"] = (x.top1_id == x.ans_A_id).astype(int)
    return x


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out

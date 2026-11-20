"""Shared helpers for the geometry analyses (post hoc, exploratory)."""
import glob, json, os, re
import numpy as np
SP = str(__import__("layout_shim").root())
R = f"{SP}/ana_runs"
HAAR = {1: .052, 2: .072, 4: .097, 8: .131, 16: .179, 32: .246}

def load(exp, stem):
    f = glob.glob(f"{R}/{exp}/*/directions/{stem}.npz")
    if not f: return None
    z = np.load(f[0]); j = json.load(open(f[0][:-4] + ".json"))
    return {k: z[k] for k in z.files}, j

def orth(U):
    q, _ = np.linalg.qr(U); return q

def pcos(A, B):
    """principal cosines between span(A) and span(B)"""
    return np.linalg.svd(orth(A).T @ orth(B), compute_uv=False)

def top(A, B): return float(pcos(A, B)[0])

def capture(A, B):
    """mean squared projection of orthonormal cols of A onto span(B): fraction of span(A) inside span(B)."""
    Qa, Qb = orth(A), orth(B)
    return float(((Qb.T @ Qa) ** 2).sum() / Qa.shape[1])

def haar_capture(k, m, d=2304):  # expected fraction of a random k-subspace in a fixed m-subspace
    return m / d

def haar_null(k, kb=None, n=500, d=2304, seed=0, stat="top"):
    rng = np.random.default_rng(seed); kb = kb or k; out = []
    for _ in range(n):
        A = rng.standard_normal((d, k)); B = rng.standard_normal((d, kb))
        out.append(top(A, B) if stat == "top" else capture(A, B))
    return np.array(out)

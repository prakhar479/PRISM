"""STATS (exploratory, post hoc). Priority 1(b): window-level regression of log splice cost on script (native vs Latin),
error size, and predictability (clean NLL/token, specified flexibly), with set as the cluster.

Data: E12 (16k) + E21 (65k) per-window npz, layers 12 and 18 (the only layers with all 12 sets and both widths):
  4 cells (layer x width) x 12 sets x ~2000 windows. y = log(cost), cost = (nll_splice - nll_clean)/pred; windows with cost <= 0
  (0.3-2.4 %, almost all Latin) are dropped for the log models (this biases Latin cost UP, i.e. against a script effect); a
  log-link quasi-Poisson (PPML) on raw cost keeps them.
Model (pooled over the 4 cells): log cost = cell FE + script*native + cell-specific error slope(s) + cell-specific f(nllc)
  [+ language x cell FE].  Every nuisance term is interacted with cell; the script coefficient is common (pooled) or per cell.
Specs: predictability P in {none, lin, rcs4, rcs6, bins(0.25 nat)}; error E in {none, lf, le, lr, lf+le}
  (lf = log FVU, le = log mean |eps|/|x|, lr = log raw |eps|^2 per position); langFE in {no, yes} (hi/ta/bn/en/fr, Hinglish = hi);
  sample in {all, support} (support = windows in cell x nllc-bin(0.25) x logFVU-bin(0.1) strata containing both scripts).
Inference for the script coefficient (G = 12 set clusters: 6 native, 6 Latin):
  CR1 cluster-robust SE (t with G-1 = 11 df); wild cluster restricted bootstrap p (Rademacher, all 2^12 sign vectors enumerated);
  stratified cluster bootstrap CI over sets (6 native + 6 Latin drawn with replacement, B = 999; draws where the script
  coefficient is not identified are counted and dropped); window bootstrap CI within sets (B = 200, main specs only);
  leave-one-set-out range.
Outputs: team/stats_out/mr_specs.csv (pooled spec grid), mr_cells.csv (per cell), mr_loso.csv, mr_ppml.csv
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".layout/team/scripts")
from stats_common import NATIVE, OUT, e12_windows  # noqa: E402

rng = np.random.default_rng(7)
B_CL, B_WIN = 999, 200
LANG = {"hi_deva_dakshina": "hi", "hi_deva_web": "hi", "hi_latn_dakshina": "hi", "hinglish_phinc": "hi",
        "ta_taml_dakshina": "ta", "ta_web": "ta", "ta_latn_dakshina": "ta", "bn_beng_dakshina": "bn", "bn_web": "bn",
        "bn_latn_dakshina": "bn", "en_pile": "en", "fr_web": "fr"}

w0 = e12_windows()
w0 = w0[w0.layer.isin([12, 18])].reset_index(drop=True)
w0["lf"] = np.log(w0.fvu)
w0["le"] = np.log(w0.eps)
w0["lr"] = np.log(w0.fvu_num / w0.pos)
w0["cell"] = w0.layer.astype(str) + "_" + w0.width
w0["lang"] = w0.set.map(LANG)
# common-support strata
w0["stratum"] = w0.cell + "|" + np.floor(w0.nllc / 0.25).astype(int).astype(str) + "|" + np.floor(w0.lf / 0.1).astype(int).astype(str)
both = w0.groupby("stratum").native.agg(lambda s: s.nunique() == 2)
w0["support"] = w0.stratum.map(both)
print("share of windows on common support by script:", w0.groupby("native").support.mean().round(3).to_dict())
print(w0.groupby(["set"]).support.mean().round(3).to_string())
CELLS = sorted(w0.cell.unique())


def rcs(x, knots):
    k = np.asarray(knots)
    K = len(k)
    out = [x]
    d = (k[-1] - k[0]) ** 2
    p = lambda u: np.maximum(u, 0) ** 3
    for j in range(K - 2):
        out.append((p(x - k[j]) - p(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                    + p(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / d)
    return np.column_stack(out)


def design(w, P, E, langfe, script_by_cell=False):
    cols, names = [], []
    cells = sorted(w.cell.unique())
    for c in cells:
        m = (w.cell == c).to_numpy().astype(float)
        cols.append(m)
        names.append(f"cell[{c}]")
        if E != "none":
            for e in E.split("+"):
                cols.append(m * w[e].to_numpy())
                names.append(f"{e}:{c}")
        x = w.nllc.to_numpy()
        if P == "lin":
            cols.append(m * x)
            names.append(f"nllc:{c}")
        elif P.startswith("rcs"):
            K = int(P[3:])
            kn = np.quantile(x[m > 0], np.linspace(0.05, 0.95, K))
            Bm = rcs(x, kn)
            for j in range(Bm.shape[1]):
                cols.append(m * Bm[:, j])
                names.append(f"rcs{j}:{c}")
        elif P == "bins":
            b = np.floor(x / 0.25).astype(int)
            ub = np.unique(b[m > 0])
            for v in ub[1:]:
                cols.append(m * (b == v))
                names.append(f"bin{v}:{c}")
        if langfe:
            for lg in ["ta", "bn", "en", "fr"]:
                cols.append(m * (w.lang == lg).to_numpy())
                names.append(f"lang[{lg}]:{c}")
    if script_by_cell:
        for c in cells:
            cols.append((w.cell == c).to_numpy() * w.native.to_numpy())
            names.append(f"native:{c}")
    else:
        cols.append(w.native.to_numpy().astype(float))
        names.append("native")
    X = np.column_stack(cols).astype(float)
    keep = X.std(0) > 0
    keep[[i for i, n in enumerate(names) if n.startswith("cell[")]] = True
    return X[:, keep], [n for n, k in zip(names, keep) if k]


def fit_ols(X, y, cl):
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    G = cl.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, cl, X * e[:, None])
    n, k = X.shape
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V)), XtXi


def wild_p(X, y, cl, j):
    """restricted wild cluster bootstrap, all 2^G Rademacher sign vectors, vectorised; returns p and t0."""
    G = cl.max() + 1
    b, se, XtXi = fit_ols(X, y, cl)
    t0 = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fr = Xr @ br
    er = y - fr
    k = X.shape[1]
    A = np.zeros((G, k))
    Bv = np.zeros((G, k))
    np.add.at(A, cl, X * fr[:, None])
    np.add.at(Bv, cl, X * er[:, None])
    r = XtXi[j]
    M_r = np.zeros((G, k))  # M_g r  (X_g'X_g r)
    Xr_ = X @ r
    np.add.at(M_r, cl, X * Xr_[:, None])
    signs = (((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2 - 1).astype(float)
    base = XtXi @ A.sum(0)
    Q = (XtXi @ Bv.T).T                         # G x k
    beta = base[None, :] + signs @ Q            # S x k
    a = A @ r                                   # G
    bb = Bv @ r                                 # G
    u = a[None, :] + signs * bb[None, :] - beta @ M_r.T   # S x G
    n = len(y)
    V = (u ** 2).sum(1) * (G / (G - 1)) * ((n - 1) / (n - k))
    t = beta[:, j] / np.sqrt(V)
    return float(np.mean(np.abs(t) >= abs(t0) - 1e-12)), float(t0)


def cluster_boot(X, y, cl, j, native_cl, B):
    """stratified (by script) cluster bootstrap using per-cluster X'X, X'y."""
    G = cl.max() + 1
    k = X.shape[1]
    M = np.zeros((G, k, k))
    v = np.zeros((G, k))
    for g in range(G):
        Xg = X[cl == g]
        M[g] = Xg.T @ Xg
        v[g] = Xg.T @ y[cl == g]
    nat = np.where(native_cl)[0]
    lat = np.where(~native_cl)[0]
    out, bad = [], 0
    for _ in range(B):
        c = np.bincount(np.concatenate([rng.choice(nat, len(nat)), rng.choice(lat, len(lat))]), minlength=G).astype(float)
        XtX = np.tensordot(c, M, 1)
        Xty = c @ v
        # identifiability of coefficient j: is X_j in the span of the other columns (weighted)?
        # partial variance of column j given the others (Schur complement) relative to its raw second moment
        o = np.array([i for i in range(k) if i != j and XtX[i, i] > 1e-9])
        if XtX[j, j] <= 1e-9:
            bad += 1
            continue
        So = np.linalg.pinv(XtX[np.ix_(o, o)], rcond=1e-12)
        pv = XtX[j, j] - XtX[j, o] @ So @ XtX[o, j]
        if pv / XtX[j, j] < 1e-4:
            bad += 1
            continue
        idx = np.concatenate([o, [j]])
        bsub = np.linalg.pinv(XtX[np.ix_(idx, idx)], rcond=1e-12) @ Xty[idx]
        out.append(bsub[-1])
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5]) if len(out) > 50 else (np.nan, np.nan), bad / B


def window_boot(X, y, cl, j, B):
    G = cl.max() + 1
    groups = [np.where(cl == g)[0] for g in range(G)]
    out = []
    for _ in range(B):
        ix = np.concatenate([gi[rng.integers(0, len(gi), len(gi))] for gi in groups])
        b = np.linalg.lstsq(X[ix], y[ix], rcond=None)[0]
        out.append(b[j])
    return np.percentile(out, [2.5, 97.5])


def prep(sample):
    w = w0[w0.cost > 0]
    if sample == "support":
        w = w[w.support]
    w = w.reset_index(drop=True)
    cl = pd.factorize(w.set)[0]
    native_cl = np.array([s in NATIVE for s in pd.factorize(w.set)[1]])
    return w, cl, native_cl


rows = []
t0 = time.time()
MAIN = {("rcs4", "lf", False, "all"), ("rcs4", "le", False, "all"), ("bins", "lf", False, "all"), ("rcs4", "lf", True, "all"),
        ("bins", "lf", True, "all"), ("rcs4", "lf", False, "support"), ("none", "none", False, "all"), ("none", "lf", False, "all")}
import os  # noqa: E402
for sample in ([] if os.environ.get("SKIP_GRID") else ["all", "support"]):
    w, cl, native_cl = prep(sample)
    y = np.log(w.cost.to_numpy())
    for P, E, lf in itertools.product(["none", "lin", "rcs4", "rcs6", "bins"], ["none", "lf", "le", "lr", "lf+le"], [False, True]):
        X, names = design(w, P, E, lf)
        j = names.index("native")
        b, se, _ = fit_ols(X, y, cl)
        p_w, t = wild_p(X, y, cl, j)
        (clo, chi), bad = cluster_boot(X, y, cl, j, native_cl, B_CL)
        r = dict(sample=sample, P=P, E=E, langFE=lf, n=len(y), k=X.shape[1], beta=b[j], se_CR1=se[j], t=t,
                 ratio=np.exp(b[j]), ratio_lo_CR1=np.exp(b[j] - 2.201 * se[j]), ratio_hi_CR1=np.exp(b[j] + 2.201 * se[j]),
                 p_wild=p_w, ratio_lo_clboot=np.exp(clo), ratio_hi_clboot=np.exp(chi), clboot_unidentified=bad)
        if (P, E, lf, sample) in MAIN:
            wlo, whi = window_boot(X, y, cl, j, B_WIN)
            r.update(ratio_lo_winboot=np.exp(wlo), ratio_hi_winboot=np.exp(whi))
            # leave one set out
            sets = pd.factorize(w.set)[1]
            lo_vals = {}
            for g, s in enumerate(sets):
                m = cl != g
                Xm, ym = X[m], y[m]
                keep = Xm.std(0) > 0
                keep[j] = True
                for i, nm in enumerate(names):
                    if nm.startswith("cell["):
                        keep[i] = True
                bm = np.linalg.lstsq(Xm[:, keep], ym, rcond=None)[0]
                lo_vals[s] = float(np.exp(bm[list(np.where(keep)[0]).index(j)]))
            r.update(loso_min=min(lo_vals.values()), loso_max=max(lo_vals.values()),
                     loso_min_set=min(lo_vals, key=lo_vals.get), loso_max_set=max(lo_vals, key=lo_vals.get),
                     loso_drop_en=lo_vals.get("en_pile"), loso_drop_fr=lo_vals.get("fr_web"),
                     loso_drop_en_fr=None)
            # drop English AND French (the only Latin sets spanning the native predictability range)
            m = ~np.isin(cl, [list(sets).index("en_pile"), list(sets).index("fr_web")])
            Xm = X[m]
            keep = Xm.std(0) > 0
            keep[j] = True
            for i, nm in enumerate(names):
                if nm.startswith("cell[") or nm.startswith("lang"):
                    keep[i] = keep[i] or nm.startswith("cell[")
            Xk = Xm[:, keep]
            if np.linalg.matrix_rank(Xk) == Xk.shape[1]:
                bm = np.linalg.lstsq(Xk, y[m], rcond=None)[0]
                r["loso_drop_en_fr"] = float(np.exp(bm[list(np.where(keep)[0]).index(j)]))
        rows.append(r)
        print(f"{sample} {P} {E} langFE={lf}: ratio {np.exp(b[j]):.3f} CR1 [{r['ratio_lo_CR1']:.3f},{r['ratio_hi_CR1']:.3f}] "
              f"wild p {p_w:.4f} clboot [{r['ratio_lo_clboot']:.3f},{r['ratio_hi_clboot']:.3f}] unident {bad:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
S = pd.DataFrame(rows) if rows else pd.read_csv(f"{OUT}/mr_specs.csv")
if rows:
    S.to_csv(f"{OUT}/mr_specs.csv", index=False)

# ---------------- per-cell script coefficients (separate fits per layer x width; 12 set clusters each)
crow = []
w, cl_all, _ = prep("all")
for c in CELLS:
    wc = w[w.cell == c].reset_index(drop=True)
    cl = pd.factorize(wc.set)[0]
    native_cl = np.array([s in NATIVE for s in pd.factorize(wc.set)[1]])
    y = np.log(wc.cost.to_numpy())
    for P, E, lf in itertools.product(["none", "lin", "rcs4", "bins"], ["none", "lf", "le"], [False, True]):
        X, names = design(wc, P, E, lf)
        j = names.index("native")
        b, se, _ = fit_ols(X, y, cl)
        p_w, t = wild_p(X, y, cl, j)
        (clo, chi), bad = cluster_boot(X, y, cl, j, native_cl, 499)
        crow.append(dict(cell=c, P=P, E=E, langFE=lf, n=len(y), ratio=np.exp(b[j]), ratio_lo_CR1=np.exp(b[j] - 2.201 * se[j]),
                         ratio_hi_CR1=np.exp(b[j] + 2.201 * se[j]), p_wild=p_w, ratio_lo_clboot=np.exp(clo),
                         ratio_hi_clboot=np.exp(chi), clboot_unidentified=bad))
C = pd.DataFrame(crow)
C.to_csv(f"{OUT}/mr_cells.csv", index=False)


# ---------------- PPML (log-link quasi-Poisson) on raw cost incl. cost <= 0
def ppml(X, y, cl, iters=50):
    b = np.linalg.lstsq(X, np.log(np.clip(y, 1e-3, None)), rcond=None)[0]
    for _ in range(iters):
        mu = np.exp(X @ b)
        z = X @ b + (y - mu) / mu
        Wt = mu
        XtWX = X.T @ (X * Wt[:, None])
        bn = np.linalg.solve(XtWX, X.T @ (Wt * z))
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn
            break
        b = bn
    mu = np.exp(X @ b)
    e = y - mu
    G = cl.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, cl, X * e[:, None])
    Hi = np.linalg.inv(X.T @ (X * mu[:, None]))
    V = Hi @ (S.T @ S) @ Hi * G / (G - 1)
    return b, np.sqrt(np.diag(V))


prow = []
wp = w0.reset_index(drop=True)
clp = pd.factorize(wp.set)[0]
for P, E, lf in [("none", "none", False), ("none", "lf", False), ("rcs4", "lf", False), ("bins", "lf", False),
                 ("rcs4", "le", False), ("rcs4", "lf", True), ("bins", "lf", True)]:
    X, names = design(wp, P, E, lf)
    j = names.index("native")
    b, se = ppml(X, wp.cost.to_numpy(), clp)
    prow.append(dict(P=P, E=E, langFE=lf, n=len(wp), ratio=np.exp(b[j]), lo_CR1=np.exp(b[j] - 2.201 * se[j]),
                     hi_CR1=np.exp(b[j] + 2.201 * se[j])))
Pp = pd.DataFrame(prow)
Pp.to_csv(f"{OUT}/mr_ppml.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(S.round(3).to_string())
print(C.round(3).to_string())
print(Pp.round(3).to_string())

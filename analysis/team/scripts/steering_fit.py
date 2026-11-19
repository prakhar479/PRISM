"""Logistic dose-response fits in log2(scale) for E19 (f_pilot, eligible pairs).

Model: flip(s) = top * sigmoid(beta * (log2 s - c)), binomial likelihood on the per-scale counts, fitted on the RISING part only
(scales <= the scale of the maximum observed flip rate; OT and RS curves turn down at high doses because the answer overshoots
to other candidates / non-candidate tokens). top in [0.01, 1], beta in [0.05, 20], c in [-8, 8].
  * s50 = scale where the fitted curve crosses 0.5 (only if top > 0.5).
  * Random envelope: each of the 20 random seeds is fitted separately (same rising window as the pooled random mean); the
    per-seed fitted curves give a fitted p95 at any scale. Window search on a grid s = 2^(-5..4 by 1/8): scales where the
    estimator's fitted flip >= 0.5 and the fitted random p95 <= 0.05. Anything below s = 0.25 is an EXTRAPOLATION.
  * Pair-bootstrap (200 resamples) CI for s50 and for the window bounds of the estimator (random fits held at point estimate).
Output: steering_out/fit_params.csv, fit_window.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from steering_common import OUT

pp = pd.read_parquet(OUT / "pp_E19.parquet")
pp = pp[pp.f_pilot & pp.elig & pp.condition.str.startswith("steer_")]
GRID = 2.0 ** np.arange(-5, 4.0001, 0.125)
rng = np.random.default_rng(0)


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit(x, k, n):
    def nll(t):
        top, b, c = t
        p = np.clip(top * sig(b * (x - c)), 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    best = None
    for c0 in [-4, -2, 0, 2, 4]:
        for b0 in [0.5, 2, 6]:
            r = minimize(nll, [max(k.max() / n.max(), 0.05), b0, c0], method="L-BFGS-B",
                         bounds=[(0.01, 1.0), (0.05, 20), (-8, 8)])
            if best is None or r.fun < best.fun:
                best = r
    return best.x


def curve(t, s):
    top, b, c = t
    return top * sig(b * (np.log2(s) - c))


def s50(t):
    top, b, c = t
    if top <= 0.5:
        return np.nan
    return 2 ** (c + np.log(0.5 / (top - 0.5)) / b)   # top*sig = .5 -> sig = .5/top -> z = log(.5/(top-.5))


def window(pred_est, p95):
    ok = (pred_est >= 0.5) & (p95 <= 0.05)
    return (GRID[ok].min(), GRID[ok].max()) if ok.any() else (np.nan, np.nan)


rows, wins = [], []
for task, T in pp.groupby("task"):
    # random fits per (r, seed)
    rfit = {}
    for r, R in T[T.estimator == "random"].groupby("r"):
        m = R.groupby("s").isA.mean()
        smax = m.idxmax()
        preds = []
        for seed, Q in R.groupby("seed"):
            a = Q[Q.s <= smax].groupby("s").isA.agg(["sum", "count"])
            t = fit(np.log2(a.index.values), a["sum"].values, a["count"].values)
            preds.append(curve(t, GRID))
            rows.append(dict(task=task, estimator="random", r=r, seed=seed, top=t[0], beta=t[1], c=t[2], s50=s50(t),
                             s_peak=smax))
        P = np.vstack(preds)
        rfit[r] = np.percentile(P, 95, axis=0)
    for (est, r), E in T[T.estimator != "random"].groupby(["estimator", "r"]):
        m = E.groupby("s").isA.mean()
        smax = m.idxmax()
        W = E[E.s <= smax].pivot_table(index="pair_id", columns="s", values="isA", aggfunc="first").astype(float)
        x = np.log2(W.columns.values.astype(float))
        X = W.to_numpy()
        t0 = fit(x, X.sum(0), np.full(len(x), len(X)))
        w0 = window(curve(t0, GRID), rfit[r])
        bs, bw = [], []
        for _ in range(200):
            Xi = X[rng.integers(0, len(X), len(X))]
            ti = fit(x, Xi.sum(0), np.full(len(x), len(X)))
            bs.append(s50(ti))
            bw.append(window(curve(ti, GRID), rfit[r]))
        bs = np.array(bs, float)
        rows.append(dict(task=task, estimator=est, r=r, seed=-1, top=t0[0], beta=t0[1], c=t0[2], s50=s50(t0),
                         s50_lo=np.nanpercentile(bs, 2.5) if np.isfinite(bs).any() else np.nan,
                         s50_hi=np.nanpercentile(bs, 97.5) if np.isfinite(bs).any() else np.nan,
                         s50_defined_frac=np.isfinite(bs).mean(), s_peak=smax, flip_peak=m.max()))
        bw = np.array(bw, float)
        wins.append(dict(task=task, estimator=est, r=r, win_lo=w0[0], win_hi=w0[1],
                         win_exists_boot=np.isfinite(bw[:, 0]).mean(),
                         win_lo_ci=(np.nanpercentile(bw[:, 0], 2.5), np.nanpercentile(bw[:, 0], 97.5)) if np.isfinite(bw[:, 0]).any() else None,
                         extrapolated=bool(np.isfinite(w0[1]) and w0[1] < 0.25),
                         rnd_p95_fit_at_0p25=float(rfit[r][np.argmin(np.abs(GRID - 0.25))]),
                         rnd_p95_fit_at_0p125=float(rfit[r][np.argmin(np.abs(GRID - 0.125))]),
                         est_fit_at_0p125=float(curve(t0, 0.125)), est_fit_at_0p0625=float(curve(t0, 0.0625))))
F = pd.DataFrame(rows)
F.to_csv(OUT / "fit_params.csv", index=False)
Wd = pd.DataFrame(wins)
Wd.to_csv(OUT / "fit_window.csv", index=False)
pd.set_option("display.width", 250)
print(F[F.estimator != "random"].round(3).to_string(index=False))
rs = F[F.estimator == "random"].groupby(["task", "r"]).agg(top=("top", "median"), s50_med=("s50", "median"),
                                                          s50_min=("s50", "min"), n_def=("s50", lambda v: np.isfinite(v).sum()))
print(rs.round(3).to_string())
print(Wd.round(3).to_string(index=False))

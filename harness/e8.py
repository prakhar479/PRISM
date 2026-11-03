"""E8: all-cell table with repair specificity and multiple-comparison corrections (handoff 9 E8, PREREGISTRATION.md).

Pure numpy/pandas over ingested per-pair outputs; no model. One (run, cell, filter) at a time, so every number in a
row comes from one run's forward passes. All conditions of a cell share the same bootstrap resamples (paired).

Per (run, cell, filter, metric):
  gaps            gap(M) = M(full_A) - M(c) for the baselines present (feat_A_errB, feat_A_zero, feat_A_meanPos, ...)
  share closed    s(c) = (M(c) - M(feat_A_errB)) / (M(full_A) - M(feat_A_errB)) for repair conditions
  specificity     per estimator e and rank r: s(same_task, e, r) - best control, where best control is, per resample,
                  the max over the available controls: other_task:<t> (same e, r), generic (pca_pooled, r) and the 95th
                  percentile of s over random-direction seeds at rank r. Preregistered margin: 0.30.
  to-half         smallest r (of those run) with point s(same_task, e, r) >= 0.5

Corrections, per family (filter, metric, estimator, r): Holm over confirmatory cells, BH (q = 0.05) over exploratory
cells. Only the canonical row per cell enters a family: the run with the most random seeds, then the most controls.
A share of a gap is undefined when there is no gap: rows whose feat_A_errB gap CI includes 0 are flagged
(base_gap_resolved = False) and left out of the corrections (on the pilot templates this happens for OT L12 under the
flip rate: every eligible pair already flips under feat_A_errB).
Nothing here interprets the numbers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from harness import stats as St
from harness.metrics import EPS

METRICS = ("flip_rate", "prob_recovery", "nldr")       # flip_rate is the preregistered primary metric
PRIMARY = "flip_rate"
CONFIRMATORY = {("entity_track", 12), ("add_chain", 18)}   # PREREGISTRATION.md (16k, pilot L0 variant, pilot families)
SPEC_MARGIN = 0.30
SPEC_RANKS = (1, 2)
MIN_RANDOM_SEEDS = 20                                    # handoff 7.8: >= 20 random U per r
BASELINES = ("feat_A_errB", "feat_A_zero", "feat_A_meanPos", "feat_B_noise", "feat_A_noise")
LEVEL = 0.95


# --------------------------------------------------------------------------- vectorized metrics
class Arr:
    """Per-pair arrays for one condition, aligned to a cell's pair order."""

    def __init__(self, df: pd.DataFrame, ans_A: np.ndarray):
        la, lb, lse = (df[c].to_numpy(float) for c in ("logit_A", "logit_B", "lse"))
        self.D = la - lb
        self.pA = np.exp(la - lse)
        self.hitA = (df["top1_id"].to_numpy() == ans_A).astype(float)


def weights(idx_list: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Resamples (index arrays, any length) -> count matrix W [K, n]: W[k, i] = times pair i is drawn in resample k."""
    return np.vstack([np.bincount(np.asarray(ix), minlength=n) for ix in idx_list]).astype(float)


def metric_samples(P: Arr, A: Arr, B: Arr, elig: np.ndarray, W: np.ndarray) -> Dict[str, np.ndarray]:
    """Metrics under resample weights W [K, n] (or [n]; ones = the point estimate). Same definitions as
    harness.metrics.per_condition; exact for pair and cluster resamples of any length."""
    tot = W.sum(-1)
    m = lambda x: (W @ x) / tot
    out = {"nldr": (m(P.D) - m(B.D)) / (m(A.D) - m(B.D) + EPS),
           "prob_recovery": (m(P.pA) - m(B.pA)) / (m(A.pA) - m(B.pA) + EPS)}
    ne = W @ elig
    with np.errstate(invalid="ignore", divide="ignore"):
        out["flip_rate"] = np.where(ne > 0, (W @ (P.hitA * elig)) / np.maximum(ne, 1e-12), np.nan)
    return out


def _ci(s: np.ndarray) -> Tuple[float, float]:
    s = s[np.isfinite(s)]
    if not len(s):
        return np.nan, np.nan
    a = (1 - LEVEL) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


# --------------------------------------------------------------------------- one cell
def role_of(task: str, layer: int, sae: str, fit_families: str = "pilot") -> str:
    ok = (task, int(layer)) in CONFIRMATORY and str(sae).startswith(f"L{int(layer)}_16k") and fit_families == "pilot"
    return "confirmatory" if ok else "exploratory"


def analyse_cell(g: pd.DataFrame, flt: str, n_boot: int = 10_000, seed: int = 0,
                 metrics: Sequence[str] = METRICS, cluster_col: Optional[str] = None
                 ) -> Tuple[List[dict], List[dict], List[dict]]:
    """g: per-pair rows of one run and one cell. Returns (gap rows, share rows, specificity rows).
    cluster_col (e.g. 'family'): resample whole clusters instead of pairs (cluster bootstrap, handoff 7.8)."""
    sel = g[g[flt]] if flt in g else g
    key_cols = ["condition", "estimator", "source", "r", "seed"]
    frames = {}
    for key, c in sel.groupby(key_cols, sort=False, dropna=False):
        if key[0].startswith("feat_A_resampC"):
            continue                                            # donor-dependent subsets; reported by E2, not here
        frames[tuple(key)] = c.set_index("pair_id")
    need = [k for k in frames if k[0] in ("clean_A", "clean_B", "full_A", "feat_A_errB")]
    if {k[0] for k in need} != {"clean_A", "clean_B", "full_A", "feat_A_errB"}:
        return [], [], []
    pairs = None
    for k, f in frames.items():
        pairs = f.index if pairs is None else pairs.intersection(f.index)
    pairs = pairs.sort_values()
    if len(pairs) < 2:
        return [], [], []
    get = lambda cond: next(f for k, f in frames.items() if k[0] == cond).loc[pairs]
    base_A = get("clean_A")
    ans_A, ans_B = base_A["ans_A_id"].to_numpy(), base_A["ans_B_id"].to_numpy()
    A, B = Arr(base_A, ans_A), Arr(get("clean_B"), ans_A)
    elig = (get("clean_B")["top1_id"].to_numpy() == ans_B).astype(float)
    n = len(pairs)
    clusters = base_A[cluster_col].to_numpy() if cluster_col else None
    idx = weights([np.arange(n)] + list(St.boot_indices(n, n_boot, seed, clusters)), n)           # row 0 = point
    S = {k: metric_samples(Arr(f.loc[pairs], ans_A), A, B, elig, idx) for k, f in frames.items()
         if k[0] not in ("clean_A",)}
    full = next(v for k, v in S.items() if k[0] == "full_A")
    base = next(v for k, v in S.items() if k[0] == "feat_A_errB")
    gaps, shares, spec = [], [], []
    common = {"n_pairs": n, "flip_n_eligible": int(elig.sum()), "ci": f"cluster:{cluster_col}" if cluster_col else "pairs"}
    for met in metrics:
        F, Bm = full[met], base[met]
        for k, v in S.items():
            if k[0] in BASELINES:
                gs = F - v[met]
                lo, hi = _ci(gs[1:])
                gaps.append({"filter": flt, "metric": met, "condition": k[0], "seed": int(k[4]),
                             "value": float(v[met][0]), "gap": float(gs[0]), "gap_lo": lo, "gap_hi": hi,
                             "full_A": float(F[0]), **common})
        denom = F - Bm
        with np.errstate(invalid="ignore", divide="ignore"):
            share = {k: np.where(np.abs(denom) > EPS, (v[met] - Bm) / denom, np.nan) for k, v in S.items()
                     if k[0] == "repair"}
        for k, s in share.items():
            lo, hi = _ci(s[1:])
            shares.append({"filter": flt, "metric": met, "estimator": k[1], "source": k[2], "r": int(k[3]),
                           "seed": int(k[4]), "share_closed": float(s[0]), "share_lo": lo, "share_hi": hi,
                           "gap_featErrB": float(denom[0]), **common})
        g_lo, g_hi = _ci(denom[1:])
        common_m = {"base_gap": float(denom[0]), "base_gap_lo": g_lo, "base_gap_hi": g_hi,
                    "base_gap_resolved": bool(np.isfinite(g_lo) and g_lo > 0)}
        same = {(k[1], int(k[3])): s for k, s in share.items() if k[2] == "same_task"}
        for (est, r), s_same in same.items():
            ctrls, names = [], []
            for k, s in share.items():
                if int(k[3]) != r:
                    continue
                if k[2].startswith("other_task:") and k[1] == est:
                    ctrls.append(s), names.append(k[2])
                elif k[2] == "generic":
                    ctrls.append(s), names.append("generic")
            rnd = [s for k, s in share.items() if k[2] == "random" and int(k[3]) == r]
            if rnd:
                with _quiet():
                    ctrls.append(np.nanpercentile(np.vstack(rnd), 95, axis=0)), names.append("random_p95")
            row = {"filter": flt, "metric": met, "estimator": est, "r": r, "share_same": float(s_same[0]),
                   "share_same_lo": _ci(s_same[1:])[0], "share_same_hi": _ci(s_same[1:])[1],
                   "controls": ",".join(names), "n_random_seeds": len(rnd),
                   "missing_controls": ",".join(c for c, ok in (("other_task", any(x.startswith("other_task") for x in names)),
                                                                 ("generic", "generic" in names),
                                                                 ("random", bool(rnd))) if not ok),
                   "to_half_r": _to_half(same, est), **common, **common_m}
            if ctrls:
                C = np.vstack(ctrls)
                with np.errstate(invalid="ignore"), _quiet():
                    best = np.nanmax(C, axis=0)
                diff = s_same - best
                row.update(best_control=names[int(np.nanargmax(C[:, 0]))] if np.isfinite(C[:, 0]).any() else "",
                           best_control_share=float(best[0]), random_p95=float(ctrls[-1][0]) if rnd else np.nan,
                           diff=float(diff[0]), diff_lo=_ci(diff[1:])[0], diff_hi=_ci(diff[1:])[1],
                           p_margin=_p(diff[1:], SPEC_MARGIN), p_positive=_p(diff[1:], 0.0))
            spec.append(row)
    return gaps, shares, spec


def _p(samples: np.ndarray, null: float) -> float:
    """Bootstrap p-value; NaN (not 1) when no resample gives a finite statistic."""
    return St.boot_pvalue(samples, null) if np.isfinite(samples).any() else np.nan


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def _to_half(same: Dict[Tuple[str, int], np.ndarray], est: str) -> float:
    rs = sorted(r for (e, r) in same if e == est and r > 0 and np.isfinite(same[(e, r)][0]) and same[(e, r)][0] >= 0.5)
    return float(rs[0]) if rs else np.nan


# --------------------------------------------------------------------------- all runs
CELL = ["task", "layer", "sae", "fit_families"]


def analyse_runs(runs: Sequence[Tuple[dict, pd.DataFrame]], filters: Optional[Sequence[str]] = None,
                 n_boot: int = 10_000, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """runs: (manifest, per_pair df) of complete, non-smoke runs. Returns {'gaps', 'shares', 'spec'} frames."""
    out = {"gaps": [], "shares": [], "spec": []}
    for m, pp in runs:
        if "fit_families" not in pp:
            pp = pp.assign(fit_families="pilot")
        flts = filters or m["config"].get("filters", ["f_pilot"])
        for cell, g in pp.groupby(CELL, sort=False):
            prov = {"exp_id": m["exp_id"], "config_name": m["config"].get("name"), "run_hash": m["run_hash"],
                    **dict(zip(CELL, cell)), "role": role_of(*cell)}
            for flt in flts:
                if flt not in g:
                    continue
                gp, sh, sp = analyse_cell(g, flt, n_boot, seed)
                for k, rows in (("gaps", gp), ("shares", sh), ("spec", sp)):
                    out[k] += [{**prov, **r} for r in rows]
    frames = {k: pd.DataFrame(v) for k, v in out.items()}
    if not frames["spec"].empty:
        frames["spec"] = correct(frames["spec"])
    return frames


def correct(spec: pd.DataFrame) -> pd.DataFrame:
    """Mark the canonical row per (cell, filter, metric, estimator, r) and add Holm / BH adjusted p-values."""
    spec = spec.copy()
    spec["canonical"] = False
    rank_cols = CELL + ["filter", "metric", "estimator", "r"]
    spec["_n_ctrl"] = spec["controls"].fillna("").str.count(",") + spec["controls"].fillna("").ne("").astype(int)
    order = spec.sort_values(["n_random_seeds", "_n_ctrl"], ascending=False)
    spec.loc[order.drop_duplicates(rank_cols).index, "canonical"] = True
    spec["underpowered_random"] = spec["n_random_seeds"] < MIN_RANDOM_SEEDS
    spec["p_holm"] = np.nan
    spec["p_bh"] = np.nan
    if "p_margin" not in spec:
        spec["p_margin"] = np.nan
    fam_cols = ["filter", "metric", "estimator", "r"]
    eligible = spec.canonical & spec.p_margin.notna() & spec.get("base_gap_resolved", True)
    for _, fam in spec[eligible].groupby(fam_cols):
        conf = fam[fam.role == "confirmatory"]
        expl = fam[fam.role == "exploratory"]
        if len(conf):
            spec.loc[conf.index, "p_holm"] = St.holm(conf.p_margin.to_numpy())
        if len(expl):
            spec.loc[expl.index, "p_bh"] = St.bh(expl.p_margin.to_numpy())
    return spec.drop(columns="_n_ctrl")

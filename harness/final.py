"""Pass-2 analysis primitives for the final compute pass (E9, E10, E11, E14). Pure pandas/numpy; no model.

`condition_table` gives, for one cell (one run, task, layer, SAE, fit families, EVALUATION family) and one filter, every
condition key's flip rate, NLDR and probability recovery with paired 95% bootstrap CIs, using exactly E8's metric code
(harness/e8.py: flip rate on pairs whose clean B run predicts B; NLDR / prob recovery against clean_A and clean_B).
`random_null` summarises the random-U rows per (condition, r): mean, 95th percentile and max over seeds, and how many seeds.
Grouping includes the evaluation family, so E14's English-fit cells evaluated on different languages stay separate.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from harness import e8 as E
from harness import stats as St

KEY = ["condition", "estimator", "source", "r", "seed"]
GROUP = ["task", "layer", "sae", "fit_families", "family"]


def eval_group(pp: pd.DataFrame) -> pd.DataFrame:
    """'family' column -> evaluation group (pilot families collapse to 'pilot', as in E3's selection)."""
    if "fit_families" not in pp:
        pp = pp.assign(fit_families="pilot")
    fam = pp["family"].fillna("pilot").astype(str)
    return pp.assign(family=np.where(fam.str.startswith("pilot"), "pilot", fam))


def condition_table(g: pd.DataFrame, flt: str, n_boot: int = 2000, seed: int = 0) -> List[dict]:
    sel = g[g[flt]] if flt in g else g
    frames = {tuple(k): f.drop_duplicates("pair_id").set_index("pair_id") for k, f in sel.groupby(KEY, sort=False, dropna=False)}
    base = {k[0]: f for k, f in frames.items() if k[0] in ("clean_A", "clean_B", "full_A")}
    if set(base) != {"clean_A", "clean_B", "full_A"}:
        return []
    pairs = base["clean_A"].index.intersection(base["clean_B"].index).intersection(base["full_A"].index).sort_values()
    n = len(pairs)
    if n < 2:
        return []
    ca = base["clean_A"].loc[pairs]
    ansA, ansB = ca.ans_A_id.to_numpy(), ca.ans_B_id.to_numpy()
    A, B = E.Arr(ca, ansA), E.Arr(base["clean_B"].loc[pairs], ansA)
    elig = (base["clean_B"].loc[pairs].top1_id.to_numpy() == ansB).astype(float)
    W = E.weights([np.arange(n)] + list(St.boot_indices(n, n_boot, seed)), n)
    full = E.metric_samples(E.Arr(base["full_A"].loc[pairs], ansA), A, B, elig, W)
    rows = []
    for k, f in frames.items():
        if k[0] in ("clean_A",):
            continue
        f = f.reindex(pairs)
        if f["top1_id"].isna().any():
            continue                                            # condition missing for some pairs (e.g. no donor)
        mm = E.metric_samples(E.Arr(f, ansA), A, B, elig, W)
        r = dict(zip(KEY, k))
        r.update({"filter": flt, "n_pairs": n, "flip_n_eligible": int(elig.sum())})
        for met in E.METRICS:
            v = mm[met]
            lo, hi = E._ci(v[1:])
            r.update({met: float(v[0]), f"{met}_lo": lo, f"{met}_hi": hi})
            gap = full[met] - v
            glo, ghi = E._ci(gap[1:])
            r.update({f"gap_{met}": float(gap[0]), f"gap_{met}_lo": glo, f"gap_{met}_hi": ghi})
        rows.append(r)
    return rows


def config_role(m: dict, task: str, layer: int, sae: str, fit: str, family: str) -> str:
    """The `role` the experiment's config gives this cell (E9/E10 make only RS L18 confirmatory; bonus and E14 cells are
    exploratory). Matched on task, layer, SAE width, fit families and evaluation family; falls back to E8's rule."""
    from harness import families as F
    width = str(sae).split("_")[1] if "_" in str(sae) else ""
    for c in (m.get("config") or {}).get("cells", []):
        if c["task"] != task or int(c["layer"]) != int(layer) or str((c.get("sae") or {}).get("width")) != width:
            continue
        if ("+".join(c.get("fit_families") or []) or "pilot") != fit:
            continue
        ev = c.get("eval_families")
        groups = F.expand(task, ev) if ev else ["pilot"]
        if family in groups or (not ev and family == "pilot"):
            return c.get("role", "exploratory")
    return E.role_of(task, layer, sae, fit)


def run_table(m: dict, pp: pd.DataFrame, filters: Sequence[str], n_boot: int = 2000) -> pd.DataFrame:
    pp = eval_group(pp)
    out = []
    for cell, g in pp.groupby(GROUP, sort=False):
        prov = {"exp_id": m["exp_id"], "run_hash": m["run_hash"], **dict(zip(GROUP, cell)), "role": config_role(m, *cell)}
        for flt in filters:
            out += [{**prov, **r} for r in condition_table(g, flt, n_boot, m["config"].get("seed", 0))]
    return pd.DataFrame(out)


def random_null(tab: pd.DataFrame, metric: str = "flip_rate") -> pd.DataFrame:
    """Per (cell, filter, condition, r): random-U distribution of `metric` over seeds."""
    r = tab[tab.estimator == "random"]
    if r.empty:
        return pd.DataFrame()
    cols = ["run_hash", *GROUP, "filter", "condition", "r"]
    return (r.groupby(cols)[metric].agg(random_mean="mean", random_p95=lambda s: float(np.nanpercentile(s, 95)),
                                          random_max="max", random_n="count").reset_index())


# --------------------------------------------------------------------------- decision rules fixed on 2026-11-16 (addendum)
# Calibrated on release-2 data and random geometry only (analysis/calibrate_thresholds.txt), before any E9/E14 output existed.
EFFECT_PTS = 0.10                                         # same 10-point bar as RQ1 and Gate B
HAAR_P99 = {1: 0.052, 2: 0.072, 4: 0.097, 8: 0.131}       # top principal cosine of independent random subspaces of R^2304
# Exploratory ranks (E18, 2026-11-18; not part of the frozen addendum): same Monte Carlo (2000 Haar pairs, seed 0, continuing the
# calibration's generator) at k = 16, 32, 64. Used only for rows at those ranks.
HAAR_P99_EXPLORATORY = {16: 0.179, 32: 0.246, 64: 0.339}
ALIGNED = 0.5
DAS_VALID_ACC = 0.5
M3_MARGIN = 0.10


def _aligned_frames(g, flt):
    sel = g[g[flt]] if flt in g else g
    frames = {tuple(k): f.drop_duplicates("pair_id").set_index("pair_id") for k, f in sel.groupby(KEY, sort=False, dropna=False)}
    base = {k[0]: f for k, f in frames.items() if k[0] in ("clean_A", "clean_B", "full_A", "feat_A_errB")}
    return frames, base


def _setup(base, pairs, n_boot, seed):
    ca = base["clean_A"].loc[pairs]
    ansA, ansB = ca.ans_A_id.to_numpy(), ca.ans_B_id.to_numpy()
    A, B = E.Arr(ca, ansA), E.Arr(base["clean_B"].loc[pairs], ansA)
    elig = (base["clean_B"].loc[pairs].top1_id.to_numpy() == ansB).astype(float)
    n = len(pairs)
    W = E.weights([np.arange(n)] + list(St.boot_indices(n, n_boot, seed)), n)
    return ansA, A, B, elig, W


def paired_diffs(g, flt, pairs_of_keys, n_boot=2000, seed=0, metric="flip_rate"):
    """[(key1, key2)] -> rows with metric(key1) - metric(key2) and a paired 95% CI (same resamples)."""
    frames, base = _aligned_frames(g, flt)
    if not {"clean_A", "clean_B"} <= set(base):
        return []
    pairs = base["clean_A"].index.intersection(base["clean_B"].index).sort_values()
    for k1, k2 in pairs_of_keys:
        if k1 in frames and k2 in frames:
            pairs = pairs.intersection(frames[k1].index).intersection(frames[k2].index)
    ansA, A, B, elig, W = _setup(base, pairs, n_boot, seed)
    out = []
    for k1, k2 in pairs_of_keys:
        if k1 not in frames or k2 not in frames:
            continue
        d = E.metric_samples(E.Arr(frames[k1].loc[pairs], ansA), A, B, elig, W)[metric] - \
            E.metric_samples(E.Arr(frames[k2].loc[pairs], ansA), A, B, elig, W)[metric]
        lo, hi = E._ci(d[1:])
        out.append({"key1": k1, "key2": k2, "metric": metric, "filter": flt, "diff": float(d[0]), "diff_lo": lo, "diff_hi": hi,
                    "n_pairs": len(pairs)})
    return out


def share_diff(g_native, g_transfer, flt, key, n_boot=2000, seed=0, metric="flip_rate"):
    """M3: share closed by `key` (repair, estimator, r) with English-fit U minus with native U, on the SAME evaluation pairs:
    share = (M(repair) - M(feat_A_errB)) / (M(full_A) - M(feat_A_errB)). Returns (native, transfer, diff, lo, hi, n)."""
    fn, bn = _aligned_frames(g_native, flt)
    ft, bt = _aligned_frames(g_transfer, flt)
    if key not in fn or key not in ft or set(bn) != {"clean_A", "clean_B", "full_A", "feat_A_errB"}:
        return None
    pairs = bn["clean_A"].index
    for f in (*bn.values(), fn[key], ft[key]):
        pairs = pairs.intersection(f.index)
    pairs = pairs.sort_values()
    if len(pairs) < 2:
        return None
    ansA, A, B, elig, W = _setup(bn, pairs, n_boot, seed)
    m = lambda f: E.metric_samples(E.Arr(f.loc[pairs], ansA), A, B, elig, W)[metric]
    full, base = m(bn["full_A"]), m(bn["feat_A_errB"])
    den = full - base
    with np.errstate(invalid="ignore", divide="ignore"):
        sn = np.where(np.abs(den) > 1e-12, (m(fn[key]) - base) / den, np.nan)
        st = np.where(np.abs(den) > 1e-12, (m(ft[key]) - base) / den, np.nan)
    d = st - sn
    lo, hi = E._ci(d[1:])
    nlo, nhi = E._ci(sn[1:])
    return {"share_native": float(sn[0]), "share_native_lo": nlo, "share_native_hi": nhi, "share_transfer": float(st[0]),
            "diff": float(d[0]), "diff_lo": lo, "diff_hi": hi, "n_pairs": len(pairs), "gap": float(den[0])}


def alignment(cos_top: float, k: int) -> str:
    if cos_top is None or not np.isfinite(cos_top):
        return ""
    if cos_top >= ALIGNED:
        return "aligned"
    p99 = HAAR_P99.get(int(k), HAAR_P99_EXPLORATORY.get(int(k)))
    if p99 is None:
        return ""
    return "unrelated" if cos_top <= p99 else "partial"

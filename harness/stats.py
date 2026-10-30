"""Statistics (pure numpy): paired / cluster bootstrap, bootstrap p-values, Holm and Benjamini-Hochberg,
random-direction distribution summaries."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

N_BOOT = 10_000


def boot_indices(n: int, n_boot: int = N_BOOT, seed: int = 0, clusters: Optional[Sequence] = None) -> list:
    """Resampling index arrays over n pairs. With clusters (one label per pair) whole clusters are resampled with
    replacement and their members concatenated (so resample sizes vary)."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        return list(rng.integers(0, n, size=(n_boot, n)))
    labels = np.asarray(clusters)
    uniq = np.unique(labels)
    members = [np.flatnonzero(labels == u) for u in uniq]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def bootstrap(stat: Callable[[np.ndarray], Dict[str, float]], n: int, n_boot: int = N_BOOT, seed: int = 0,
              clusters: Optional[Sequence] = None, level: float = 0.95) -> Dict[str, dict]:
    """stat(idx) -> {name: value}. Returns {name: {point, lo, hi, se, samples}} (percentile intervals)."""
    point = stat(np.arange(n))
    samples = {k: [] for k in point}
    for idx in boot_indices(n, n_boot, seed, clusters):
        v = stat(idx)
        for k in point:
            samples[k].append(v[k])
    a = (1 - level) / 2
    out = {}
    for k, p in point.items():
        s = np.asarray(samples[k], dtype=float)
        s_ok = s[np.isfinite(s)]
        lo, hi = (np.percentile(s_ok, [100 * a, 100 * (1 - a)]) if len(s_ok) else (np.nan, np.nan))
        out[k] = {"point": float(p), "lo": float(lo), "hi": float(hi), "se": float(s_ok.std(ddof=1)) if len(s_ok) > 1 else np.nan,
                  "samples": s}
    return out


def boot_pvalue(samples: np.ndarray, null: float = 0.0, alternative: str = "greater") -> float:
    """One-sided bootstrap p-value for H1: stat > null ('greater') or stat < null ('less'); +1 smoothing."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    bad = (s <= null).sum() if alternative == "greater" else (s >= null).sum()
    return float((bad + 1) / (len(s) + 1))


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        run = min(run, p[i] * m / (rank + 1))
        adj[i] = min(run, 1.0)
    return adj


def distribution(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan,
            "p95": float(np.percentile(v, 95)) if len(v) else np.nan, "max": float(v.max()) if len(v) else np.nan}

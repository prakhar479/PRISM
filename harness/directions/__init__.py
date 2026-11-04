"""Direction estimators (handoff §7.6). Fit on TRAIN pairs only: every estimator takes a FitSet, and a FitSet can only
be built from train pairs (LeakageError otherwise). Returns orthonormal U [d, r_max] (float64 numpy) + metadata.

Pilot estimators are the pilot's own functions (src/experiments/exp3_hybrid_repair.py), so U matches bit-for-bit:
    pca_pooled  = fit_unsupervised on last-position train errors, A and B rows stacked ("top principal directions")
    answer_sep  = fit_supervised (class-centroid SVD padded with residual SVD; "answer-separating directions")
    random      = random_subspace(d, r, seed)  (QR of a Gaussian, sign-fixed)
    generic     = fit_cov_eig on generic-text errors
New:
    pca_pos_centered            PCA of span-row errors after subtracting the per-key (slot / offset) train mean
    pca_slot_centered_pairdiff  PCA of (eps_A - eps_B) span rows after subtracting the per-key train mean
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class FitSet:
    """The pairs a fitted object may see. split must be 'train'; eval pair ids must be disjoint from pair_ids."""
    pair_ids: tuple
    split: str = "train"
    source: str = "same_task"           # same_task | other_task:<task> | generic:<corpus> | random

    def __post_init__(self):
        if self.split != "train":
            raise LeakageError(f"fitting on split '{self.split}' is not allowed")
        ids = [str(p) for p in self.pair_ids]
        if any("-test-" in p for p in ids):
            raise LeakageError("FitSet contains test pair ids")

    def require_train(self, pair_ids: Sequence[str]) -> None:
        """The data passed to an estimator must belong to this FitSet."""
        extra = set(map(str, pair_ids)) - set(map(str, self.pair_ids))
        if extra:
            raise LeakageError(f"{len(extra)} pairs outside the FitSet, e.g. {sorted(extra)[:3]}")

    def assert_disjoint(self, eval_ids: Sequence[str]) -> None:
        both = set(map(str, eval_ids)) & set(map(str, self.pair_ids))
        if both:
            raise LeakageError(f"{len(both)} eval pairs were used for fitting, e.g. {sorted(both)[:3]}")

    @property
    def digest(self) -> str:
        return hashlib.sha256("\n".join(sorted(map(str, self.pair_ids))).encode()).hexdigest()[:16]

    @classmethod
    def generic(cls, corpus: str) -> "FitSet":
        return cls(pair_ids=(), split="train", source=f"generic:{corpus}")


@dataclass
class Direction:
    U: np.ndarray                         # [d, r_max] orthonormal columns
    estimator: str
    source: str
    fit_digest: str
    n_fit_rows: int
    evr: Optional[np.ndarray] = None      # variance explained per column on the fit data (where defined)
    meta: Dict = field(default_factory=dict)

    def r(self, r: int) -> torch.Tensor:
        if r > self.U.shape[1]:
            raise ValueError(f"r={r} > r_max={self.U.shape[1]}")
        return torch.from_numpy(np.ascontiguousarray(self.U[:, :r])).float()


def _evr(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    tot = float((M ** 2).sum())
    return ((M @ U) ** 2).sum(0) / tot if tot > 0 else np.zeros(U.shape[1])


def _key_center(E: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    keys = np.asarray(keys)
    out = E.copy()
    for k in np.unique(keys):
        m = keys == k
        out[m] -= E[m].mean(0)
    return out


def _top_right_sv(M: np.ndarray, r: int) -> np.ndarray:
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:r].T.copy()


def pca_pooled(fs: FitSet, pair_ids, E_last: torch.Tensor, r_max: int) -> Direction:
    """E_last: [2n, d] last-position train errors (A rows then B rows), as the pilot stacked them."""
    from src.experiments.exp2_geometry import centered_svd
    from src.experiments.exp3_hybrid_repair import fit_unsupervised

    fs.require_train(pair_ids)
    r = min(r_max, len(E_last) - 1, E_last.shape[1])
    U = fit_unsupervised(E_last, r)
    S, _ = centered_svd(E_last)
    return Direction(U, "pca_pooled", fs.source, fs.digest, len(E_last), evr=(S[:r] ** 2) / (S ** 2).sum())


def answer_sep(fs: FitSet, pair_ids, E_last: torch.Tensor, labels: np.ndarray, r_max: int) -> Direction:
    from src.experiments.exp3_hybrid_repair import fit_supervised

    fs.require_train(pair_ids)
    r = min(r_max, len(E_last) - 1, E_last.shape[1])
    U = fit_supervised(E_last, np.asarray(labels), r)
    Ec = E_last.double().numpy() - E_last.double().numpy().mean(0)
    n_cls = len(np.unique(labels))
    return Direction(U, "answer_sep", fs.source, fs.digest, len(E_last), evr=_evr(Ec, U),
                     meta={"n_classes": int(n_cls), "centroid_dims": int(min(n_cls - 1, r))})


def pca_pos_centered(fs: FitSet, pair_ids, E_rows: torch.Tensor, keys: Sequence[str], r_max: int) -> Direction:
    fs.require_train(pair_ids)
    M = _key_center(E_rows.double().numpy(), keys)
    U = _top_right_sv(M, min(r_max, *M.shape))
    return Direction(U, "pca_pos_centered", fs.source, fs.digest, len(M), evr=_evr(M, U))


def pca_slot_centered_pairdiff(fs: FitSet, pair_ids, dE_rows: torch.Tensor, keys: Sequence[str], r_max: int) -> Direction:
    fs.require_train(pair_ids)
    M = _key_center(dE_rows.double().numpy(), keys)
    U = _top_right_sv(M, min(r_max, *M.shape))
    return Direction(U, "pca_slot_centered_pairdiff", fs.source, fs.digest, len(M), evr=_evr(M, U))


def random(d: int, r_max: int, seed: int) -> Direction:
    from src.experiments.exp3_hybrid_repair import random_subspace

    return Direction(random_subspace(d, r_max, seed), "random", "random", f"seed{seed}", 0, meta={"seed": seed})


def generic(fs: FitSet, E_generic: torch.Tensor, r_max: int) -> Direction:
    from src.experiments.exp3_hybrid_repair import fit_cov_eig

    if not fs.source.startswith("generic:"):
        raise LeakageError("generic directions need a FitSet.generic(corpus)")
    U = fit_cov_eig(E_generic, min(r_max, E_generic.shape[1]))
    Ec = E_generic.double().numpy() - E_generic.double().numpy().mean(0)
    return Direction(U, "pca_pooled", fs.source, fs.source, len(E_generic), evr=_evr(Ec, U))


def principal_cosines(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between span(U) and span(V) (descending)."""
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    return np.clip(np.linalg.svd(Qu.T @ Qv, compute_uv=False), 0, 1)

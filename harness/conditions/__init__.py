"""Condition library (handoff §7.5): each condition is the tensor written at the patched positions of the run on B,
built as a pure function of cached tensors. All tensors here are float32 CPU [M, d] "span rows": for pair i the rows
row_ptr[i]:row_ptr[i+1] are its positions first_diff..T-1 (the last row is the final token).

Error-slot ingredients that need fitting (mean errors, covariance noise) are fitted on TRAIN rows only; see
MeanError.fit / CovNoise.fit, which require a train FitSet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from harness.directions import FitSet

EPS = 1e-8


@dataclass
class Spans:
    """Span rows of N pairs at one layer with their SAE decomposition (A = clean prompt, B = corrupt prompt)."""
    pair_ids: List[str]
    row_ptr: np.ndarray                 # [N+1]
    pos: torch.Tensor                   # [M] absolute token position
    off: torch.Tensor                   # [M] offset from the final token (0 = final)
    x_A: torch.Tensor
    x_B: torch.Tensor
    xh_A: Optional[torch.Tensor] = None
    xh_B: Optional[torch.Tensor] = None
    e_A: Optional[torch.Tensor] = None
    e_B: Optional[torch.Tensor] = None
    slot: Optional[List[str]] = None    # [M] semantic slot label per row (None: use offsets)
    meta: Dict[str, np.ndarray] = field(default_factory=dict)   # per pair: T, first_diff, ans_A, ans_B, family

    @property
    def n(self) -> int:
        return len(self.pair_ids)

    @property
    def m(self) -> int:
        return int(self.row_ptr[-1])

    def rows(self, i: int) -> slice:
        return slice(int(self.row_ptr[i]), int(self.row_ptr[i + 1]))

    def row_pair(self) -> torch.Tensor:
        return torch.repeat_interleave(torch.arange(self.n), torch.from_numpy(np.diff(self.row_ptr)).long())

    def keys(self) -> List[str]:
        """Row keys for position-matched statistics: semantic slot if known, else offset from the final token."""
        if self.slot is not None:
            return list(self.slot)
        return [f"off:{int(o)}" for o in self.off]

    def decompose_with(self, sae, **kw) -> "Spans":
        from harness.sae import decompose

        self.xh_A, self.e_A, self.l0_A = (t.cpu() for t in decompose(sae, self.x_A, **kw))
        self.xh_B, self.e_B, self.l0_B = (t.cpu() for t in decompose(sae, self.x_B, **kw))
        return self


# --------------------------------------------------------------------------- fitted error-slot ingredients
@dataclass
class MeanError:
    """Mean train error overall (task) and per row key (slot, else offset from final); fallback key -> task mean."""
    task: torch.Tensor
    by_key: Dict[str, torch.Tensor]
    n_by_key: Dict[str, int]
    fitset: FitSet

    @classmethod
    def fit(cls, fitset: FitSet, sp: Spans, use: Sequence[str] = ("A", "B")) -> "MeanError":
        fitset.require_train(sp.pair_ids)
        E = torch.cat([getattr(sp, f"e_{u}") for u in use])
        keys = sp.keys() * len(use)
        offs = [f"off:{int(o)}" for o in sp.off] * len(use)
        sums: Dict[str, torch.Tensor] = {}
        cnt: Dict[str, int] = {}
        for k_list in (keys, offs):                    # offsets are always available as a fallback
            for j, k in enumerate(k_list):
                sums[k] = sums.get(k, 0) + E[j].double()
                cnt[k] = cnt.get(k, 0) + 1
        return cls(E.double().mean(0).float(), {k: (v / cnt[k]).float() for k, v in sums.items()}, cnt, fitset)

    def per_row(self, sp: Spans) -> torch.Tensor:
        """[M, d] position-matched mean error; records how each row was matched in self.last_coverage."""
        out = torch.empty(sp.m, len(self.task))
        cov = {"slot": 0, "offset": 0, "task": 0}
        for j, (k, o) in enumerate(zip(sp.keys(), sp.off)):
            if k in self.by_key:
                out[j] = self.by_key[k]
                cov["slot" if not k.startswith("off:") else "offset"] += 1
            elif f"off:{int(o)}" in self.by_key:
                out[j] = self.by_key[f"off:{int(o)}"]
                cov["offset"] += 1
            else:
                out[j] = self.task
                cov["task"] += 1
        self.last_coverage = cov
        return out


@dataclass
class CovNoise:
    """Zero-mean Gaussian with the (centered) covariance of train errors."""
    root: torch.Tensor                  # [d, d] with root @ root.T = Sigma
    fitset: FitSet

    @classmethod
    def fit(cls, fitset: FitSet, E_train: torch.Tensor, pair_ids: Sequence[str]) -> "CovNoise":
        fitset.require_train(pair_ids)
        E = E_train.double()
        Ec = E - E.mean(0, keepdim=True)
        w, V = torch.linalg.eigh(Ec.T @ Ec / max(len(E) - 1, 1))
        return cls((V * w.clamp_min(0).sqrt()).float(), fitset)

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        return torch.randn(n, self.root.shape[0], generator=gen) @ self.root.T


def rescale_rows(eta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Rescale each row of eta to the norm of the matching row of ref."""
    return eta * (ref.norm(dim=-1, keepdim=True) / (eta.norm(dim=-1, keepdim=True) + EPS))


def gaussian_noise(ref: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Per-token Gaussian rescaled to ||ref|| (identical to the pilot's rescaled_noise for the same generator)."""
    from src.experiments.common import rescaled_noise

    return rescaled_noise(ref, gen)


# --------------------------------------------------------------------------- resampled error donors
def select_donors(meta: Dict[str, np.ndarray], same_answer: bool, seed: int) -> List[dict]:
    """Donor prompt C for every pair i (C is a prompt of another test pair j != i, never pair i's A or B).

    meta: per pair arrays T, first_diff, ans_A, ans_B, family. Constraints: same family, same T, and C's cached span
    covers i's patched positions (first_diff_j <= first_diff_i). same_answer: answer(C) == ans_A[i]. Otherwise
    answer(C) not in {ans_A[i], ans_B[i]}; if impossible, answer(C) != ans_A[i] ('relaxed').
    Returns per pair {j, prompt ('A'|'B'), status ('ok'|'relaxed'|'none')}."""
    rng = np.random.default_rng(seed)
    n = len(meta["T"])
    fam = meta.get("family", np.zeros(n, dtype=object))
    donors_ans = np.concatenate([meta["ans_A"], meta["ans_B"]])
    donors_pair = np.concatenate([np.arange(n), np.arange(n)])
    donors_prompt = np.array(["A"] * n + ["B"] * n)
    out = []
    for i in range(n):
        ok = ((donors_pair != i) & (fam[donors_pair] == fam[i]) & (meta["T"][donors_pair] == meta["T"][i])
              & (meta["first_diff"][donors_pair] <= meta["first_diff"][i]))
        if same_answer:
            cand, status = np.flatnonzero(ok & (donors_ans == meta["ans_A"][i])), "ok"
        else:
            cand = np.flatnonzero(ok & (donors_ans != meta["ans_A"][i]) & (donors_ans != meta["ans_B"][i]))
            status = "ok"
            if not len(cand):
                cand, status = np.flatnonzero(ok & (donors_ans != meta["ans_A"][i])), "relaxed"
        if not len(cand):
            out.append({"j": -1, "prompt": None, "status": "none"})
            continue
        k = int(rng.choice(cand))
        out.append({"j": int(donors_pair[k]), "prompt": str(donors_prompt[k]), "status": status})
    return out


def donor_errors(sp: Spans, donors: List[dict]) -> torch.Tensor:
    """[M, d] error rows of each pair's donor at the same absolute positions; NaN rows where no donor exists."""
    out = torch.full((sp.m, sp.e_A.shape[1]), float("nan"))
    fd = sp.meta["first_diff"]
    for i, d in enumerate(donors):
        if d["j"] < 0:
            continue
        j = d["j"]
        src = sp.e_A if d["prompt"] == "A" else sp.e_B
        start = int(sp.row_ptr[j]) + int(fd[i] - fd[j])
        r = sp.rows(i)
        out[r] = src[start:start + (r.stop - r.start)]
    return out


# --------------------------------------------------------------------------- projections
def proj(U: Optional[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
    """P_U v = U U^T v for rows v; U [d, r] orthonormal (r = 0 or None -> 0)."""
    if U is None or U.shape[1] == 0:
        return torch.zeros_like(v)
    U = U.to(v.dtype)
    return (v @ U) @ U.T


# --------------------------------------------------------------------------- the library
def _need(x, what):
    if x is None:
        raise ValueError(f"condition needs {what}")
    return x


def build(name: str, sp: Spans, U: Optional[torch.Tensor] = None, eps_mean_pos: Optional[torch.Tensor] = None,
          eps_mean_task: Optional[torch.Tensor] = None, eps_C: Optional[torch.Tensor] = None,
          eta: Optional[torch.Tensor] = None, codes_A=None, sae=None, alpha: Optional[float] = None,
          slot_error: str = "B", latents: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Patch rows [M, d] for condition `name` (None = no patch, i.e. clean_B)."""
    xh_A, xh_B, e_A, e_B = sp.xh_A, sp.xh_B, sp.e_A, sp.e_B
    if name == "clean_B":
        return None
    if name == "full_A":
        return sp.x_A.clone()
    if name == "full_B":
        return sp.x_B.clone()
    if name == "feat_A_errB":
        return xh_A + e_B
    if name == "feat_B_errA":
        return xh_B + e_A
    if name == "feat_B_noise":
        return xh_B + _need(eta, "eta")
    if name == "feat_A_zero":
        return xh_A.clone()
    if name == "feat_A_meanTask":
        return xh_A + _need(eps_mean_task, "eps_mean_task")[None, :]
    if name == "feat_A_meanPos":
        return xh_A + _need(eps_mean_pos, "eps_mean_pos")
    if name in ("feat_A_resampC_sameAns", "feat_A_resampC_diffAns"):
        return xh_A + _need(eps_C, "eps_C")
    if name in ("feat_A_noise", "feat_A_covnoise"):
        return xh_A + _need(eta, "eta")
    if name == "repair":
        return xh_A + e_B + proj(U, e_A - e_B)               # x̂_A + P ε_A + (I-P) ε_B
    if name == "repair_mean":
        m = _need(eps_mean_pos, "eps_mean_pos")
        return xh_A + m + proj(U, e_A - m)                  # x̂_A + P ε_A + (I-P) ε̄_pos
    if name == "feat_A_along_U":
        return xh_B + proj(U, xh_A - xh_B) + e_B
    if name == "full_along_U":
        return sp.x_B + proj(U, sp.x_A - sp.x_B)
    if name == "subthresh":
        return xh_A + subthreshold_addback(_need(codes_A, "codes_A"), _need(sae, "sae"), _need(alpha, "alpha"), latents) \
            + (e_B if slot_error == "B" else _need(eps_mean_pos, "eps_mean_pos"))
    raise KeyError(f"unknown condition {name}")


def subthreshold_addback(codes, sae, alpha: float, latents: Optional[torch.Tensor] = None) -> torch.Tensor:
    """sum over entries with alpha*theta <= z < theta, z > 0 (optionally only `latents`) of z * W_dec[i]."""
    from harness.sae import decode_entries

    if alpha < codes.alpha_min - 1e-12:
        raise ValueError(f"codes were kept down to alpha={codes.alpha_min}, asked for {alpha}")
    m = (codes.z >= alpha * codes.threshold) & (codes.z < codes.threshold) & (codes.z > 0)
    if latents is not None:
        m &= torch.isin(codes.col, torch.as_tensor(latents))
    return decode_entries(sae, codes, m)


@dataclass
class GreedyLatents:
    latents: List[int]                 # in selection order
    objective: List[float]             # sum of squared residuals after 0, 1, ..., k picks
    alpha: float
    fitset: FitSet
    n_candidates: int

    def first(self, k: int) -> torch.Tensor:
        return torch.tensor(self.latents[:k], dtype=torch.long)


def greedy_subthreshold(fitset: FitSet, pair_ids: Sequence[str], codes, dec, target: torch.Tensor, alpha: float,
                        k_max: int = 10, chunk: int = 65536) -> GreedyLatents:
    """Greedy forward selection of <= k_max latents whose sub-threshold add-back (entries alpha*theta <= z < theta,
    z > 0 of A) best explains `target` on TRAIN span rows, in least squares:

        minimise  sum_rows || target - sum_{i in L} s_i ||^2,   s_i = z_i W_dec[i] on the rows where i qualifies.

    With target = eps_A - eps_B this picks the latents that move feat_A_errB (x_hat_A + eps_B) closest to x_A.
    A latent occurs at most once per row, so the gain of adding i is 2 <R, s_i> - ||s_i||^2 with ||s_i||^2 =
    sum z^2 ||W_i||^2. Model-free; uses train pairs only (fitset.require_train)."""
    fitset.require_train(pair_ids)
    m = (codes.z >= alpha * codes.threshold) & (codes.z < codes.threshold) & (codes.z > 0)
    r, c, z = codes.row[m], codes.col[m], codes.z[m]
    R = target.float().clone()
    if not len(c):
        return GreedyLatents([], [float((R ** 2).sum())], alpha, fitset, 0)
    cand, inv = torch.unique(c, return_inverse=True)
    W = dec.rows(cand)                                         # [n_cand, d]
    wn2 = (W ** 2).sum(1)
    self_term = torch.zeros(len(cand)).index_add_(0, inv, (z ** 2) * wn2[inv])
    chosen: List[int] = []
    obj = [float((R ** 2).sum())]
    alive = torch.ones(len(cand), dtype=torch.bool)
    for _ in range(min(k_max, len(cand))):
        dots = torch.zeros(len(cand))
        for a in range(0, len(r), chunk):
            rr, ii, zz = r[a:a + chunk], inv[a:a + chunk], z[a:a + chunk]
            dots.index_add_(0, ii, zz * (R[rr] * W[ii]).sum(1))
        gain = 2 * dots - self_term
        gain[~alive] = -float("inf")
        j = int(torch.argmax(gain))
        if gain[j] <= 0:
            break
        sel = inv == j
        R.index_add_(0, r[sel], -(z[sel][:, None] * W[j][None, :]))
        alive[j] = False
        chosen.append(int(cand[j]))
        obj.append(float((R ** 2).sum()))
    return GreedyLatents(chosen, obj, alpha, fitset, int(len(cand)))


CONDITIONS = ["clean_B", "full_A", "full_B", "feat_A_errB", "feat_B_errA", "feat_B_noise", "feat_A_zero",
              "feat_A_meanTask", "feat_A_meanPos", "feat_A_resampC_sameAns", "feat_A_resampC_diffAns", "feat_A_noise",
              "feat_A_covnoise", "repair", "repair_mean", "feat_A_along_U", "full_along_U", "subthresh"]
PILOT_NAMES = {"feat_A_errB": "sae_only", "feat_B_errA": "error_only", "feat_B_noise": "noise_control",
               "feat_A_zero": "sae_splice", "full_A": "clean_full"}

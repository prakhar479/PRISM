"""Final-position metrics (handoff §7.7), pure numpy. Inputs are the stored raw quantities per (pair, condition):

    logit_A, logit_B   post-softcap logits of A's / B's answer token
    lse                logsumexp over the full vocabulary (post-softcap)
    cand_logits        post-softcap logits of every candidate answer (NaN where not a single token)
    top1_id            argmax token id;  ans_A_id, ans_B_id  answer token ids
The pilot's D used post-softcap logits, so all logit metrics here do too.

Per-pair arrays are aligned by pair; `base_A`, `base_B` are the clean A run and the clean B run (no patch).
Normalised metrics are reported as a ratio of means (pilot convention for NLDR) and as a mean of per-pair ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

EPS = 1e-8


@dataclass
class Out:
    """Final-position quantities for N pairs under one condition."""
    logit_A: np.ndarray
    logit_B: np.ndarray
    lse: np.ndarray
    top1_id: np.ndarray
    cand_logits: Optional[np.ndarray] = None     # [N, K]

    def take(self, idx) -> "Out":
        return Out(self.logit_A[idx], self.logit_B[idx], self.lse[idx], self.top1_id[idx],
                   None if self.cand_logits is None else self.cand_logits[idx])

    @property
    def D(self) -> np.ndarray:
        return self.logit_A - self.logit_B

    @property
    def p_A(self) -> np.ndarray:
        return np.exp(self.logit_A - self.lse)

    @property
    def p_B(self) -> np.ndarray:
        return np.exp(self.logit_B - self.lse)

    def p_candidates(self) -> np.ndarray:
        c = np.where(np.isfinite(self.cand_logits), self.cand_logits, -np.inf)
        m = c.max(1, keepdims=True)
        return np.exp((m[:, 0] + np.log(np.exp(c - m).sum(1))) - self.lse)

    def cand_entropy(self) -> np.ndarray:
        """Entropy (nats) of the distribution renormalised over the candidate answers."""
        c = np.where(np.isfinite(self.cand_logits), self.cand_logits, -np.inf)
        z = c - c.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)


def outcome(o: Out, ans_A_id: np.ndarray, ans_B_id: np.ndarray) -> np.ndarray:
    """Three-way top-1 outcome: 'A', 'B' or 'other'."""
    return np.where(o.top1_id == ans_A_id, "A", np.where(o.top1_id == ans_B_id, "B", "other"))


def _ratio(num_p, num_b, den_a, den_b):
    rom = (np.mean(num_p) - np.mean(num_b)) / (np.mean(den_a) - np.mean(den_b) + EPS)
    pair = np.mean((num_p - num_b) / (den_a - den_b + EPS))
    return float(rom), float(pair)


def per_condition(p: Out, base_A: Out, base_B: Out, ans_A_id: np.ndarray, ans_B_id: np.ndarray) -> Dict[str, float]:
    """All scalar metrics for one condition over the given pairs (call with resampled views for bootstrap)."""
    out = {}
    out["nldr"], out["nldr_pair_mean"] = _ratio(p.D, base_B.D, base_A.D, base_B.D)
    out["prob_recovery"], out["prob_recovery_pair_mean"] = _ratio(p.p_A, base_B.p_A, base_A.p_A, base_B.p_A)
    out["D_mean"] = float(p.D.mean())
    out["logprob_diff_mean"] = float(((p.logit_A - p.lse) - (p.logit_B - p.lse)).mean())   # = D (shared lse)
    out["p_A"], out["p_B"] = float(p.p_A.mean()), float(p.p_B.mean())
    if p.cand_logits is not None:
        pc = p.p_candidates()
        out["p_other_candidates"] = float((pc - p.p_A - p.p_B).mean())
        out["p_rest"] = float((1 - pc).mean())
        out["cand_entropy"] = float(p.cand_entropy().mean())
    oc = outcome(p, ans_A_id, ans_B_id)
    out["top1_is_A"], out["top1_is_B"], out["top1_other"] = (float((oc == k).mean()) for k in ("A", "B", "other"))
    eligible = base_B.top1_id == ans_B_id                      # clean B predicts B's answer
    out["flip_n_eligible"] = int(eligible.sum())
    out["flip_rate"] = float((p.top1_id[eligible] == ans_A_id[eligible]).mean()) if eligible.any() else np.nan
    return out


GAP_METRICS = ("nldr", "prob_recovery", "flip_rate")


def gaps(m_cond: Dict[str, float], m_full_A: Dict[str, float], metrics=GAP_METRICS) -> Dict[str, float]:
    """gap(M) = M(full_A) - M(cond). For NLDR and prob recovery M(full_A) = 1 by construction (up to bf16), so this
    equals the handoff's 1 - M(cond); for flip rate full_A need not be 1 and the measured value is used."""
    return {f"gap_{k}": m_full_A[k] - m_cond[k] for k in metrics if k in m_cond and k in m_full_A}


def share_closed(m_repair: Dict[str, float], m_base: Dict[str, float], m_full_A: Dict[str, float],
                 metrics=GAP_METRICS) -> Dict[str, float]:
    """(M(repair) - M(base)) / (M(full_A) - M(base)): share of the base condition's gap that the repair closes."""
    out = {}
    for k in metrics:
        if k in m_repair and k in m_base and k in m_full_A:
            g = m_full_A[k] - m_base[k]
            out[f"share_closed_{k}"] = (m_repair[k] - m_base[k]) / g if abs(g) > EPS else np.nan
    return out

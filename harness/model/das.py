"""Distributed alignment search (minimal; full E9 use is Phase 3). A rotation R [d, k] with orthonormal columns
(torch orthogonal parametrization) defines the interchange x_B + R R^T (x_A - x_B) at the patched positions. Training
minimises cross-entropy toward A's answer at the final position on TRAIN pairs; the model is frozen, gradients flow
only into R."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models import runtime


class Rotation(nn.Module):
    def __init__(self, d: int, k: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        lin = nn.Linear(k, d, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.linalg.qr(torch.randn(d, k, generator=g))[0])
        self.lin = nn.utils.parametrizations.orthogonal(lin)

    @property
    def R(self) -> torch.Tensor:
        return self.lin.weight                   # [d, k], orthonormal columns

    def intervene(self, x_B: torch.Tensor, x_A: torch.Tensor) -> torch.Tensor:
        R = self.R.to(x_B.dtype)
        return x_B + ((x_A - x_B) @ R) @ R.T


def train_das(model, layer: int, rows: Sequence[dict], train_idx: Sequence[int], spans, k: int, steps: int = 200,
              lr: float = 1e-2, seed: int = 0, device="cpu") -> Rotation:
    """spans: Spans for train_idx (same order). One pair per step (batch 1), cycling; returns the trained rotation."""
    rot = Rotation(spans.x_A.shape[1], k, seed).to(device)
    opt = torch.optim.Adam(rot.parameters(), lr=lr)
    for s in range(steps):
        q = s % len(train_idx)
        r = rows[train_idx[q]]
        tB = torch.tensor([r["corrupt_ids"]], device=device)
        with torch.no_grad():
            R = runtime.resid_at(model, tB, layer)
        sl = spans.rows(q)
        x = R.clone()
        x[0, r["first_diff"]:r["T"]] = rot.intervene(spans.x_B[sl].to(device), spans.x_A[sl].to(device)).to(x.dtype)
        lg = runtime.last_logits_from(model, x, layer, tB)
        loss = nn.functional.cross_entropy(lg.float(), torch.tensor([r["ans_A"]], device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rot

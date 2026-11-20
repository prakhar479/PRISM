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


def fit_das(model, layer: int, rows: Sequence[dict], train_idx: Sequence[int], spans, k: int, epochs: int = 3,
            lr: float = 1e-2, seed: int = 0, device="cpu"):
    """E9: train a rank-k DAS rotation on the TRAIN pairs (same interchange as `full_along_U` with U = R) and return
    (U [d, k] float32 numpy, info). Runs outside inference mode with gradients only into R; the model stays frozen.
    info: training loss of the first and last epoch, and the interchange accuracy (top-1 == A's answer) on the train pairs
    after training, so a failed fit is visible in the run's directions/ record."""
    import numpy as np
    with torch.inference_mode(False), torch.enable_grad():
        sp = type("S", (), {})()
        sp.x_A, sp.x_B = spans.x_A.clone().float(), spans.x_B.clone().float()
        sp.rows = spans.rows
        steps = epochs * len(train_idx)
        torch.manual_seed(seed)
        rot = train_das_logged(model, layer, rows, train_idx, sp, k, steps, lr, seed, device)
        acc = 0
        with torch.no_grad():
            for q, i in enumerate(train_idx):
                r = rows[i]
                tB = torch.tensor([r["corrupt_ids"]], device=device)
                R0 = runtime.resid_at(model, tB, layer)
                sl = sp.rows(q)
                x = R0.clone()
                x[0, r["first_diff"]:r["T"]] = rot.intervene(sp.x_B[sl].to(device), sp.x_A[sl].to(device)).to(x.dtype)
                acc += int(runtime.last_logits_from(model, x, layer, tB).argmax(-1).item() == r["ans_A"])
        U = rot.R.detach().float().cpu().numpy()
    info = {"k": k, "epochs": epochs, "lr": lr, "seed": seed, "n_train": len(train_idx),
            "loss_first_epoch": rot.losses[0], "loss_last_epoch": rot.losses[-1], "train_interchange_acc": acc / max(1, len(train_idx))}
    return U.astype(np.float32), info


def train_das_logged(model, layer, rows, train_idx, spans, k, steps, lr, seed, device) -> Rotation:
    """train_das with the mean loss per epoch recorded on the returned rotation (`.losses`)."""
    rot = Rotation(spans.x_A.shape[1], k, seed).to(device)
    opt = torch.optim.Adam(rot.parameters(), lr=lr)
    n = len(train_idx)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    losses, run = [], 0.0
    for s in range(steps):
        q = order[s % n]
        r = rows[train_idx[q]]
        tB = torch.tensor([r["corrupt_ids"]], device=device)
        with torch.no_grad():
            R0 = runtime.resid_at(model, tB, layer)
        sl = spans.rows(q)
        x = R0.clone()
        x[0, r["first_diff"]:r["T"]] = rot.intervene(spans.x_B[sl].to(device), spans.x_A[sl].to(device)).to(x.dtype)
        lg = runtime.last_logits_from(model, x, layer, tB)
        loss = nn.functional.cross_entropy(lg.float(), torch.tensor([r["ans_A"]], device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += float(loss.detach())
        if (s + 1) % n == 0:
            losses.append(run / n)
            run = 0.0
    rot.losses = losses or [run / max(1, steps)]
    return rot

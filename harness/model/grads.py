"""P7 (breadth pilot): directions chosen by the model's output sensitivity instead of the error's variance.

For each train pair the corrupt run on B is patched with feat_A_errB rows (x_hat_A + eps_B, the point every repair starts from)
and g = d(logit_A - logit_B) / d(patched rows) is taken by one backward pass from the patched layer. With
delta = eps_A - eps_B on the same rows:
    grad_pooled  top right singular vectors of the stacked g rows (uncentred): where the output is most sensitive;
    grad_err     top eigenvectors of S = mean over pairs of sum_rows sym(g delta^T). For orthonormal U, trace(U^T S U) is the
                 first-order change of the logit difference from patching eps_A for eps_B along U, so these directions are
                 first-order optimal for the repair conditions (largest eigenvalues first).
Fitted on train pairs only; the model stays frozen."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from src.models import runtime


def fit_grad_dirs(model, layer: int, rows: Sequence[dict], train_idx: Sequence[int], spans, r_max: int,
                  device="cpu") -> Tuple[Dict[str, np.ndarray], dict]:
    d = spans.x_A.shape[1]
    S = np.zeros((d, d))
    G = []
    first_order = []
    with torch.inference_mode(False), torch.enable_grad():
        for q, i in enumerate(train_idx):
            r = rows[i]
            tB = torch.tensor([r["corrupt_ids"]], device=device)
            with torch.no_grad():
                R0 = runtime.resid_at(model, tB, layer)
            sl = spans.rows(q)
            base = (spans.xh_A[sl] + spans.e_B[sl]).to(device)
            v = base.clone().float().requires_grad_(True)
            x = R0.clone()
            x[0, r["first_diff"]:r["T"]] = v.to(x.dtype)
            lg = runtime.last_logits_from(model, x, layer, tB)[0]
            (lg[int(r["ans_A"])] - lg[int(r["ans_B"])]).float().backward()
            g = v.grad.detach().double().cpu().numpy()                     # [span, d]
            dl = (spans.e_A[sl] - spans.e_B[sl]).double().numpy()
            M = g.T @ dl
            S += 0.5 * (M + M.T)
            G.append(g)
            first_order.append(float((g * dl).sum()))
    n = max(1, len(train_idx))
    S /= n
    ev, V = np.linalg.eigh(S)
    order = np.argsort(ev)[::-1]
    r = min(r_max, d)
    U_err = V[:, order[:r]]
    Gs = np.concatenate(G) if G else np.zeros((1, d))
    _, sv, Vt = np.linalg.svd(Gs, full_matrices=False)
    U_pool = Vt[:r].T
    info = {"n_pairs": len(train_idx), "n_rows": int(len(Gs)), "eig_top": [float(x) for x in ev[order[:8]]],
            "eig_bottom": [float(x) for x in ev[order[-4:]]], "grad_sv_top": [float(x) for x in sv[:8]],
            "first_order_full_mean": float(np.mean(first_order)) if first_order else None,
            "note": "first_order_full_mean = mean over pairs of g . (eps_A - eps_B): the linear prediction of feat_A_errA - feat_A_errB"}
    return {"grad_pooled": U_pool.astype(np.float32), "grad_err": U_err.astype(np.float32)}, info

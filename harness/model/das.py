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


def init_seed_v2(seed: int, task: str, layer: int, fit_digest: str) -> int:
    """v2's initial-rotation seed: a stable function of (seed, task, layer, train set), so two cells never share a start
    (v1 fits with the same (k, seed) all started from one rotation; PAPER_FINDINGS P4)."""
    import hashlib
    h = hashlib.sha256(f"{seed}|{task}|{layer}|{fit_digest}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


def split_holdout(n: int, frac: float, seed: int):
    """(fit positions, validation positions) over range(n): a seeded permutation; >= 2 validation pairs when n >= 10."""
    import numpy as np
    order = np.random.default_rng(seed).permutation(n).tolist()
    n_val = int(round(frac * n)) if n >= 10 else 0
    n_val = max(2, n_val) if n >= 10 else 0
    return sorted(order[n_val:]), sorted(order[:n_val])


def _cand(row) -> list:
    """The pair's candidate answer ids (ans_A and ans_B always included), -1 entries dropped."""
    c = [int(x) for x in row.get("cand_ids", []) if int(x) >= 0]
    for a in (row["ans_A"], row["ans_B"]):
        if int(a) not in c:
            c.append(int(a))
    return c


def _bucket_logp(logits: torch.Tensor, cand: list) -> torch.Tensor:
    """log-probabilities over (candidates..., rest) from full-vocabulary logits [V] (float32)."""
    lp = torch.log_softmax(logits.float(), -1)
    c = lp[torch.tensor(cand, device=lp.device)]
    rest = torch.log1p(-torch.exp(torch.logsumexp(c, 0)).clamp(max=1 - 1e-6))
    return torch.cat([c, rest[None]])


def fit_das_v2(model, layer: int, rows: Sequence[dict], train_idx: Sequence[int], spans, k: int, objective: str = "kl",
               epochs: int = 10, lr: float = 1e-2, seed: int = 0, init_seed: int = 0, holdout: float = 0.2,
               step_tokens: int = 1024, device="cpu", log=None):
    """P0's DAS: rank-k rotation R; interchange x_B + R R^T (x_A - x_B) at positions first_diff..T-1 of the run on B.

    The train pairs are split into fit / validation (seeded); the rotation is trained on the fit part only and validity is
    reported on the validation part. Per pair, the unpatched residual at `layer` and the FULL patch (x_A at every patched
    position) are computed once without gradients: D_full and the full patch's distribution are the targets of the bounded
    objectives, and D_B (no patch) is the baseline:
        ce: cross-entropy toward ans_A (v1's loss)
        ld: mean of (NLDR_i - 1)^2, NLDR_i = (D_patch - D_B) / (D_full - D_B), denominator clamped to |.| >= 0.5
        kl: KL(full || patch) over the pair's candidate answers + a 'rest' bucket
    Returns (U [d, k] float32 numpy, info)."""
    import numpy as np

    n = len(train_idx)
    fit_pos, val_pos = split_holdout(n, holdout, seed)
    with torch.inference_mode(False), torch.enable_grad():
        xA = spans.x_A.clone().float()
        xB = spans.x_B.clone().float()
        pre = []                                              # per pair: (T, fd, R0 cpu, tB cpu, ans_A, ans_B, cand, D_B, D_full, logp_full)
        with torch.no_grad():
            for q, i in enumerate(train_idx):
                r = rows[i]
                tB = torch.tensor([r["corrupt_ids"]], device=device)
                R0 = runtime.resid_at(model, tB, layer)
                lgB = runtime.last_logits_from(model, R0, layer, tB)[0]
                x = R0.clone()
                sl = spans.rows(q)
                x[0, r["first_diff"]:r["T"]] = xA[sl].to(device).to(x.dtype)
                lgF = runtime.last_logits_from(model, x, layer, tB)[0]
                a, b = int(r["ans_A"]), int(r["ans_B"])
                cand = _cand(r)
                pre.append({"T": int(r["T"]), "fd": int(r["first_diff"]), "R0": R0[0].cpu(), "tB": tB[0].cpu(), "a": a, "b": b,
                            "cand": cand, "D_B": float(lgB[a] - lgB[b]), "D_full": float(lgF[a] - lgF[b]),
                            "lp_full": _bucket_logp(lgF, cand).cpu(), "sl": sl})
        rot = Rotation(xA.shape[1], k, init_seed).to(device)
        opt = torch.optim.Adam(rot.parameters(), lr=lr)
        rng = np.random.default_rng(seed + 17)

        def batches(pos):
            by_T = {}
            for p in pos:
                by_T.setdefault(pre[p]["T"], []).append(p)
            out = []
            for T, ps in by_T.items():
                per = max(1, int(step_tokens) // max(1, T))
                ps = [ps[j] for j in rng.permutation(len(ps))]
                out += [ps[j:j + per] for j in range(0, len(ps), per)]
            return [out[j] for j in rng.permutation(len(out))]

        def forward(ps, grad: bool):
            T = pre[ps[0]]["T"]
            R0 = torch.stack([pre[p]["R0"] for p in ps]).to(device)
            tB = torch.stack([pre[p]["tB"] for p in ps]).to(device)
            x = R0.clone()
            for j, p in enumerate(ps):
                e = pre[p]
                x[j, e["fd"]:T] = rot.intervene(xB[e["sl"]].to(device), xA[e["sl"]].to(device)).to(x.dtype)
            with torch.set_grad_enabled(grad):
                return runtime.last_logits_from(model, x, layer, tB)

        def loss_of(lg, ps):
            out = []
            for j, p in enumerate(ps):
                e = pre[p]
                if objective == "ce":
                    out.append(nn.functional.cross_entropy(lg[j:j + 1].float(), torch.tensor([e["a"]], device=lg.device)))
                elif objective == "ld":
                    den = e["D_full"] - e["D_B"]
                    den = den if abs(den) >= 0.5 else (0.5 if den >= 0 else -0.5)
                    nldr = ((lg[j, e["a"]] - lg[j, e["b"]]).float() - e["D_B"]) / den
                    out.append((nldr - 1.0) ** 2)
                else:
                    lp = _bucket_logp(lg[j], e["cand"])
                    pf = e["lp_full"].to(lp.device)
                    out.append((pf.exp() * (pf - lp)).sum())
            return torch.stack(out).mean()

        oom = {"backoffs": 0}

        def step(ps):
            """One optimizer step on batch ps; on a CUDA OOM the batch is split in halves (each its own step) and the token
            budget of later batches is halved, so an 11 GB card degrades to v1's batch-1 steps instead of failing.
            The retry runs AFTER the except block: inside it the exception's traceback still references the failed batch's whole
            activation graph (several GB), so the half batches would run out of memory too (P21, 2026-11-26: running-sum DAS at
            layer 2 backpropagates through 23 blocks) and the fit would fail at batch size 1."""
            nonlocal step_tokens
            loss = None
            try:
                loss = loss_of(forward(ps, True), ps)
                opt.zero_grad()
                loss.backward()
                opt.step()
                return float(loss.detach()) * len(ps)
            except Exception as e:  # noqa: BLE001 - re-raised unless it is an OOM we can back off from
                from harness.device import is_oom
                if not is_oom(e) or len(ps) == 1:
                    raise
            from harness.device import empty_cache
            loss = None                                   # drop the graph of a failed backward before retrying
            opt.zero_grad()
            empty_cache()
            oom["backoffs"] += 1
            step_tokens = max(1, step_tokens // 2)
            h = len(ps) // 2
            return step(ps[:h]) + step(ps[h:])

        losses = []
        for ep in range(epochs):
            tot, cnt = 0.0, 0
            for ps in batches(fit_pos):
                tot += step(ps)
                cnt += len(ps)
            losses.append(tot / max(1, cnt))
            if log:
                log(f"DAS v2 k={k} {objective} epoch {ep + 1}/{epochs}: loss {losses[-1]:.4f}")

        def evaluate(pos):
            if not pos:
                return {"n": 0}
            acc, nl, over = [], [], []
            with torch.no_grad():
                for ps in [[p] for p in pos]:
                    lg = forward(ps, False)[0]
                    e = pre[ps[0]]
                    acc.append(int(lg.argmax().item() == e["a"]))
                    den = e["D_full"] - e["D_B"]
                    v = (float(lg[e["a"]] - lg[e["b"]]) - e["D_B"]) / den if abs(den) > 1e-6 else float("nan")
                    nl.append(v)
                    over.append(v > 1.05)
            nl_ = np.array(nl, dtype=float)
            return {"n": len(pos), "iia": float(np.mean(acc)), "nldr_mean": float(np.nanmean(nl_)),
                    "nldr_median": float(np.nanmedian(nl_)), "share_above_full": float(np.mean(over)),
                    "full_iia": float(np.mean([pre[p]["D_full"] > 0 for p in pos]))}

        fit_ev, val_ev = evaluate(fit_pos), evaluate(val_pos)
        U = rot.R.detach().float().cpu().numpy()
    info = {"k": k, "version": 2, "objective": objective, "epochs": epochs, "lr": lr, "seed": seed, "init_seed": init_seed,
            "holdout": holdout, "step_tokens": step_tokens, "oom_backoffs": oom["backoffs"], "n_train": n, "n_fit": len(fit_pos), "n_val": len(val_pos),
            "losses": losses, "loss_first_epoch": losses[0] if losses else None, "loss_last_epoch": losses[-1] if losses else None,
            "fit": fit_ev, "val": val_ev,
            # v1-compatible fields (make_final / E9 rules read these): validity judged on HELD-OUT pairs in v2
            "train_interchange_acc": fit_ev.get("iia"), "val_interchange_acc": val_ev.get("iia")}
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

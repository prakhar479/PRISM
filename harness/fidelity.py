"""E12: how well one SAE reconstructs ordinary text of one language/script (handoff 7.10).

Per (token-window set, layer, SAE) it measures, over positions >= skip of every window (BOS / attention sinks untouched):
  CE clean / spliced (x -> x_hat) / mean-ablated / zero-ablated, and
  CE loss recovered = (CE_ablate - CE_splice) / (CE_ablate - CE_clean), for both ablations (the pilot reported the mean one),
  FVU = sum |x - x_hat|^2 / sum |x - mean|^2, |eps|/|x| (mean of per-position ratios), L0 (active latents per position).
Everything is kept per window so that confidence intervals come from a bootstrap over windows (the unit of resampling),
not over tokens. The mean residual is taken over the same set (as the pilot did); pass 1 collects it, pass 2 splices.
The math is in pure functions (`recovered`, `summarize`) tested without a model; `measure` drives the model.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from harness.model import hook_name
from harness.sae import decompose


def recovered(ce_clean, ce_splice, ce_ablate):
    """(CE_ablate - CE_splice) / (CE_ablate - CE_clean); nan when ablation costs nothing."""
    d = ce_ablate - ce_clean
    return float("nan") if abs(d) < 1e-9 else (ce_ablate - ce_splice) / d


def summarize(per_doc: Dict[str, np.ndarray], n_scored: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Point estimates and 95% bootstrap CIs over windows. `per_doc` has, per window: nll_clean / nll_splice / nll_mean /
    nll_zero (summed NLL over scored predictions), pred (number of scored predictions), fvu_num, fvu_den, eps_ratio_sum,
    l0_sum, pos (positions)."""
    n = len(per_doc["pred"])
    def stats(ix):
        p = per_doc["pred"][ix].sum()
        c, s, m, z = (per_doc[k][ix].sum() / p for k in ("nll_clean", "nll_splice", "nll_mean", "nll_zero"))
        pos = per_doc["pos"][ix].sum()
        return {"ce_clean": c, "ce_splice": s, "ce_mean_ablate": m, "ce_zero_ablate": z,
                "recovered_mean": recovered(c, s, m), "recovered_zero": recovered(c, s, z),
                "ce_delta": s - c,
                "fvu": per_doc["fvu_num"][ix].sum() / per_doc["fvu_den"][ix].sum(),
                "eps_over_x": per_doc["eps_ratio_sum"][ix].sum() / pos, "l0": per_doc["l0_sum"][ix].sum() / pos}
    point = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [stats(rng.integers(0, n, n)) for _ in range(n_boot)]
    out = {"n_windows": n, "n_tokens": int(n_scored)}
    for k, v in point.items():
        arr = np.array([b[k] for b in boots], dtype=float)
        out[k] = float(v)
        out[k + "_lo"], out[k + "_hi"] = (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))) if not np.all(np.isnan(arr)) else (float("nan"),) * 2
    return out


def to_device(sae, device):
    for a in ("W_enc_T", "b_enc", "threshold", "W_dec", "b_dec"):
        setattr(sae, a, getattr(sae, a).to(device))
    return sae


def _doc_nll(model, runtime, tokens: torch.Tensor, hooks, skip: int) -> torch.Tensor:
    """Summed NLL per row over predictions at positions skip..L-2 (logits in position chunks)."""
    R = runtime.final_resid(model, tokens, hooks)
    B, L = tokens.shape
    step = max(1, runtime.LOGIT_CHUNK_BYTES // (B * model.cfg.d_vocab * 4 * 2))
    tot = torch.zeros(B, dtype=torch.float64)
    for a in range(skip, L - 1, step):
        b = min(a + step, L - 1)
        lp = torch.log_softmax(runtime.head(model, R[:, a:b]), -1)
        tot -= lp.gather(-1, tokens[:, a + 1:b + 1, None].to(lp.device)).sum((1, 2)).double().cpu()
    return tot


@torch.inference_mode()
def measure(model, sae, layer: int, ids: Sequence[Sequence[int]], skip: int, budget, device, log=print) -> Dict[str, np.ndarray]:
    """Per-window arrays described in `summarize` for one SAE at one layer."""
    from src.models import runtime
    from harness.device import empty_cache, is_oom
    docs = [list(r) for r in ids]
    T = len(docs[0])
    hn = hook_name(layer)
    sae = to_device(sae, device)
    d = int(sae.b_dec.shape[0])

    def loop(fn):
        a = 0
        while a < len(docs):
            n = budget.rows_for(T)
            t = torch.tensor(docs[a:a + n], dtype=torch.long, device=device)
            try:
                fn(a, t)
            except Exception as e:  # noqa: BLE001
                if is_oom(e) and budget.shrink():
                    empty_cache()
                    continue
                raise
            a += len(t)

    N = len(docs)
    z = lambda: np.zeros(N)
    out = {k: z() for k in ("nll_clean", "nll_splice", "nll_mean", "nll_zero", "pred", "fvu_num", "fvu_den", "eps_ratio_sum", "l0_sum", "pos")}
    pred = (T - 1 - skip)
    out["pred"][:] = pred
    out["pos"][:] = T - skip
    s1 = torch.zeros(d, dtype=torch.float64)

    # pass 1: clean NLL and the mean residual
    def p1(a, t):
        acc = {}
        def grab(act, hook):
            acc["x"] = act[:, skip:].detach().double().sum((0, 1)).cpu()
        nll = _doc_nll(model, runtime, t, [(hn, grab)], skip)
        out["nll_clean"][a:a + len(t)] = nll.numpy()
        nonlocal s1
        s1 += acc["x"]
    loop(p1)
    mean = (s1 / (N * (T - skip))).float()
    mean_d = mean.to(device)

    # pass 2: spliced / mean-ablated / zero-ablated NLL and the reconstruction statistics
    def p2(a, t):
        B = len(t)
        st = {}
        def splice(act, hook):
            act = act.clone()
            x = act[:, skip:]
            xh, eps, l0 = decompose(sae, x.reshape(-1, x.shape[-1]), device=device)
            x32 = x.float().reshape(-1, x.shape[-1])
            sh = (B, T - skip)
            st["num"] = (eps ** 2).sum(-1).reshape(sh).double().sum(1).cpu()
            st["den"] = ((x32 - mean_d) ** 2).sum(-1).reshape(sh).double().sum(1).cpu()
            st["ratio"] = (eps.norm(dim=-1) / x32.norm(dim=-1).clamp_min(1e-12)).reshape(sh).double().sum(1).cpu()
            st["l0"] = l0.reshape(sh).double().sum(1).cpu()
            act[:, skip:] = xh.reshape(x.shape).to(act.device, act.dtype)
            return act
        def ablate(val):
            def f(act, hook):
                act = act.clone()
                act[:, skip:] = val.to(act.device, act.dtype)
                return act
            return f
        zero = torch.zeros_like(mean_d)
        sl = slice(a, a + B)
        out["nll_splice"][sl] = _doc_nll(model, runtime, t, [(hn, splice)], skip).numpy()
        out["nll_mean"][sl] = _doc_nll(model, runtime, t, [(hn, ablate(mean_d))], skip).numpy()
        out["nll_zero"][sl] = _doc_nll(model, runtime, t, [(hn, ablate(zero))], skip).numpy()
        out["fvu_num"][sl], out["fvu_den"][sl] = st["num"].numpy(), st["den"].numpy()
        out["eps_ratio_sum"][sl], out["l0_sum"][sl] = st["ratio"].numpy(), st["l0"].numpy()
    loop(p2)
    return out

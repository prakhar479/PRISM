"""Task-agnostic output sensitivity of a layer (batch 2, branch repair; docs/NEXT_BATCH_PLAN.md H1).

The Fisher information of the model's next-token distribution with respect to the residual stream at layer L,

    F_L = E_x E_{y ~ p(.|x)} [ g g^T ],     g = d( sum_s log p(y_s | x_<s) ) / d(resid_L at position t),

where y_s is *sampled from the model's own distribution* at every position s >= skip (teacher-forced prefix), so g is the score
vector and E[g g^T] is the Fisher (not the empirical Fisher). g_t is the TOTAL derivative: it includes the effect of resid_L at
position t on the predictions at later positions through attention. For a small perturbation delta of resid_L at position t the
expected KL between the clean and perturbed next-token distributions, summed over the affected positions, is delta^T F delta / 2:
F says which residual directions the model's output is sensitive to, on ordinary text, with no task in sight.

One pass gives every requested layer: a zero tensor delta_L (a float32 leaf) is added to resid_post of each layer, and
d(loss)/d(delta_L) is the gradient with respect to that layer's residual. Cached per (model, corpus, layer) as an `Entry`
("sensitivity"); the add-on fitter (harness.dicts.fit_addon) reads it to choose error subspaces.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import torch

from harness.model import hook_name
from src.models import runtime

log = logging.getLogger("harness")
SENS_WINDOWS = 2000            # generic windows (documents) of the fit set whose positions enter F: part of the cache key
SENS_SAMPLE = True             # y ~ p (Fisher); False uses the observed next token (empirical Fisher)
SENS_FORMAT = 1                # bump when the estimator changes


def entry(cache, spec, corpus_hash: str, layer: int, skip: int):
    return cache.entry("sensitivity", model=spec.key(), corpus=corpus_hash, layer=int(layer), skip=int(skip),
                       windows=SENS_WINDOWS, sample=SENS_SAMPLE, version=SENS_FORMAT)


def score_batch(model, tokens: torch.Tensor, layers: Sequence[int], skip: int, sample: bool = True,
                generator: Optional[torch.Generator] = None, positions: Optional[tuple] = None) -> Dict[int, torch.Tensor]:
    """Gradients g [B, T, d] (float32, on tokens' device) of sum_{s=skip}^{T-2} log p(y_s | x_<=s) with respect to resid_post of
    each layer in `layers`. The loss uses positions skip..T-2 (the prediction at s reads token s); g at positions after skip.. is
    what F accumulates (the caller slices). `positions` = (a, b) restricts the loss to predictions a..b-1 (tests)."""
    B, T = tokens.shape
    d = int(model.cfg.d_model)
    deltas = {L: torch.zeros(B, T, d, device=tokens.device, dtype=torch.float32, requires_grad=True) for L in layers}
    hooks = []
    for L in layers:
        def fn(act, hook, L=L):
            return act + deltas[L].to(act.dtype)
        hooks.append((hook_name(L), fn))
    lo, hi = positions if positions is not None else (skip, T - 1)
    with torch.inference_mode(False), torch.enable_grad():
        R = runtime.final_resid(model, tokens, hooks)                      # [B, T, d]
        vocab = int(model.cfg.d_vocab)
        step = max(1, runtime.LOGIT_CHUNK_BYTES // (B * vocab * 4 * 3))
        total = R.new_zeros((), dtype=torch.float32)
        for a in range(lo, hi, step):
            b = min(a + step, hi)
            lp = torch.log_softmax(runtime.head(model, R[:, a:b]), -1)     # [B, c, V] float32
            if sample:
                y = torch.multinomial(lp.detach().exp().reshape(-1, vocab), 1, generator=generator).reshape(B, b - a)
            else:
                y = tokens[:, a + 1:b + 1]
            total = total + lp.gather(-1, y[..., None].to(lp.device)).sum()
        total.backward()
    return {L: deltas[L].grad.detach() for L in layers}


@torch.no_grad()
def accumulate(F: Dict[int, torch.Tensor], g: Dict[int, torch.Tensor], skip: int) -> int:
    """F[L] += sum over positions skip..T-2 of g g^T (float64). Returns the number of rows added."""
    n = 0
    for L, gl in g.items():
        rows = gl[:, skip:gl.shape[1] - 1].reshape(-1, gl.shape[-1]).double()
        F[L] += rows.T @ rows
        n = len(rows)
    return n


def fit(model, ids: Sequence[Sequence[int]], layers: Sequence[int], skip: int, device, batch: int = 4, sample: bool = SENS_SAMPLE,
        seed: int = 0, progress=None) -> Dict[int, dict]:
    """F_L for every layer from the windows `ids` ([N][T] token ids). Returns {layer: {"F": [d,d] float64 (mean over rows),
    "n_rows": int, "trace": float}}."""
    d = int(model.cfg.d_model)
    F = {L: torch.zeros(d, d, device=device, dtype=torch.float64) for L in layers}
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = 0
    for a in range(0, len(ids), batch):
        t = torch.tensor([list(r) for r in ids[a:a + batch]], device=device)
        g = score_batch(model, t, layers, skip, sample=sample, generator=gen)
        rows += accumulate(F, g, skip)
        if progress and (a // batch) % 50 == 0:
            progress(a, len(ids), rows)
    out = {}
    for L in layers:
        M = (F[L] / max(1, rows)).cpu()
        out[L] = {"F": M, "n_rows": rows, "trace": float(torch.trace(M))}
    return out


def fit_cached(model, spec, cache, corpus: dict, layers: Sequence[int], device, batch: int = 4, seed: int = 0) -> Dict[int, object]:
    """Fill (and return) the cached entries of `layers` that are missing; `corpus` = {"ids", "hash", "skip"}."""
    ents = {L: entry(cache, spec, corpus["hash"], L, corpus["skip"]) for L in layers}
    todo = [L for L, e in ents.items() if not e.complete]
    if todo:
        ids = corpus["ids"][:SENS_WINDOWS]
        log.info("sensitivity: %d windows, layers %s", len(ids), todo)
        res = fit(model, ids, todo, corpus["skip"], device, batch=batch, seed=seed,
                  progress=lambda a, n, r: log.info("sensitivity %d/%d windows (%d rows)", a, n, r))
        for L in todo:
            e = ents[L]
            with e.writer_lock():
                e.open()
                if not e.complete:
                    e.write_shard(0, {"F": res[L]["F"].float()})
                    e.mark_complete({"n_rows": res[L]["n_rows"], "trace": res[L]["trace"], "windows": len(ids),
                                     "sample": SENS_SAMPLE, "skip": corpus["skip"]})
    return ents


def load(entry_) -> torch.Tensor:
    """F [d, d] float64 from a complete entry."""
    return entry_.require_complete().read_shard(0)["F"].double()

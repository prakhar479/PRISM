"""Memory-lean forward passes and resource planning (works on 8 GB GPUs, scales up on bigger ones).

Every experiment only needs the logits at the LAST position, but a plain forward computes [B, L, 256k] logits
(~2 GB per 4k-token batch, plus softcap temporaries). Here the model runs to the final residual
(stop_at_layer=n_layers) and ln_final + unembed (+ Gemma-2 softcap) are applied to the last position only.

Other levers, all chosen automatically from the GPU size (config block `runtime`):
    offload_embed      the 256k x 2304 embedding (1.2 GB) stays on CPU; embeddings are gathered there and
                       the forward starts at block 0 with the residual as input (bit-identical).
    max_batch_tokens   token budget per forward batch, from the VRAM left after the model is loaded.
    resume_from_layer  patched runs start at block layer+1 from the cached unpatched residual instead of
                       recomputing blocks 0..layer for every intervention (verified once per run).
    cpu_threads        torch / BLAS threads (physical cores); cpu_workers: parallel CPU jobs (exp2 probes)
                       that overlap with GPU work.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import torch

log = logging.getLogger("prism")
GiB = 2 ** 30
BYTES_PER_TOKEN = 256 * 1024        # conservative peak activation memory per token in one Gemma-2-2b forward
LOGIT_CHUNK_BYTES = 256 * 2 ** 20   # [B, chunk, vocab] float32 slab when full-sequence logits are needed (CE)
_STATE = {"max_batch_tokens": 4096, "cpu_workers": 2}
_PHYSICAL_CORES = torch.get_num_threads()   # torch's default = physical cores (read before anyone changes it)

Hooks = Sequence[Tuple[str, Callable]]


# --------------------------------------------------------------------------- resource planning
def total_vram() -> int:
    return torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0


def free_vram() -> int:
    return torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0


def available_ram() -> int:
    """MemAvailable from /proc/meminfo (bytes); a large number if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 50


def _auto(v, default):
    return default if v in (None, "auto") else v


def runtime_cfg(cfg: Optional[dict]) -> dict:
    return dict((cfg or {}).get("runtime") or {})


def want_offload_embed(cfg: Optional[dict], device: torch.device) -> bool:
    rt = runtime_cfg(cfg)
    return device.type == "cuda" and bool(_auto(rt.get("offload_embed"), total_vram() < 16 * GiB))


def setup_runtime(cfg: dict, model, device: torch.device) -> dict:
    """Resolve the 'auto' knobs once the model is loaded; mutates cfg['batch_size'] and returns the plan."""
    rt = runtime_cfg(cfg)
    threads = int(_auto(rt.get("cpu_threads"), _PHYSICAL_CORES))   # hyperthreads only add OpenMP contention
    torch.set_num_threads(threads)
    workers = int(_auto(rt.get("cpu_workers"), max(1, min(4, threads // 3))))
    if device.type == "cuda":
        free = free_vram()
        act = min(max(0.45 * (free - 0.5 * GiB), 0.4 * GiB), 4 * GiB)
        tokens = int(_auto(rt.get("max_batch_tokens"), max(1024, min(16384, int(act // BYTES_PER_TOKEN) // 512 * 512))))
    else:
        free, tokens = 0, int(_auto(rt.get("max_batch_tokens"), 4096))
    cfg["batch_size"] = int(_auto(cfg.get("batch_size"), 128))
    _STATE.update(max_batch_tokens=tokens, cpu_workers=workers)
    from src.data import synthetic_tasks
    synthetic_tasks.MAX_BATCH_TOKENS = tokens
    plan = {"device": str(device), "total_vram_gb": round(total_vram() / GiB, 2), "free_vram_after_model_gb": round(free / GiB, 2),
            "embed_on_cpu": bool(getattr(model, "_prism_embed_on_cpu", False)), "max_batch_tokens": tokens,
            "batch_size_cap": cfg["batch_size"], "cpu_threads": threads, "cpu_workers": workers,
            "resume_from_layer": bool(rt.get("resume_from_layer", True)), "ram_available_gb": round(available_ram() / GiB, 1)}
    log.info("runtime plan: %s", plan)
    return plan


def is_oom(e: BaseException) -> bool:
    """CUDA out-of-memory, including the cuBLAS / cuDNN allocation failures that surface as RuntimeError."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e)
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "CUBLAS_STATUS_ALLOC_FAILED" in msg)


def shrink_batches(floor: int = 256) -> bool:
    """Halve the per-forward token budget (after a CUDA OOM). False once it is already at the floor."""
    cur = _STATE["max_batch_tokens"]
    if cur <= floor:
        return False
    new = max(floor, cur // 2)
    _STATE["max_batch_tokens"] = new
    from src.data import synthetic_tasks
    synthetic_tasks.MAX_BATCH_TOKENS = new
    log.warning("CUDA out of memory: max_batch_tokens %d -> %d, retrying", cur, new)
    return True


def oom_retry(fn: Callable, what: str = "", tries: int = 3):
    """Run fn(); on a CUDA OOM free the cache, halve the token budget and retry (the smaller budget is kept)."""
    import gc

    for attempt in range(tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised unless it is an OOM we can back off from
            if not is_oom(e) or attempt == tries:
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not shrink_batches():
                raise
            log.warning("retrying %s after OOM (attempt %d/%d)", what or "step", attempt + 1, tries)


def cpu_workers() -> int:
    return _STATE["cpu_workers"]


def blas_threads_per_worker() -> int:
    return max(1, torch.get_num_threads() // _STATE["cpu_workers"])


def act_reserve_bytes() -> int:
    """VRAM to keep free for activations when deciding whether an SAE can be GPU-resident."""
    return _STATE["max_batch_tokens"] * BYTES_PER_TOKEN + GiB // 4


# --------------------------------------------------------------------------- forward passes
def _input(model, tokens: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """(input, kwargs) for run_with_hooks: token ids, or CPU-gathered embeddings when the embedding is offloaded."""
    if getattr(model, "_prism_embed_on_cpu", False):
        emb = model.embed(tokens.to(model.embed.W_E.device))
        return emb.to(model.cfg.device, non_blocking=True), {"start_at_layer": 0}
    return tokens, {}


def head(model, resid: torch.Tensor) -> torch.Tensor:
    """ln_final + unembed (+ output softcap) on residuals [..., d] -> float32 logits [..., vocab]."""
    if getattr(model.cfg, "normalization_type", None) is not None:
        resid = model.ln_final(resid)
    logits = model.unembed(resid).float()
    cap = float(getattr(model.cfg, "output_logits_soft_cap", 0.0) or 0.0)
    if cap > 0:
        logits = cap * torch.tanh(logits / cap)
    return logits


def final_resid(model, tokens: torch.Tensor, fwd_hooks: Hooks = ()) -> torch.Tensor:
    """Residual after the last block [B, L, d] (no logits computed)."""
    inp, kw = _input(model, tokens)
    return model.run_with_hooks(inp, fwd_hooks=list(fwd_hooks), stop_at_layer=model.cfg.n_layers, tokens=tokens, **kw)


def last_logits(model, tokens: torch.Tensor, fwd_hooks: Hooks = ()) -> torch.Tensor:
    """float32 logits at the last position [B, vocab]."""
    return head(model, final_resid(model, tokens, fwd_hooks)[:, -1:])[:, 0]


def resid_at(model, tokens: torch.Tensor, layer: int) -> torch.Tensor:
    """resid_post of block `layer` for every position [B, L, d] (runs blocks 0..layer only)."""
    inp, kw = _input(model, tokens)
    return model.run_with_hooks(inp, fwd_hooks=[], stop_at_layer=layer + 1, tokens=tokens, **kw)


def last_logits_from(model, resid: torch.Tensor, layer: int, tokens: torch.Tensor) -> torch.Tensor:
    """Continue a forward from resid_post[layer] (blocks layer+1..end) -> last-position float32 logits."""
    out = model.run_with_hooks(resid, fwd_hooks=[], start_at_layer=layer + 1, stop_at_layer=model.cfg.n_layers,
                               tokens=tokens)
    return head(model, out[:, -1:])[:, 0]


def token_nll_sum(model, tokens: torch.Tensor, fwd_hooks: Hooks = (), skip: int = 0) -> float:
    """Sum of next-token NLL over predictions at positions skip..L-2, with logits computed in position chunks."""
    R = final_resid(model, tokens, fwd_hooks)
    B, L = tokens.shape
    vocab = model.cfg.d_vocab
    step = max(1, LOGIT_CHUNK_BYTES // (B * vocab * 4 * 2))
    tot = 0.0
    for a in range(skip, L - 1, step):
        b = min(a + step, L - 1)
        lp = torch.log_softmax(head(model, R[:, a:b]), -1)
        tot -= float(lp.gather(-1, tokens[:, a + 1:b + 1, None].to(lp.device)).sum())
    return tot


def resume_enabled(model) -> bool:
    return bool(getattr(model, "_prism_resume", False))


def set_resume(model, on: bool) -> None:
    model._prism_resume = bool(on)


def pin(t: torch.Tensor) -> torch.Tensor:
    """Page-locked copy for fast async host->device copies (falls back silently, e.g. WSL pinned-memory limits)."""
    if not torch.cuda.is_available():
        return t
    try:
        return t.pin_memory()
    except RuntimeError:
        return t

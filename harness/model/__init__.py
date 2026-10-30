"""Model-dependent code: loading (real gemma-2-2b at a pinned revision, or a tiny random Gemma-2 for tests) and the
final-position output summary. Forward primitives are the pilot's (src/models/runtime.py): last-position-only
unembedding with Gemma-2's soft-cap, resume from layer L+1, CPU-held embedding on small GPUs.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch

from harness.device import dtype_name, dtype_of, is_cuda

log = logging.getLogger("harness")
HOOK_FMT = "blocks.{layer}.hook_resid_post"          # pilot hook site (Gemma Scope 'res' SAEs)


@dataclass(frozen=True)
class ModelSpec:
    name: str = "gemma-2-2b"                         # "gemma-2-2b" | "tiny"
    revision: Optional[str] = None                   # HF commit of google/gemma-2-2b (required for real runs)
    dtype: str = "bfloat16"
    site: str = "resid_post"
    tiny_seed: int = 0
    tiny_vocab: int = 0
    tiny_layers: int = 4

    def key(self) -> dict:
        return asdict(self)


def hook_name(layer: int) -> str:
    return HOOK_FMT.format(layer=layer)


def tiny_gemma2(d_vocab: int, n_layers: int = 4, d_model: int = 64, seed: int = 0, dtype="float32", device="cpu"):
    """Random Gemma-2-shaped HookedTransformer: RMS pre+post norms, attention/logit soft-caps, rotary, GQA, gated MLP,
    alternating local/global attention. For tests only; its numbers are never results."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=n_layers, d_model=d_model, n_ctx=1024, d_head=16, n_heads=4, n_key_value_heads=2, d_mlp=2 * d_model,
        d_vocab=d_vocab, act_fn="gelu_pytorch_tanh", normalization_type="RMS", use_normalization_before_and_after=True,
        attn_scores_soft_cap=50.0, output_logits_soft_cap=30.0, positional_embedding_type="rotary", rotary_dim=16,
        gated_mlp=True, final_rms=True, use_attn_scale=True, attn_scale=4.0,
        attn_types=(["local", "global"] * n_layers)[:n_layers], window_size=8, original_architecture="Gemma2ForCausalLM",
        default_prepend_bos=True, seed=seed, init_weights=True, dtype=dtype_of(dtype), device=str(device))
    m = HookedTransformer(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m._prism_embed_on_cpu = False
    return m


def load_gemma(spec: ModelSpec, device: torch.device, offload_embed: bool = False, tokenizer=None):
    """gemma-2-2b through TransformerLens `from_pretrained_no_processing` (as the pilot), with the HF weights loaded at
    the pinned revision and handed over as hf_model. Built on CPU, moved child by child (embedding optionally kept on
    CPU, as the pilot did on < 16 GB GPUs)."""
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    if not spec.revision:
        raise ValueError("real model runs need a pinned model revision (config model.revision)")
    dt = dtype_of(spec.dtype)
    try:
        hf = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", revision=spec.revision, dtype=dt)
    except TypeError:
        hf = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", revision=spec.revision, torch_dtype=dt)
    model = HookedTransformer.from_pretrained_no_processing(spec.name, hf_model=hf, dtype=dt, device=str(device),
                                                            move_to_device=False, tokenizer=tokenizer)
    del hf
    offload = bool(offload_embed and is_cuda(device))
    for name, child in model.named_children():
        if not (offload and name in ("embed", "hook_embed")):
            child.to(device)
    model.cfg.device = str(device)
    model._prism_embed_on_cpu = offload
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def build_model(spec: ModelSpec, device: torch.device, offload_embed: bool = False, tokenizer=None):
    if spec.name == "tiny":
        return tiny_gemma2(spec.tiny_vocab, n_layers=spec.tiny_layers, seed=spec.tiny_seed, dtype=spec.dtype, device=device)
    return load_gemma(spec, device, offload_embed, tokenizer)


def final_outputs(logits: torch.Tensor, ans_A: torch.Tensor, ans_B: torch.Tensor, cand: torch.Tensor) -> Dict[str, np.ndarray]:
    """Post-softcap float32 last-position logits [B, V] -> stored quantities (see harness.metrics.Out)."""
    lg = logits.float()
    idx = torch.arange(len(lg), device=lg.device)
    c = cand.to(lg.device)
    cl = lg.gather(1, c.clamp_min(0))
    cl = torch.where(c >= 0, cl, torch.full_like(cl, float("nan")))
    return {"logit_A": lg[idx, ans_A.to(lg.device)].cpu().numpy(), "logit_B": lg[idx, ans_B.to(lg.device)].cpu().numpy(),
            "lse": torch.logsumexp(lg, -1).cpu().numpy(), "top1_id": lg.argmax(-1).cpu().numpy(),
            "cand_logits": cl.cpu().numpy()}

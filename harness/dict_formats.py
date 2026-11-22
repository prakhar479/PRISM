"""Checkpoint formats of registry dictionaries -> harness.dicts.Dictionary (latent-major, raw-x encoder bias).

Verified formats (file headers and library source, 2026-11-22; see docs/PILOT_RUNBOOK.md, 'Dictionary sources'):

saprmarks/dictionary_learning checkpoints (SAEBench 'date-0107/0108' sweeps, the Bhalla et al. temporal SAE): ae.pt state dict,
F32, inputs are RAW activations (no norm_factor in any SAEBench config; dictionary_learning folds a normalisation into the
weights at save time).
  standard / p_anneal  bias [d], encoder.weight [S,d], encoder.bias [S], decoder.weight [d,S]
                       f = ReLU(enc(x - bias)); x_hat = dec(f) + bias
  topk                 b_dec, encoder.weight, encoder.bias, decoder.weight [d,S], k, threshold
                       f = TopK_k(ReLU(enc(x - b_dec)))                       (use_threshold=False at inference)
  batchtopk            same keys; f = r * 1[r > threshold], r = ReLU(enc(x - b_dec)) (one global scalar threshold)
  matryoshka           W_enc [d,S], b_enc, W_dec [S,d], b_dec, k, threshold, group_sizes; encoded like batchtopk
  jumprelu             W_enc [d,S], b_enc, W_dec [S,d], b_dec, threshold [S]; pre = x W_enc + b_enc; f = ReLU(pre) 1[pre > thr]
  gated                decoder_bias, encoder.weight [S,d] (no bias), r_mag, gate_bias, mag_bias, decoder.weight [d,S]
                       h = enc(x - decoder_bias); f = 1[h + gate_bias > 0] * ReLU(exp(r_mag) h + mag_bias)
Gemma Scope 2 (google/gemma-scope-2-*-pt, resid_post/layer_{L}_width_{W}_l0_{small|medium|big}/params.safetensors):
  w_enc [d,S], w_dec [S,d], b_enc, b_dec, threshold; z = x w_enc + b_enc; f = z 1[z > threshold] (Matryoshka-trained, plain
  JumpReLU at inference; nothing subtracted from x).
Every pre-encoder subtraction of a decoder bias c is folded: enc(x - c) = x W^T + (b - W c).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from harness.dicts import Dictionary


def _state(paths: Sequence[Path]) -> dict:
    pt = [p for p in paths if str(p).endswith(".pt")]
    if not pt:
        raise ValueError(f"no ae.pt among {paths}")
    sd = torch.load(pt[0], map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}


def _scalar(v) -> float:
    return float(v.item() if torch.is_tensor(v) else v)


def from_dictionary_learning(paths: Sequence[Path], variant: str, layer: int, width: str, key: str,
                             dtype=torch.float32) -> Dictionary:
    sd = _state(paths)
    if variant in ("standard", "p_anneal"):
        We, be, c = sd["encoder.weight"], sd["encoder.bias"], sd["bias"]
        Wd = sd["decoder.weight"].T.contiguous()
        return Dictionary((We).to(dtype), (be - We @ c).to(dtype), torch.zeros(len(be), dtype=dtype), Wd.to(dtype), c.float(),
                          width, layer, key, kind="jumprelu")
    if variant in ("topk", "batchtopk"):
        We, be, c = sd["encoder.weight"], sd["encoder.bias"], sd["b_dec"]
        Wd = sd["decoder.weight"].T.contiguous()
        b = (be - We @ c).to(dtype)
        if variant == "topk":
            return Dictionary(We.to(dtype), b, torch.zeros(len(be), dtype=dtype), Wd.to(dtype), c.float(), width, layer, key,
                              kind="topk", k=int(_scalar(sd["k"])))
        th = max(0.0, _scalar(sd["threshold"]))
        return Dictionary(We.to(dtype), b, torch.full((len(be),), th, dtype=dtype), Wd.to(dtype), c.float(), width, layer, key,
                          kind="jumprelu")
    if variant == "matryoshka":
        We = sd["W_enc"].T.contiguous()                                   # [S, d]
        be, c, Wd = sd["b_enc"], sd["b_dec"], sd["W_dec"]
        th = max(0.0, _scalar(sd["threshold"]))
        return Dictionary(We.to(dtype), (be - We @ c).to(dtype), torch.full((len(be),), th, dtype=dtype), Wd.to(dtype), c.float(),
                          width, layer, key, kind="jumprelu")
    if variant == "jumprelu":
        We = sd["W_enc"].T.contiguous()
        th = sd["threshold"].clamp_min(0.0)                               # ReLU(pre) 1[pre > thr] = pre 1[pre > max(thr, 0)]
        return Dictionary(We.to(dtype), sd["b_enc"].to(dtype), th.to(dtype), sd["W_dec"].to(dtype), sd["b_dec"].float(),
                          width, layer, key, kind="jumprelu")
    if variant == "gated":
        We, c = sd["encoder.weight"], sd["decoder_bias"]
        Wc = We @ c
        scale = torch.exp(sd["r_mag"])
        Wd = sd["decoder.weight"].T.contiguous()
        return Dictionary(We.to(dtype), (sd["gate_bias"] - Wc).to(dtype), torch.zeros(We.shape[0], dtype=dtype), Wd.to(dtype),
                          c.float(), width, layer, key, kind="gated", mag_scale=scale.to(dtype),
                          b_mag=(sd["mag_bias"] - scale * Wc).to(dtype))
    raise ValueError(f"unknown dictionary_learning variant {variant}")


def from_gemma_scope_2(paths: Sequence[Path], layer: int, width: str, key: str, dtype=torch.float32) -> Dictionary:
    from safetensors.torch import load_file
    st = [p for p in paths if str(p).endswith(".safetensors")]
    if not st:
        raise ValueError(f"no params.safetensors among {paths}")
    t = {k: v.float() for k, v in load_file(str(st[0])).items()}
    return Dictionary(t["w_enc"].T.contiguous().to(dtype), t["b_enc"].to(dtype), t["threshold"].to(dtype), t["w_dec"].to(dtype),
                      t["b_dec"].float(), width, layer, key, kind="jumprelu")

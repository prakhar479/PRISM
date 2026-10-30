"""Peak GPU-memory estimates per stage, from config + profile only (no GPU needed). Used by plan_runs.py and the
runner's --dry-run. Numbers are deliberately conservative; preflight measurements refine the time side."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from harness.device import dtype_of
from harness.sae import parse_width

GiB = 2 ** 30
GEMMA2_2B = {"params": 2_614_341_888, "vocab": 256_000, "d_model": 2304, "n_layers": 26}
BYTES_PER_TOKEN = 256 * 1024          # pilot heuristic: peak activation memory per token in one forward (bf16)
CUDA_CONTEXT = int(0.45 * GiB)
LOGIT_TEMPS = 3                        # fp32 [rows, vocab] copies alive in head() (logits, softcap, float)


def nbytes(dtype) -> int:
    return dtype_of(dtype).itemsize


def model_bytes(dtype: str = "bfloat16", offload_embed: bool = False) -> int:
    p = GEMMA2_2B["params"] - (GEMMA2_2B["vocab"] * GEMMA2_2B["d_model"] if offload_embed else 0)
    return p * nbytes(dtype)


def sae_bytes(width: str, dtype: str = "float32", d_model: int = 2304) -> int:
    n = parse_width(width)
    return (2 * n * d_model + 2 * n) * nbytes(dtype) + 4 * d_model


def sae_dtype_for(width: str, profile) -> str:
    return "bfloat16" if parse_width(width) >= parse_width(profile.sae_bf16_min_width) else profile.sae_dtype


@dataclass
class Estimate:
    stage: str
    peak_bytes: int
    parts: dict
    placement: Optional[str] = None
    block_latents: Optional[int] = None

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / GiB


def forward_bytes(max_tokens: int, rows: int, dtype: str) -> int:
    act = max_tokens * BYTES_PER_TOKEN * nbytes(dtype) // 2
    logits = rows * GEMMA2_2B["vocab"] * 4 * LOGIT_TEMPS          # final position only
    return act + logits


def stage1(profile) -> Estimate:
    mb = model_bytes(profile.model_dtype, profile.offload_embed)
    fw = forward_bytes(profile.max_batch_tokens, profile.max_batch_size, profile.model_dtype)
    return Estimate("stage1", CUDA_CONTEXT + mb + fw, {"model": mb, "forward": fw})


def stage3(profile, seq_len: int = 400) -> Estimate:
    e = stage1(profile)
    rows = max(1, profile.max_batch_tokens // seq_len)
    resid = 2 * rows * seq_len * GEMMA2_2B["d_model"] * nbytes(profile.model_dtype)     # B residual + patched copy
    return Estimate("stage3", e.peak_bytes + resid, {**e.parts, "resid_copies": resid})


def stage2(profile, width: str, cap_bytes: Optional[int], rows_block: int = 8192) -> Estimate:
    """SAE pass without the model. Placement: gpu if the whole SAE + a row block fits under the cap, else block-wise
    over latents (block sized to ~1 GiB of weights), or cpu if there is no GPU budget."""
    dt = sae_dtype_for(width, profile)
    sb = sae_bytes(width, dt)
    work = rows_block * GEMMA2_2B["d_model"] * 4 * 3
    n = parse_width(width)
    pref = profile.sae_placement
    if cap_bytes is None or pref == "cpu":
        return Estimate("stage2", 0, {"sae": sb, "work": work}, "cpu", None)
    if pref in ("gpu", "auto") and CUDA_CONTEXT + sb + work + rows_block * n * 4 * 2 <= cap_bytes:
        return Estimate("stage2", CUDA_CONTEXT + sb + work + rows_block * n * 4 * 2, {"sae": sb, "work": work}, "gpu", None)
    per_latent = 2 * GEMMA2_2B["d_model"] * dtype_of(dt).itemsize + 8
    block = max(1024, min(n, int(GiB // per_latent)) // 1024 * 1024)
    blk = block * per_latent
    rows_part = 2048 * block * 4 * 2
    return Estimate("stage2", CUDA_CONTEXT + blk + rows_part + work, {"sae_block": blk, "work": work},
                    "blockwise-gpu", block)


def fits(est: Estimate, cap_bytes: Optional[int]) -> bool:
    return cap_bytes is None or est.peak_bytes <= cap_bytes

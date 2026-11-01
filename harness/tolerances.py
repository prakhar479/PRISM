"""Central tolerances. Never loosen one to make a test pass without a dated entry in DECISIONS.md."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float
    note: str = ""


# exact fp32 math on CPU vs golden fixtures produced by the same arithmetic
CPU_VS_GOLDEN = Tol(1e-5, 1e-5, "fp32 CPU recomputation of golden tensors (SAE, conditions)")
# float32 reconstruction identity x_hat + eps == x
RECON_IDENTITY = Tol(0.0, 1e-5, "eps is defined as x - x_hat in fp32")
# block-wise vs one-shot SAE encode/decode (fp32 summation order differs)
BLOCKWISE = Tol(1e-5, 2e-5, "latent-blocked accumulation vs one matmul")
# refit direction U vs golden U: per-column |cos|
U_COLUMN_ABS_COS = 0.999
# model-dependent, same machine, bf16: logit-difference D re-export vs golden
D_SAME_MACHINE = Tol(0.0, 0.05, "same machine/stack, batched bf16")
# batch-1 vs batched bf16 forward
D_BATCHING = Tol(0.0, 0.25, "bf16 kernels differ with batch composition")
# cross-machine / bf16 NLDR regression (handoff: +-1 point)
NLDR_POINTS = 1.0
# head() (ln_final + unembed + fp32 soft-cap on the last position) vs TransformerLens' full forward (soft-cap in the
# model dtype). Identical in fp32 (tested on DEV); in bf16 they differ by rounding: allow 2 units in the last place of
# bf16 at the largest |logit| (RUN 2026-11-01: 0.19 at |logit| ~ 20, where one bf16 ulp = 0.125). DECISIONS 2026-11-01.
HEAD_VS_FULL_ULPS = 2


def head_vs_full_atol(max_abs_logit: float, dtype_name: str) -> float:
    import math
    if dtype_name in ("float32", "fp32"):
        return 1e-3
    mant = {"bfloat16": 7, "bf16": 7, "float16": 10, "fp16": 10}[dtype_name]
    return HEAD_VS_FULL_ULPS * 2.0 ** (math.floor(math.log2(max(max_abs_logit, 1e-6))) - mant)


# harness stage-1 span residuals vs the pilot's cached span rows (golden), relative row error. Different batch
# composition changes bf16 kernels; a hook-site or position error is O(10-100%). RUN 2026-11-01: 0.0201, against the
# pilot's own batch-1 vs batched spread (reported next to it). DECISIONS 2026-11-01.
SPAN_RESID_REL = 0.05

# hook identity on the real model (writing x_B back must reproduce clean B)
HOOK_IDENTITY_D = Tol(0.0, 1e-3, "writing the cached bf16 residual back is exact up to kernel nondeterminism")

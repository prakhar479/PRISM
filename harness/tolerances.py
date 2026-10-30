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
# hook identity on the real model (writing x_B back must reproduce clean B)
HOOK_IDENTITY_D = Tol(0.0, 1e-3, "writing the cached bf16 residual back is exact up to kernel nondeterminism")

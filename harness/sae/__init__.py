"""JumpReLU SAE math (pure tensor code; identical on CPU and GPU) and Gemma Scope identifiers.

Weights use the pilot's container `src.models.sae_loader.JumpReLUSAE` (latent-major: W_enc_T [n, d], b_enc [n],
threshold [n], W_dec [n, d], b_dec [d]) so loading is byte-identical to the pilot. The arithmetic here is:
    z = x W_enc + b_enc                 (pre-activation, in the weight dtype)
    f = z * 1[z > threshold]            (JumpReLU; threshold > 0, so this equals ReLU(z) * 1[z > threshold])
    x_hat = f W_dec + b_dec             (accumulated in float32)
    eps = x - x_hat                     (float32, exact by definition)
Large widths are processed in latent blocks; weights stay where they are (CPU or device) and each block is copied
to the compute device, so the same function serves 'gpu', 'blockwise-gpu' and 'cpu' placement.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from harness.device import dtype_of

SAE_REPO = "google/gemma-scope-2b-pt-res"
BLOCK_BYTES = 256 * 2 ** 20          # target size of one [rows, block] float32 slab


def parse_width(width: str) -> int:
    w = str(width).lower()
    if w.endswith("k"):
        return int(w[:-1]) * 1024
    if w.endswith("m"):
        return int(w[:-1]) * 1024 * 1024
    return int(w)


@dataclass(frozen=True)
class SAEId:
    layer: int
    width: str
    l0: int                       # the `average_l0_<l0>` variant
    repo: str = SAE_REPO
    revision: Optional[str] = None

    @property
    def path_in_repo(self) -> str:
        return f"layer_{self.layer}/width_{self.width}/average_l0_{self.l0}/params.npz"

    @property
    def key(self) -> str:
        return f"L{self.layer}_{self.width}_l0{self.l0}"

    @property
    def d_sae(self) -> int:
        return parse_width(self.width)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- loading
def load_params(path, layer: int = -1, width: str = "?", dtype="float32", sae_id: Optional[str] = None):
    """Gemma Scope params.npz -> CPU JumpReLUSAE (the pilot's loader, memory-mapped block conversion)."""
    from src.models.sae_loader import load_gemma_scope_npz

    return load_gemma_scope_npz(str(path), width, layer, dtype_of(dtype), sae_id)


def write_random_params(path, d_model: int, d_sae: int, seed: int = 0, thr: float = 0.5) -> Path:
    """A random JumpReLU SAE in Gemma Scope's params.npz layout (uncompressed; W_enc [d, n], W_dec [n, d])."""
    g = np.random.default_rng(seed)
    W_dec = g.standard_normal((d_sae, d_model)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_enc = (W_dec.T * 1.5).astype(np.float32)
    b_enc = (g.standard_normal(d_sae) * 0.1).astype(np.float32)
    threshold = (thr * (0.5 + g.random(d_sae))).astype(np.float32)
    b_dec = (g.standard_normal(d_model) * 0.01).astype(np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, threshold=threshold)
    return path


# --------------------------------------------------------------------------- arithmetic
def _weights(sae) -> Tuple[torch.Tensor, ...]:
    if getattr(sae, "streamed", False):
        raise ValueError("harness SAE math expects a non-streamed JumpReLUSAE (weights on CPU or device)")
    return sae.W_enc_T, sae.b_enc, sae.threshold, sae.W_dec, sae.b_dec


def latent_blocks(sae, block_latents: Optional[int], device) -> Iterator[Tuple[int, Tuple[torch.Tensor, ...]]]:
    """Yield (start, (W_enc_T, b_enc, threshold, W_dec)) blocks on `device`. block_latents=None: one block."""
    We, be, th, Wd, _ = _weights(sae)
    n = We.shape[0]
    step = n if not block_latents else int(block_latents)
    for a in range(0, n, step):
        yield a, tuple(t[a:a + step].to(device, non_blocking=True) for t in (We, be, th, Wd))


def _row_step(n_block: int, rows: int) -> int:
    return max(1, min(rows, BLOCK_BYTES // max(1, n_block * 4 * 3)))


@torch.inference_mode()
def decompose(sae, X: torch.Tensor, block_latents: Optional[int] = None, device=None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X [N, d] -> (x_hat [N, d] fp32, eps [N, d] fp32, l0 [N]) on `device` (default: the weights' device)."""
    device = torch.device(device) if device is not None else sae.W_enc_T.device
    x32 = X.reshape(-1, X.shape[-1]).to(device, torch.float32)
    xw = x32.to(sae.W_enc_T.dtype)
    x_hat = torch.zeros_like(x32)
    l0 = torch.zeros(len(x32), dtype=torch.long, device=device)
    for _, (We, be, th, Wd) in latent_blocks(sae, block_latents, device):
        Wd32 = Wd.float()
        rb = _row_step(len(be), len(x32))
        for r in range(0, len(x32), rb):
            z = xw[r:r + rb] @ We.T + be
            f = z * (z > th)
            l0[r:r + rb] += (f != 0).sum(-1)
            x_hat[r:r + rb] += f.float() @ Wd32
    x_hat += sae.b_dec.to(device).float()
    shape = X.shape
    return x_hat.reshape(shape), (x32 - x_hat).reshape(shape), l0.reshape(shape[:-1])


@dataclass
class SparseCodes:
    """Sparse pre-activations of N rows: entries with z > 0 and z >= alpha_min * threshold.
    active = z > threshold (these are exactly the nonzero f)."""
    row: torch.Tensor
    col: torch.Tensor
    z: torch.Tensor
    threshold: torch.Tensor        # threshold[col] for each entry
    n_rows: int
    alpha_min: float

    @property
    def active(self) -> torch.Tensor:
        return self.z > self.threshold

    def l0(self) -> torch.Tensor:
        return torch.bincount(self.row[self.active], minlength=self.n_rows)

    def subset_rows(self, rows: torch.Tensor) -> "SparseCodes":
        """Codes for rows[k] renumbered to k."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        remap = torch.full((self.n_rows,), -1, dtype=torch.long)
        remap[rows] = torch.arange(len(rows))
        keep = remap[self.row] >= 0
        return SparseCodes(remap[self.row[keep]], self.col[keep], self.z[keep], self.threshold[keep], len(rows),
                           self.alpha_min)


@torch.inference_mode()
def encode_sparse(sae, X: torch.Tensor, alpha_min: float = 0.5, block_latents: Optional[int] = None,
                  device=None) -> SparseCodes:
    """Sparse near-threshold pre-activations (CPU tensors); same arithmetic as decompose()."""
    device = torch.device(device) if device is not None else sae.W_enc_T.device
    xw = X.reshape(-1, X.shape[-1]).to(device, sae.W_enc_T.dtype)
    rows, cols, zs, ths = [], [], [], []
    for a, (We, be, th, _) in latent_blocks(sae, block_latents, device):
        rb = _row_step(len(be), len(xw))
        for r in range(0, len(xw), rb):
            z = xw[r:r + rb] @ We.T + be
            keep = (z > 0) & (z >= alpha_min * th)
            i, j = keep.nonzero(as_tuple=True)
            rows.append((i + r).cpu())
            cols.append((j + a).cpu())
            zs.append(z[i, j].float().cpu())
            ths.append(th[j].float().cpu())
    cat = lambda xs, dt: torch.cat(xs) if xs else torch.zeros(0, dtype=dt)
    return SparseCodes(cat(rows, torch.long), cat(cols, torch.long), cat(zs, torch.float32), cat(ths, torch.float32),
                       len(xw), alpha_min)


def decode_entries(sae, codes: SparseCodes, mask: torch.Tensor) -> torch.Tensor:
    """sum over selected entries of z * W_dec[col], per row -> [n_rows, d] float32 (no bias)."""
    Wd = sae.W_dec
    out = torch.zeros(codes.n_rows, Wd.shape[1], dtype=torch.float32)
    if mask.any():
        r, c, z = codes.row[mask], codes.col[mask], codes.z[mask]
        out.index_add_(0, r, z[:, None] * Wd[c.to(Wd.device)].float().cpu())
    return out


# --------------------------------------------------------------------------- inventory / L0 matching
_L0_RE = re.compile(r"layer_(\d+)/width_([0-9]+[km])/average_l0_(\d+)")


def parse_tree(paths: Sequence[str]) -> Dict[int, Dict[str, List[int]]]:
    """HF repo paths (any depth below layer_*/width_*/average_l0_*) -> {layer: {width: sorted L0 list}}."""
    inv: Dict[int, Dict[str, set]] = {}
    for p in paths:
        m = _L0_RE.search(p)
        if m:
            inv.setdefault(int(m.group(1)), {}).setdefault(m.group(2), set()).add(int(m.group(3)))
    order = lambda w: parse_width(w)
    return {L: {w: sorted(v) for w, v in sorted(ws.items(), key=lambda kv: order(kv[0]))} for L, ws in sorted(inv.items())}


def match_l0(target: float, variants: Sequence[int]) -> dict:
    """Closest L0 variant to `target`, plus the nearest variants below and above it (bracketing)."""
    v = sorted(int(x) for x in variants)
    if not v:
        raise ValueError("no L0 variants")
    closest = min(v, key=lambda x: (abs(np.log(x) - np.log(target)), x))
    below = [x for x in v if x <= target]
    above = [x for x in v if x >= target]
    return {"target": target, "closest": closest, "below": below[-1] if below else None,
            "above": above[0] if above else None, "rel_diff": abs(closest - target) / target}


def canonical_l0(variants: Sequence[int], target: float = 100.0) -> int:
    """Gemma Scope's 'canonical' SAE = the variant with average L0 closest to 100 (checked against every SAE the pilot
    resolved through sae_lens; tests/harness/test_sae.py)."""
    return min(variants, key=lambda x: (abs(x - target), x))


def params_path(sid: SAEId) -> Path:
    """Local path of params.npz at the pinned revision (downloads once; honours HF_HUB_OFFLINE)."""
    from huggingface_hub import hf_hub_download

    if not sid.revision:
        raise ValueError(f"{sid.key}: SAE revision must be pinned")
    return Path(hf_hub_download(sid.repo, sid.path_in_repo, revision=sid.revision))

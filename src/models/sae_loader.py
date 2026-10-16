"""Model / SAE loading: real (TransformerLens + SAELens) and a CPU-only mock world.

Common interface used by all experiments
    model.run_with_hooks(tokens, fwd_hooks=[(name, fn)]) -> logits [B, L, V]   (fn(act, hook) -> Tensor | None)
    model.cfg.d_model, model.cfg.n_layers, model.device, model.tokenizer
    sae.decompose(x) -> (x_hat, epsilon, feature_acts)   (float32, x_hat + epsilon == x up to 1 ulp)

Mock world design (see README): a reserved "dark" subspace holds the algorithmic state. Every SAE atom is
orthogonal to it, so for mod_sum / dyck_depth / graph_reach the state lands entirely in epsilon, while the
ioi_control signal is a dictionary atom and lands in x_hat.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field, fields
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.synthetic_tasks import NAMES, parse_state

log = logging.getLogger("prism")

HOOK_FMT = "blocks.{layer}.hook_resid_post"
N_DARK = 6          # 3 algorithmic tasks x 2-dim plane
N_IOI = len(NAMES)  # reserved dictionary atoms 0..7 carry the IOI signal
MOCK_D_SAE = {"16k": 1024, "32k": 2048, "65k": 4096, "131k": 8192, "262k": 16384, "524k": 32768, "1m": 65536}
_PLANES = {"mod_sum": (0, 1), "dyck_depth": (2, 3), "graph_reach": (4, 5)}


def hook_name(layer: int) -> str:
    return HOOK_FMT.format(layer=layer)


# =========================================================================== mock
@dataclass
class MockConfig:
    d_model: int = 256
    n_layers: int = 26
    d_sae: Dict[str, int] = field(default_factory=lambda: dict(MOCK_D_SAE))
    noise_std: float = 0.02
    n_true: int = 128        # orthonormal "true feature" atoms shared by all widths
    k: int = 16              # TopK sparsity of the mock SAE
    scale: float = 20.0      # readout gain
    ramp_center: float = 8.0  # layer at which the state is half-written
    ramp_width: float = 2.0
    seed: int = 1234

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MockConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def _phi(task: str, s: int) -> Tuple[float, float]:
    """Ground-truth state manifold: circle for mod_sum, monotone half-circle arc for the counters."""
    if task == "mod_sum":
        a = 2 * math.pi * s / 7
    else:
        a = (s - 1) / 5 * 0.8 * math.pi
    return math.cos(a), math.sin(a)


class MockWorld:
    """Shared random geometry for MockModel and MockSAE (deterministic in cfg.seed)."""

    def __init__(self, cfg: MockConfig, tokenizer) -> None:
        d = cfg.d_model
        assert d >= N_DARK + cfg.n_true, "d_model too small for the mock world"
        self.cfg, self.tok = cfg, tokenizer
        g = torch.Generator().manual_seed(cfg.seed)
        Q, _ = torch.linalg.qr(torch.randn(d, N_DARK + cfg.n_true, generator=g))
        self.D = Q[:, :N_DARK].contiguous()                  # [d, 6] dark basis
        self.T = Q[:, N_DARK:].T.contiguous()                # [n_true, d] orthonormal true atoms
        self.T_np = self.T.numpy()
        V = tokenizer.vocab_size
        # token embeddings = sparse combos of generic atoms (so the SAE reconstructs them well)
        coef = np.zeros((V, cfg.n_true), dtype=np.float32)
        for t in range(V):
            r = np.random.default_rng(cfg.seed * 7919 + t)
            coef[t, r.choice(np.arange(N_IOI, cfg.n_true), 3, replace=False)] = 0.6
        self.E_tok = torch.from_numpy(coef) @ self.T         # [V, d]
        self.noise_table = torch.randn(4096, d, generator=g) * cfg.noise_std
        # unembedding: answer-token columns read the task state planes / IOI atoms
        W_U = torch.zeros(d, V)
        for digit in range(10):
            tid = tokenizer.encode(f" {digit}", add_special_tokens=False)[0]
            for task, (i, j) in _PLANES.items():
                c1, c2 = _phi(task, digit)
                W_U[:, tid] += cfg.scale * (c1 * self.D[:, i] + c2 * self.D[:, j])
        for idx, name in enumerate(NAMES):
            tid = tokenizer.encode(f" {name}", add_special_tokens=False)[0]
            W_U[:, tid] += cfg.scale * self.T[idx]
        self.W_U = W_U
        ramp = 1 / (1 + np.exp(-(np.arange(cfg.n_layers) - cfg.ramp_center) / cfg.ramp_width))
        ramp[-1] = 1.0
        self.ramp = ramp.astype(np.float32)
        self._cache: Dict[bytes, Tuple[int, np.ndarray, np.ndarray]] = {}

    def prompt_info(self, row: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """(hash, context vector, state-signal vector) for one tokenised prompt (cached)."""
        key = row.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        h = int.from_bytes(hashlib.md5(key).digest()[:8], "little")
        r = np.random.default_rng(h % (2 ** 32))
        atoms = r.choice(np.arange(N_IOI, self.cfg.n_true), 6, replace=False)
        ctx = 0.7 * self.T_np[atoms].sum(0)
        try:
            task, s = parse_state(self.tok.decode(row.tolist()))
        except ValueError:          # generic (non-task) text carries no state signal
            task, s = None, None
        if task is None:
            sig = np.zeros(self.cfg.d_model)
        elif task == "ioi_control":
            sig = self.T_np[s].copy()
        else:
            i, j = _PLANES[task]
            c1, c2 = _phi(task, s)
            Dn = self.D.numpy()
            sig = c1 * Dn[:, i] + c2 * Dn[:, j]
        out = (h, ctx.astype(np.float32), sig.astype(np.float32))
        self._cache[key] = out
        return out


class MockHookedTransformer:
    """Position-wise toy 'transformer' with TransformerLens-like hook semantics.

    resid_l = token_embed + context(prompt hash) + noise + w_l * signal(state), where w_l ramps from ~0 to 1
    across layers. Patching resid_l therefore recovers a fraction ~w_l of the logit difference.
    """

    def __init__(self, world: MockWorld, device="cpu", dtype=torch.float32) -> None:
        self.world, self.device, self.dtype = world, torch.device(device), dtype
        self.tokenizer = world.tok
        self.cfg = SimpleNamespace(d_model=world.cfg.d_model, n_layers=world.cfg.n_layers, model_name="mock")
        self._W_U = world.W_U.to(self.device)
        self._E = world.E_tok.to(self.device)
        self._noise = world.noise_table.to(self.device)

    def eval(self) -> "MockHookedTransformer":
        return self

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.run_with_hooks(tokens, fwd_hooks=[])

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks: Optional[Sequence[Tuple[str, Callable]]] = None):
        hooks: Dict[str, List[Callable]] = {}
        for name, fn in fwd_hooks or []:
            hooks.setdefault(name, []).append(fn)
        B, L = tokens.shape
        rows = tokens.detach().cpu().numpy()
        infos = [self.world.prompt_info(r) for r in rows]
        ctx = torch.from_numpy(np.stack([i[1] for i in infos])).to(self.device)
        sig = torch.from_numpy(np.stack([i[2] for i in infos])).to(self.device)
        idx = torch.tensor([[(i[0] + 7919 * t) % 4096 for t in range(L)] for i in infos], device=self.device)
        x = (self._E[tokens.to(self.device)] + ctx[:, None, :] + self._noise[idx]).to(self.dtype)
        prev = 0.0
        for layer in range(self.cfg.n_layers):
            w = float(self.world.ramp[layer])
            x = x + ((w - prev) * sig[:, None, :]).to(self.dtype)
            prev = w
            for fn in hooks.get(hook_name(layer), []):
                out = fn(x, SimpleNamespace(name=hook_name(layer)))
                if out is not None:
                    x = out.to(self.dtype)
        return x.float() @ self._W_U


class MockSAE:
    """TopK SAE whose dictionary is orthogonal to the dark subspace (so it is blind to the algorithmic state)."""

    def __init__(self, world: MockWorld, width: str, layer: int, device="cpu") -> None:
        cfg = world.cfg
        n = cfg.d_sae.get(width) or MOCK_D_SAE[width]
        g = torch.Generator().manual_seed(cfg.seed + 17 * n)
        extra = torch.randn(n - cfg.n_true, cfg.d_model, generator=g)
        extra = extra - (extra @ world.D) @ world.D.T
        extra = extra / extra.norm(dim=-1, keepdim=True)
        self.W_dec = torch.cat([world.T, extra]).to(device)          # [n, d]
        self.W_enc = self.W_dec.T.contiguous()                       # tied
        self.b_enc = torch.zeros(n, device=device)
        self.b_dec = torch.zeros(cfg.d_model, device=device)
        self.k, self.width, self.layer, self.device = cfg.k, width, layer, torch.device(device)
        self.d_sae = n
        self.dtype = torch.float32

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(x @ self.W_enc + self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, vals)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    @torch.inference_mode()
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _decompose(self, x)


# =========================================================================== real
class RealSAE:
    """Thin wrapper around a SAELens SAE exposing decompose(); weights stay float32."""

    def __init__(self, sae: Any, width: str, layer: int) -> None:
        self.sae, self.width, self.layer = sae, width, layer
        self.device = next(sae.parameters()).device
        self.d_sae = getattr(getattr(sae, "cfg", None), "d_sae", None)
        self.dtype = next(sae.parameters()).dtype
        self.sae_id = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Gemma Scope JumpReLU computed from the weights directly (robust to SAELens dtype/version quirks):
        f(x) = ReLU(x W_enc + b_enc) * 1[x W_enc + b_enc > theta]."""
        s = self.sae
        if hasattr(s, "threshold") and hasattr(s, "W_enc"):
            x = x.to(s.W_enc.dtype)
            pre = x @ s.W_enc + s.b_enc
            return torch.relu(pre) * (pre > s.threshold)
        return s.encode(x)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        s = self.sae
        if hasattr(s, "W_dec"):
            return acts.to(s.W_dec.dtype) @ s.W_dec + s.b_dec
        return s.decode(acts)

    @torch.inference_mode()
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _decompose(self, x)


def _decompose(sae, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x -> (x_hat, eps, acts) in float32 with eps = x - x_hat (so x_hat + eps == x up to 1 ulp)."""
    x32 = x.to(device=sae.device, dtype=torch.float32)
    acts = sae.encode(x32.to(sae.dtype))
    x_hat = sae.decode(acts).to(torch.float32)       # bf16 SAEs (>= 524k): eps still computed in fp32
    return x_hat, x32 - x_hat, acts


# =========================================================================== factories
def load_model(mock: bool = False, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
               mock_cfg: Optional[Dict[str, Any]] = None, tokenizer=None, model_name: str = "gemma-2-2b"):
    from src.utils import get_device, get_dtype

    device = device or (torch.device("cpu") if mock else get_device())
    if mock:
        from src.data.synthetic_tasks import FallbackTokenizer

        cfg = MockConfig.from_dict(mock_cfg)
        tok = tokenizer if tokenizer is not None else FallbackTokenizer()
        return MockHookedTransformer(MockWorld(cfg, tok), device="cpu", dtype=torch.float32)
    from transformer_lens import HookedTransformer  # lazy: not needed in mock mode

    dtype = dtype or get_dtype(device)
    # no_processing: Gemma Scope SAEs were trained on the raw residual stream (no weight folding/centering).
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=str(device), dtype=dtype)
    return model.eval()


def parse_width(width: str) -> int:
    w = width.lower()
    if w.endswith("k"):
        return int(w[:-1]) * 1024
    if w.endswith("m"):
        return int(w[:-1]) * 1024 * 1024
    return int(w)


def resolve_sae_id(layer: int, width: str, release: str = "gemma-scope-2b-pt-res-canonical",
                   fallback_release: str = "gemma-scope-2b-pt-res", target_l0: float = 100.0) -> Tuple[str, str]:
    """(release, sae_id). Canonical id if SAELens lists it, else the fallback-release id whose average_l0
    is closest to `target_l0`. Raises KeyError if neither exists."""
    import re

    canonical = f"layer_{layer}/width_{width}/canonical"
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError:
        try:
            from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
        except ImportError:
            return release, canonical          # cannot check; let from_pretrained decide
    directory = get_pretrained_saes_directory()
    if release in directory and canonical in directory[release].saes_map:
        return release, canonical
    if fallback_release in directory:
        prefix = f"layer_{layer}/width_{width}/average_l0_"
        cands = [k for k in directory[fallback_release].saes_map if k.startswith(prefix)]
        if cands:
            best = min(cands, key=lambda k: abs(float(re.findall(r"[\d.]+$", k)[0]) - target_l0))
            return fallback_release, best
    raise KeyError(f"no Gemma Scope SAE for layer {layer} width {width}")


def load_sae(layer: int, width: str, model=None, mock: bool = False, device: Optional[torch.device] = None,
             release: str = "gemma-scope-2b-pt-res-canonical", bf16_min_width: str = "524k"):
    """Load ONE SAE (caller must delete it and call utils.free_memory() before loading the next).

    SAEs with >= bf16_min_width latents are held in bf16 (1m in fp32 is ~19 GB and would not fit next to the model).
    """
    if mock:
        assert isinstance(model, MockHookedTransformer), "mock SAE needs the mock model's world"
        return MockSAE(model.world, width, layer, device="cpu")
    from sae_lens import SAE  # lazy

    from src.utils import get_device

    device = device or get_device()
    rel, sae_id = resolve_sae_id(layer, width, release)
    big = parse_width(width) >= parse_width(bf16_min_width)
    load_dev = "cpu" if big else str(device)
    out = SAE.from_pretrained(release=rel, sae_id=sae_id, device=load_dev)
    sae = out[0] if isinstance(out, tuple) else out   # older SAELens returns (sae, cfg, sparsity)
    sae = sae.to(dtype=torch.bfloat16 if big else torch.float32).to(device).eval()
    w = RealSAE(sae, width, layer)
    w.sae_id = f"{rel}/{sae_id}"
    log.info("loaded SAE %s (%s, %s)", w.sae_id, w.dtype, device)
    return w

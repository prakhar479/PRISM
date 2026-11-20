"""Model / SAE loading: real (TransformerLens + Gemma Scope params.npz, SAELens fallback) and a CPU-only mock world.

Common interface used by all experiments
    model.run_with_hooks(tokens, fwd_hooks=[(name, fn)], start_at_layer=, stop_at_layer=, tokens=)
        TransformerLens semantics; experiments call it through src/models/runtime.py (last-position logits only)
    model.cfg.d_model, model.cfg.n_layers, model.cfg.d_vocab, model.tokenizer
    sae.decompose(x) -> (x_hat, epsilon, l0)   (float32, epsilon = x - x_hat; l0 = active latents per row)

Mock world design (see README): a reserved "dark" subspace holds the algorithmic state (one 2-d plane per
state-tracking task). Every SAE atom is orthogonal to it, so for add_chain / entity_track / var_binding (and the
legacy mod_sum / dyck_depth / graph_reach) the state lands entirely in epsilon, while the ioi_control signal is a
dictionary atom and lands in x_hat.
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

from src.data.synthetic_tasks import BOXES, NAMES, TASK_STATES, parse_state

log = logging.getLogger("prism")

HOOK_FMT = "blocks.{layer}.hook_resid_post"
_DARK_TASKS = ["add_chain", "entity_track", "var_binding", "mod_sum", "dyck_depth", "graph_reach"]
N_DARK = 2 * len(_DARK_TASKS)   # one 2-d state plane per algorithmic task
N_IOI = len(NAMES)  # reserved dictionary atoms 0..N_IOI-1 carry the IOI signal
MOCK_D_SAE = {"16k": 1024, "32k": 2048, "65k": 4096, "131k": 8192, "262k": 16384, "524k": 32768, "1m": 65536}
_PLANES = {t: (2 * i, 2 * i + 1) for i, t in enumerate(_DARK_TASKS)}
_ANSWER_TOKENS = {t: [f" {d}" for d in range(10)] for t in _DARK_TASKS}
_ANSWER_TOKENS["entity_track"] = [f" {b}" for b in BOXES]   # state = box index


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
    """Ground-truth state manifold: circle for mod_sum, monotone arc otherwise (>= 0.5 rad between neighbouring
    states so the readout separates them; the legacy 6-state counters keep their original 0.8*pi arc)."""
    if task == "mod_sum":
        a = 2 * math.pi * s / 7
    else:
        lo, hi = min(TASK_STATES[task]), max(TASK_STATES[task])
        a = (s - lo) * max(0.8 * math.pi / (hi - lo), 0.5)
    return math.cos(a), math.sin(a)


class MockWorld:
    """Shared random geometry for MockModel and MockSAE (deterministic in cfg.seed)."""

    def __init__(self, cfg: MockConfig, tokenizer) -> None:
        d = cfg.d_model
        assert d >= N_DARK + cfg.n_true, "d_model too small for the mock world"
        self.cfg, self.tok = cfg, tokenizer
        g = torch.Generator().manual_seed(cfg.seed)
        Q, _ = torch.linalg.qr(torch.randn(d, N_DARK + cfg.n_true, generator=g))
        self.D = Q[:, :N_DARK].contiguous()                  # [d, N_DARK] dark basis
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
        for task, (i, j) in _PLANES.items():
            for value, tok_str in enumerate(_ANSWER_TOKENS[task]):
                tid = tokenizer.encode(tok_str, add_special_tokens=False)[0]
                c1, c2 = _phi(task, value)
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
        self.cfg = SimpleNamespace(d_model=world.cfg.d_model, n_layers=world.cfg.n_layers, model_name="mock",
                                   d_vocab=world.W_U.shape[1], normalization_type=None, output_logits_soft_cap=0.0,
                                   device=str(self.device))
        self._W_U = world.W_U.to(self.device)
        self._E = world.E_tok.to(self.device)
        self._noise = world.noise_table.to(self.device)

    def eval(self) -> "MockHookedTransformer":
        return self

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.run_with_hooks(tokens, fwd_hooks=[])

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self._W_U

    def run_with_hooks(self, input: torch.Tensor, fwd_hooks: Optional[Sequence[Tuple[str, Callable]]] = None,
                       start_at_layer: Optional[int] = None, stop_at_layer: Optional[int] = None,
                       tokens: Optional[torch.Tensor] = None):
        """TransformerLens semantics: start_at_layer takes a residual [B, L, d] (tokens= still identifies the
        prompt); stop_at_layer returns the residual instead of logits."""
        hooks: Dict[str, List[Callable]] = {}
        for name, fn in fwd_hooks or []:
            hooks.setdefault(name, []).append(fn)
        start = start_at_layer or 0
        tokens = input if start_at_layer is None else tokens
        B, L = tokens.shape
        rows = tokens.detach().cpu().numpy()
        infos = [self.world.prompt_info(r) for r in rows]
        sig = torch.from_numpy(np.stack([i[2] for i in infos])).to(self.device)
        if start_at_layer is None:
            ctx = torch.from_numpy(np.stack([i[1] for i in infos])).to(self.device)
            idx = torch.tensor([[(i[0] + 7919 * t) % 4096 for t in range(L)] for i in infos], device=self.device)
            x = (self._E[tokens.to(self.device)] + ctx[:, None, :] + self._noise[idx]).to(self.dtype)
        else:
            x = input.to(self.device, self.dtype)
        prev = float(self.world.ramp[start - 1]) if start > 0 else 0.0
        stop = self.cfg.n_layers if stop_at_layer is None else stop_at_layer
        for layer in range(start, stop):
            w = float(self.world.ramp[layer])
            x = x + ((w - prev) * sig[:, None, :]).to(self.dtype)
            prev = w
            for fn in hooks.get(hook_name(layer), []):
                out = fn(x, SimpleNamespace(name=hook_name(layer)))
                if out is not None:
                    x = out.to(self.dtype)
        return x if stop_at_layer is not None else self.unembed(x)


class MockSAE:
    """TopK SAE whose dictionary is orthogonal to the dark subspace (so it is blind to the algorithmic state)."""

    row_chunk = 4096          # rows per decompose() call in decompose_batched
    encode_rows = 4096        # rows per encode() call

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
        self.streamed = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(x @ self.W_enc + self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, vals)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    @torch.inference_mode()
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x -> (x_hat, eps, l0) in float32 with eps = x - x_hat; l0 = active latents per row."""
        x32 = x.to(device=self.device, dtype=torch.float32)
        acts = self.encode(x32)
        x_hat = self.decode(acts)
        return x_hat, x32 - x_hat, (acts > 0).sum(-1)


# =========================================================================== real
SAE_CHUNK_BYTES = 256 * 2 ** 20     # size of one [rows, latents] float32 slab inside encode/decompose
STREAM_CHUNK_LATENTS = 16384        # latents per host->device weight chunk for streamed SAEs


class JumpReLUSAE:
    """Gemma Scope JumpReLU SAE held as plain tensors, latent-major:

        pre = x W_enc + b_enc,   f = ReLU(pre) * 1[pre > threshold],   x_hat = f W_dec + b_dec

    Weights are float32 (bf16 for very wide SAEs). Placement:
        resident  weights live on the compute device;
        streamed  weights stay in (pinned) CPU memory and latent chunks are streamed through the GPU with
                  copy/compute overlap, so any width runs next to the model on an 8 GB card.
    Both use the same chunked arithmetic (pre-activations in the weight dtype, decode accumulated in float32),
    so results do not depend on placement beyond float32 summation order.
    """

    def __init__(self, W_enc_T: torch.Tensor, b_enc: torch.Tensor, threshold: torch.Tensor, W_dec: torch.Tensor,
                 b_dec: torch.Tensor, width: str, layer: int, sae_id: Optional[str] = None) -> None:
        self.W_enc_T, self.b_enc, self.threshold, self.W_dec, self.b_dec = W_enc_T, b_enc, threshold, W_dec, b_dec
        self.width, self.layer, self.sae_id = width, layer, sae_id
        self.d_sae, self.d_model = W_dec.shape
        self.dtype = W_dec.dtype
        self.device = W_dec.device
        self.streamed = False
        self._host: List[Tuple[torch.Tensor, ...]] = []
        self.row_chunk = 8192
        self.encode_rows = max(1, SAE_CHUNK_BYTES // (self.d_sae * 4 * 4))   # encode() keeps ~4 [rows, d_sae] temporaries

    def nbytes(self) -> int:
        ts = [self.W_enc_T, self.b_enc, self.threshold, self.W_dec, self.b_dec] + [t for ch in self._host for t in ch]
        return sum(t.numel() * t.element_size() for t in ts if t is not None)

    def to_device(self, device) -> "JumpReLUSAE":
        device = torch.device(device)
        if self.streamed:
            raise RuntimeError("SAE is already streamed")
        for k in ("W_enc_T", "b_enc", "threshold", "W_dec", "b_dec"):
            setattr(self, k, getattr(self, k).to(device))
        self.device = device
        return self

    def stream_to(self, device, chunk_latents: int = STREAM_CHUNK_LATENTS, pin: bool = True) -> "JumpReLUSAE":
        """Keep weights on CPU and compute on `device` by streaming chunks of latents."""
        from src.models.runtime import pin as pin_fn

        cpu = [getattr(self, k).cpu() for k in ("W_enc_T", "b_enc", "threshold", "W_dec")]
        self._host = []
        for a in range(0, self.d_sae, chunk_latents):
            ch = tuple(t[a:a + chunk_latents] for t in cpu)
            self._host.append(tuple(pin_fn(t) for t in ch) if pin else ch)
        self.W_enc_T = self.b_enc = self.threshold = self.W_dec = None     # only the chunks hold the weights now
        self.b_dec = self.b_dec.to(device)
        self.device, self.streamed = torch.device(device), True
        return self

    def _chunks(self):
        """Yield (W_enc_T, b_enc, threshold, W_dec) latent chunks on the compute device."""
        if not self.streamed:
            n = self.d_sae if self.d_sae * self.d_model * 4 <= 4 * SAE_CHUNK_BYTES else STREAM_CHUNK_LATENTS
            for a in range(0, self.d_sae, n):
                yield self.W_enc_T[a:a + n], self.b_enc[a:a + n], self.threshold[a:a + n], self.W_dec[a:a + n]
            return
        dev = self.device
        if dev.type != "cuda":
            for ch in self._host:
                yield tuple(t.to(dev) for t in ch)
            return
        stream = torch.cuda.Stream(dev)

        def fetch(i):
            with torch.cuda.stream(stream):
                return tuple(t.to(dev, non_blocking=True) for t in self._host[i])

        nxt = fetch(0)
        for i in range(len(self._host)):
            torch.cuda.current_stream(dev).wait_stream(stream)
            cur = nxt
            for t in cur:
                t.record_stream(torch.cuda.current_stream(dev))
            nxt = fetch(i + 1) if i + 1 < len(self._host) else None     # overlaps with compute on chunk i
            yield cur

    @staticmethod
    def _act(xw: torch.Tensor, We: torch.Tensor, be: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        pre = xw @ We.T + be
        return torch.relu(pre) * (pre > th)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent activations [N, d_sae] (weight dtype, on the compute device). Keep N <= encode_rows."""
        xw = x.to(self.device, self.dtype)
        out = torch.empty(len(xw), self.d_sae, dtype=self.dtype, device=self.device)
        a = 0
        for We, be, th, _ in self._chunks():
            out[:, a:a + len(be)] = self._act(xw, We, be, th)
            a += len(be)
        return out

    @torch.inference_mode()
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        out = self.b_dec.float().expand(len(acts), -1).clone()
        a = 0
        for _, be, _, Wd in self._chunks():
            out += acts[:, a:a + len(be)].to(self.device).float() @ Wd.float()
            a += len(be)
        return out

    @torch.inference_mode()
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [..., d] -> (x_hat, eps, l0) in float32 on the compute device; eps = x - x_hat exactly."""
        shape = x.shape
        x32 = x.reshape(-1, shape[-1]).to(self.device, torch.float32)
        xw = x32.to(self.dtype)
        x_hat = torch.zeros_like(x32)
        l0 = torch.zeros(len(x32), dtype=torch.long, device=self.device)
        for We, be, th, Wd in self._chunks():
            rb = max(1, SAE_CHUNK_BYTES // (len(be) * 4 * 3))
            Wd32 = Wd.float()
            for r in range(0, len(x32), rb):
                f = self._act(xw[r:r + rb], We, be, th)
                l0[r:r + rb] += (f > 0).sum(-1)
                x_hat[r:r + rb] += f.float() @ Wd32
        x_hat += self.b_dec.float()
        return x_hat.reshape(shape), (x32 - x_hat).reshape(shape), l0.reshape(shape[:-1])


class RealSAE(JumpReLUSAE):
    """JumpReLUSAE built from a SAELens SAE module (fallback loader; same arithmetic as the npz path)."""

    def __init__(self, sae: Any, width: str, layer: int) -> None:
        if not all(hasattr(sae, k) for k in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")):
            raise TypeError("only JumpReLU SAEs (Gemma Scope) are supported")
        g = lambda t: t.detach()
        super().__init__(g(sae.W_enc).T.contiguous(), g(sae.b_enc), g(sae.threshold), g(sae.W_dec), g(sae.b_dec),
                         width, layer)


def _npz_member(path: str, name: str) -> np.ndarray:
    """Memory-map one array of an uncompressed .npz (Gemma Scope params.npz); full read otherwise."""
    import struct
    import zipfile

    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name + ".npy")
        stored = info.compress_type == zipfile.ZIP_STORED
    if stored:
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            n_name, n_extra = struct.unpack("<HH", f.read(30)[26:30])
            f.seek(info.header_offset + 30 + n_name + n_extra)
            major, _ = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(f)
            offset = f.tell()
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")
    with np.load(path) as z:
        return z[name]


def _to_torch(arr: np.ndarray, dtype: torch.dtype, transpose: bool = False, block: int = 4096) -> torch.Tensor:
    """Convert (a memmap of) arr to a torch tensor block by block, so peak RAM ~ the final tensor only."""
    shape = (arr.shape[1], arr.shape[0]) if transpose else tuple(arr.shape)
    out = torch.empty(shape, dtype=dtype)
    if arr.ndim == 1:
        out.copy_(torch.from_numpy(np.array(arr, dtype=np.float32)))
        return out
    for i in range(0, shape[0], block):
        src = arr[:, i:i + block].T if transpose else arr[i:i + block]
        out[i:i + block] = torch.from_numpy(np.array(src, dtype=np.float32)).to(dtype)
    return out


def load_gemma_scope_npz(path: str, width: str, layer: int, dtype: torch.dtype, sae_id: Optional[str] = None) -> JumpReLUSAE:
    """Gemma Scope params.npz (W_enc [d, n], W_dec [n, d], b_enc, b_dec, threshold) -> CPU JumpReLUSAE."""
    m = lambda k: _npz_member(path, k)
    return JumpReLUSAE(_to_torch(m("W_enc"), dtype, transpose=True), _to_torch(m("b_enc"), dtype),
                       _to_torch(m("threshold"), dtype), _to_torch(m("W_dec"), dtype), _to_torch(m("b_dec"), torch.float32),
                       width, layer, sae_id)


# =========================================================================== factories
def load_model(mock: bool = False, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
               mock_cfg: Optional[Dict[str, Any]] = None, tokenizer=None, model_name: str = "gemma-2-2b",
               runtime_cfg: Optional[Dict[str, Any]] = None):
    """Model on `device`. On GPUs < 16 GB (or runtime.offload_embed: true) the 256k-row embedding stays on CPU."""
    from src.models.runtime import want_offload_embed
    from src.utils import get_device, get_dtype

    device = device or (torch.device("cpu") if mock else get_device())
    if mock:
        from src.data.synthetic_tasks import FallbackTokenizer

        cfg = MockConfig.from_dict(mock_cfg)
        tok = tokenizer if tokenizer is not None else FallbackTokenizer()
        return MockHookedTransformer(MockWorld(cfg, tok), device="cpu", dtype=torch.float32)
    from transformer_lens import HookedTransformer  # lazy: not needed in mock mode

    dtype = dtype or get_dtype(device)
    offload = want_offload_embed({"runtime": runtime_cfg}, device)
    # no_processing: Gemma Scope SAEs were trained on the raw residual stream (no weight folding/centering).
    # Built on CPU first, then moved module by module, so the GPU never holds the embedding when offloaded.
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=str(device), dtype=dtype,
                                                            move_to_device=False)
    if offload and (model.cfg.positional_embedding_type != "rotary" or getattr(model.cfg, "post_embedding_ln", False)):
        offload = False   # the CPU-embedding path skips positional embeddings: only valid for rotary models (Gemma)
    for name, child in model.named_children():
        if not (offload and name in ("embed", "hook_embed")):
            child.to(device)
    model.cfg.device = str(device)            # TL moves the residual to cfg.device before every block
    model._prism_embed_on_cpu = offload
    if device.type == "cuda":
        log.info("model loaded: %.2f GiB VRAM in use (embedding on %s)", torch.cuda.memory_allocated() / 2 ** 30,
                 "CPU" if offload else "GPU")
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


def sae_bytes(width: str, d_model: int = 2304, bf16_min_width: str = "524k") -> int:
    n = parse_width(width)
    elt = 2 if n >= parse_width(bf16_min_width) else 4
    return (2 * d_model * n + 2 * n) * elt + 4 * d_model


def load_sae(layer: int, width: str, model=None, mock: bool = False, device: Optional[torch.device] = None,
             release: str = "gemma-scope-2b-pt-res-canonical", bf16_min_width: str = "524k", place: bool = True,
             placement: str = "auto"):
    """Load ONE SAE (caller deletes it and calls utils.free_memory() before loading the next).

    SAEs with >= bf16_min_width latents are held in bf16 (1m in fp32 is ~19 GB). With place=False the SAE is
    returned on CPU (used by the background prefetcher); place_sae() then puts it on the GPU or streams it.
    """
    if mock:
        assert isinstance(model, MockHookedTransformer), "mock SAE needs the mock model's world"
        return MockSAE(model.world, width, layer, device="cpu")
    from src.models.runtime import GiB, available_ram
    from src.utils import get_device

    device = device or get_device()
    rel, sae_id = resolve_sae_id(layer, width, release)
    dtype = torch.bfloat16 if parse_width(width) >= parse_width(bf16_min_width) else torch.float32
    need, avail = sae_bytes(width, bf16_min_width=bf16_min_width), available_ram()
    if need * 1.15 > avail:
        raise MemoryError(f"SAE L{layer} {width} needs {need / GiB:.1f} GiB RAM, only {avail / GiB:.1f} GiB available")
    try:
        sae = load_gemma_scope_npz(_sae_params_path(rel, sae_id), width, layer, dtype, f"{rel}/{sae_id}")
    except Exception as e:  # noqa: BLE001 - unusual layout: fall back to SAELens (needs ~2x RAM while loading)
        log.warning("direct params.npz load failed for %s/%s (%s); using SAELens", rel, sae_id, e)
        from sae_lens import SAE

        out = SAE.from_pretrained(release=rel, sae_id=sae_id, device="cpu")
        mod = out[0] if isinstance(out, tuple) else out   # older SAELens returns (sae, cfg, sparsity)
        sae = RealSAE(mod.to(dtype=dtype).eval(), width, layer)
        sae.sae_id = f"{rel}/{sae_id}"
    return place_sae(sae, device, placement) if place else sae


def _sae_params_path(release: str, sae_id: str) -> str:
    from huggingface_hub import hf_hub_download

    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError:
        from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    info = get_pretrained_saes_directory()[release]
    return hf_hub_download(info.repo_id, f"{info.saes_map[sae_id]}/params.npz")


def place_sae(sae, device, placement: str = "auto"):
    """GPU-resident if it fits next to the activation reserve, else streamed from CPU ('gpu'/'stream' force it)."""
    from src.models.runtime import GiB, act_reserve_bytes, available_ram, free_vram

    device = torch.device(device)
    if isinstance(sae, MockSAE):
        return sae
    if device.type != "cuda":
        return sae.stream_to(device, pin=False) if placement == "stream" else sae.to_device(device)
    need, free = sae.nbytes(), free_vram()
    resident = placement == "gpu" or (placement == "auto" and need + act_reserve_bytes() <= free)
    if resident:
        sae.to_device(device)
    else:
        sae.stream_to(device, pin=need < 0.25 * available_ram())
    log.info("SAE %s: %s (%.2f GiB %s, %.2f GiB VRAM free)", sae.sae_id, "GPU-resident" if resident else "streamed from CPU",
             need / GiB, str(sae.dtype).replace("torch.", ""), free / GiB)
    return sae

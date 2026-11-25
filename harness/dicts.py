"""Dictionaries beyond Gemma Scope's JumpReLU SAEs (breadth pilot, docs/PILOT_BREADTH_PLAN.md: P2, P4, P6, P8).

Every dictionary is used through harness.sae.decompose (stage 2, fidelity, spans), so all experiments see one arithmetic:
    x_hat, eps = x - x_hat, l0
Kinds (the container is the pilot's latent-major JumpReLUSAE: W_enc_T [n, d], b_enc [n], threshold [n], W_dec [n, d], b_dec [d]):
    jumprelu   z = x W_enc + b_enc, f = z * 1[z > threshold]    (Gemma Scope / SAEBench JumpReLU; ReLU SAEs are threshold 0;
                                                                 BatchTopK at inference is one global threshold)
    topk       f = TopK_k(ReLU(z)) per row                          (SAEBench TopK; k latents per token)
    gated      z = x W_gate; f = 1[z + b_gate > 0] * ReLU(mag_scale * z + b_mag)
Any pre-encoder subtraction of b_dec (dictionary_learning: encoder(x - b_dec)) is folded into the encoder bias at load time,
so b_enc here always multiplies raw x.
Wrappers:
    AddonDict  x_hat' = x_hat + A(x, eps), eps' = eps - A    (P4/P6): 'lin' A = [x, 1] W (ridge of eps on x: the linearly
               predictable error, Engels et al.), 'pcx<r>' A = eps P_r with P_r the top-r eigenvectors of the uncentered second
               moment of x (the dense low-rank 'scaffold' of Deng et al.; pcx1 = the massive-activation direction alone).
    Batch 2 (docs/NEXT_BATCH_PLAN.md, H1 / H2): the correction is a k-dimensional piece of the error.
      var<k> / fis<k> / sens<k> / rnd<k>   A = [x, 1] W P_U, the ridge predictor of eps from x restricted to the k-dim subspace U
               chosen by error variance (top eigenvectors of Cov eps), by output sensitivity alone (top eigenvectors of the
               generic-text Fisher F of harness/sensitivity.py), by both (top eigenvectors of sym(Cov(eps) F): the directions
               where unrepaired error costs most), or at random;
      ovar / ofis / osens / ornd <k>       the same subspaces as ORACLE projections A = eps P_U (the true error, no predictor):
               an upper bound that separates 'wrong subspace' from 'badly predicted';
      clin     A = [f, 1] W2: the decoder refit by ridge on the frozen codes f (is the missing part linear in the codes?);
      mlp<h> / cmlp<h>   A = a one-hidden-layer GELU network of x (mlp) or of the codes (cmlp): the nonlinear references.
Identifiers: DictId extends SAEId. Gemma Scope dictionaries keep SAEId's key and to_dict() exactly, so cached stage-2 outputs
of earlier runs stay valid; other sources are named in configs/dict_registry.yaml or trained by scripts/train_dictionary.py.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from harness.sae import SAE_REPO, SAEId, SparseCodes, parse_width

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs" / "dict_registry.yaml"
ADDON_LAMBDA = 1e-2            # ridge strength for 'lin', relative to the mean diagonal of X^T X
ADDON_MAX_ROWS = 262_144       # generic rows an add-on is fitted on (the first shards of the generic set)
KINDS = ("jumprelu", "topk", "gated")


# --------------------------------------------------------------------------- add-on names
_ADDON_RE = re.compile(r"^(?:(lin)|(clin)|(mlp|cmlp)(\d+)|(o?)(var|fis|sens|rnd|pcx)(\d+))$")
SUBSPACE_KINDS = ("var", "fis", "sens", "rnd")


def parse_addon(mode: str) -> dict:
    """'lin' | 'clin' | 'mlp<h>' | 'cmlp<h>' | '[o](var|fis|sens|rnd)<k>' | 'pcx<r>' -> {kind, k, oracle, needs_F, needs_codes}.
    pcx is an oracle projection (eps P_r) by construction; k >= 1."""
    m = _ADDON_RE.match(mode or "")
    if not m or (m.group(7) is not None and int(m.group(7)) < 1) or (m.group(4) is not None and int(m.group(4)) < 1):
        raise ValueError(f"unknown addon {mode!r} (lin | clin | mlp<h> | cmlp<h> | [o]var|fis|sens|rnd<k> | pcx<r>)")
    if m.group(1):
        kind, k, oracle = "lin", 0, False
    elif m.group(2):
        kind, k, oracle = "clin", 0, False
    elif m.group(3):
        kind, k, oracle = m.group(3), int(m.group(4)), False
    else:
        kind, k = m.group(6), int(m.group(7))
        oracle = bool(m.group(5)) or kind == "pcx"
    return {"kind": kind, "k": k, "oracle": oracle, "needs_F": kind in ("fis", "sens"), "needs_codes": kind in ("clin", "cmlp")}


# --------------------------------------------------------------------------- identifiers
@dataclass(frozen=True)
class DictId(SAEId):
    source: str = "gemma_scope"            # gemma_scope | registry | trained
    name: str = ""                         # registry / trained dictionary name
    arch: str = "jumprelu"                 # loader format (registry): see load_registry_dict
    files: Tuple[str, ...] = ()            # files in `repo` (registry)
    addon: str = ""                        # '' | 'lin' | 'pcx<r>'

    @property
    def base_key(self) -> str:
        return SAEId.key.fget(self) if self.source == "gemma_scope" else f"L{self.layer}_{self.name}"

    @property
    def key(self) -> str:
        return self.base_key + (f"__{self.addon}" if self.addon else "")

    @property
    def path_in_repo(self) -> str:
        return SAEId.path_in_repo.fget(self) if self.source == "gemma_scope" else (self.files[0] if self.files else "")

    def repo_files(self) -> Tuple[Tuple[str, str, Optional[str]], ...]:
        """(repo, file, revision) to download; trained dictionaries have none (they live in the cache)."""
        if self.source == "gemma_scope":
            return ((self.repo, self.path_in_repo, self.revision),)
        if self.source == "registry":
            return tuple((self.repo, f, self.revision) for f in self.files)
        return ()

    def base(self) -> "DictId":
        return dataclasses.replace(self, addon="")

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.source == "gemma_scope" and not self.addon:
            return {k: d[k] for k in ("layer", "width", "l0", "repo", "revision")}     # = SAEId.to_dict(): old cache keys
        d["files"] = list(self.files)
        return d


def load_registry(path: Optional[Path] = None) -> Dict[str, dict]:
    """configs/dict_registry.yaml (HL_DICT_REGISTRY overrides the path; tests use it)."""
    import os
    import yaml
    path = Path(path or os.environ.get("HL_DICT_REGISTRY") or REGISTRY)
    if not Path(path).exists():
        return {}
    d = yaml.safe_load(Path(path).read_text()) or {}
    return dict(d.get("dictionaries") or {})


def resolve(layer: int, spec, inventory: dict, sae_revision: Optional[str], registry: Optional[dict] = None) -> DictId:
    """A cell's SAESpec -> DictId. spec.dict names a registry or trained dictionary; otherwise Gemma Scope (width, l0 rule)."""
    name = getattr(spec, "dict", None)
    addon = getattr(spec, "addon", None) or ""
    if addon:
        parse_addon(addon)
    if not name:
        did = DictId(layer, spec.width, spec.resolve(layer, inventory), revision=sae_revision)
    elif name.startswith("trained:"):
        nm = name.split(":", 1)[1]
        did = DictId(layer, spec.width, int(spec.l0 or 0), repo="", revision=None, source="trained", name=nm, arch="npz")
    else:
        reg = registry if registry is not None else load_registry()
        if name not in reg:
            raise ValueError(f"dictionary {name!r} is not in {REGISTRY.name}")
        e = reg[name]
        if int(e["layer"]) != int(layer):
            raise ValueError(f"dictionary {name} is for layer {e['layer']}, cell is at layer {layer}")
        did = DictId(layer, str(e["width"]), int(e.get("l0") or 0), repo=e["repo"], revision=e["revision"], source="registry",
                     name=name, arch=e["arch"], files=tuple(e["files"]))
    return dataclasses.replace(did, addon=addon) if addon else did


# --------------------------------------------------------------------------- containers
def _jumprelu_cls():
    from src.models.sae_loader import JumpReLUSAE
    return JumpReLUSAE


class Dictionary(_jumprelu_cls()):
    """JumpReLUSAE plus a kind. kind 'jumprelu' uses harness.sae's own arithmetic (no custom hooks); 'topk' and 'gated' carry
    decompose_rows / encode_active."""

    def __init__(self, W_enc_T, b_enc, threshold, W_dec, b_dec, width, layer, sae_id=None, kind: str = "jumprelu",
                 k: Optional[int] = None, mag_scale: Optional[torch.Tensor] = None, b_mag: Optional[torch.Tensor] = None):
        super().__init__(W_enc_T, b_enc, threshold, W_dec, b_dec, width, layer, sae_id)
        if kind not in KINDS:
            raise ValueError(f"kind {kind}")
        self.kind, self.k, self.mag_scale, self.b_mag = kind, k, mag_scale, b_mag
        if kind == "topk" and not k:
            raise ValueError("topk needs k")
        if kind == "gated" and (mag_scale is None or b_mag is None):
            raise ValueError("gated needs mag_scale and b_mag")
        if kind != "jumprelu":
            self.decompose_rows = self._decompose_rows
            self.encode_active = self._encode_active

    def to_device(self, device):
        super().to_device(device)
        for a in ("mag_scale", "b_mag"):
            if getattr(self, a) is not None:
                setattr(self, a, getattr(self, a).to(device))
        return self

    # ---- arithmetic for topk / gated (pure, chunked over rows; latents in blocks for gated)
    def _on(self, device) -> dict:
        """The weights on `device`, moved once per decompose / encode call (not once per row chunk)."""
        w = {"We": self.W_enc_T.to(device), "be": self.b_enc.to(device), "Wd": self.W_dec.to(device)}
        if self.kind == "gated":
            w.update(ms=self.mag_scale.to(device), bm=self.b_mag.to(device))
        return w

    def _codes(self, xw: torch.Tensor, w: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """(values [R, m], latent ids [R, m]) of the active latents of rows xw (m = k for topk; padded with 0 for gated)."""
        z = xw @ w["We"].T
        if self.kind == "topk":
            post = torch.relu(z + w["be"])
            v, i = torch.topk(post, min(int(self.k), post.shape[-1]), dim=-1)
            return v, i
        f = (z + w["be"] > 0) * torch.relu(w["ms"] * z + w["bm"])
        m = int((f != 0).sum(-1).max().item()) if len(f) else 0
        v, i = torch.topk(f, max(1, m), dim=-1)
        return v, i

    def _row_step(self, rows: int) -> int:
        """Rows per chunk: ~256 MiB per [rows, n_latents] float32 buffer, four of them alive at once (pre-activations,
        post / gated codes, the top-k copy, the dense code matrix of the reconstruction)."""
        n = self.W_enc_T.shape[0]
        return max(1, min(rows, (256 * 2 ** 20) // max(1, n * 4 * 4)))

    @torch.inference_mode()
    def _decompose_rows(self, X: torch.Tensor, block_latents=None, device=None):
        device = torch.device(device) if device is not None else self.W_enc_T.device
        x32 = X.reshape(-1, X.shape[-1]).to(device, torch.float32)
        xw = x32.to(self.W_enc_T.dtype)
        w = self._on(device)
        Wd = w["Wd"].float()
        x_hat = torch.zeros_like(x32)
        l0 = torch.zeros(len(x32), dtype=torch.long, device=device)
        rb = self._row_step(len(x32))
        for r in range(0, len(x32), rb):
            v, i = self._codes(xw[r:r + rb], w)
            v = v.float()
            l0[r:r + rb] = (v > 0).sum(-1)
            # dense codes [rows, n] @ W_dec: memory is rows x n (budgeted by _row_step). Gathering W_dec[i] instead is
            # rows x m x d_model, which for a high-L0 gated SAE (m = the chunk's max L0) asked for 14 GiB on an 11 GB card
            f = torch.zeros(len(v), Wd.shape[0], device=device, dtype=torch.float32).scatter_(1, i, v)
            x_hat[r:r + rb] = f @ Wd
            del f
        x_hat += self.b_dec.to(device).float()
        shape = X.shape
        return x_hat.reshape(shape), (x32 - x_hat).reshape(shape), l0.reshape(shape[:-1])

    @torch.inference_mode()
    def _encode_active(self, X: torch.Tensor, block_latents=None, device=None) -> SparseCodes:
        device = torch.device(device) if device is not None else self.W_enc_T.device
        xw = X.reshape(-1, X.shape[-1]).to(device, self.W_enc_T.dtype)
        rows, cols, zs = [], [], []
        w = self._on(device)
        rb = self._row_step(len(xw))
        for r in range(0, len(xw), rb):
            v, i = self._codes(xw[r:r + rb], w)
            keep = v > 0
            rr = torch.arange(r, r + len(v), device=device)[:, None].expand_as(v)
            rows.append(rr[keep].cpu())
            cols.append(i[keep].cpu())
            zs.append(v[keep].float().cpu())
        cat = lambda xs, dt: torch.cat(xs) if xs else torch.zeros(0, dtype=dt)
        z = cat(zs, torch.float32)
        return SparseCodes(cat(rows, torch.long), cat(cols, torch.long), z, torch.zeros_like(z), len(xw), 1.0)


class AddonDict:
    """base dictionary + a correction fitted on generic text (see module doc). Codes are the base's."""

    def __init__(self, base, mode: str, W: torch.Tensor, c: Optional[torch.Tensor] = None, info: Optional[dict] = None,
                 extra: Optional[Dict[str, torch.Tensor]] = None):
        self.base, self.mode, self.W, self.c, self.info = base, mode, W.float(), (None if c is None else c.float()), info or {}
        self.extra = {k: v.float() for k, v in (extra or {}).items()}
        self.spec = parse_addon(mode)
        self.width, self.layer = getattr(base, "width", "?"), getattr(base, "layer", -1)
        self.streamed = False
        if self.spec["needs_codes"] and (not hasattr(base, "W_enc_T") or getattr(base, "kind", "jumprelu") != "jumprelu"):
            raise ValueError(f"add-on {mode} needs a JumpReLU base (it reads the codes z * 1[z > threshold])")
        if getattr(base, "encode_active", None) is not None:      # TopK / gated base: its own active codes; a JumpReLU base
            self.encode_active = base.encode_active               # goes through encode_sparse (alpha_min kept, z > thr filter)

    @property
    def b_dec(self):
        return self.base.b_dec

    @property
    def b_enc(self):
        return self.base.b_enc

    @property
    def threshold(self):
        return self.base.threshold

    @property
    def W_dec(self):
        return self.base.W_dec

    @property
    def W_enc_T(self):
        return self.base.W_enc_T

    def to_device(self, device):
        self.base.to_device(device)
        self.W = self.W.to(device)
        if self.c is not None:
            self.c = self.c.to(device)
        self.extra = {k: v.to(device) for k, v in self.extra.items()}
        return self

    def codes(self, x32: torch.Tensor) -> torch.Tensor:
        """Dense JumpReLU codes f = z * 1[z > threshold] of rows x32 [R, d] (the arithmetic of harness.sae.decompose)."""
        b = self.base
        We, be, th = b.W_enc_T.to(x32.device), b.b_enc.to(x32.device), b.threshold.to(x32.device)
        z = x32.to(We.dtype) @ We.T + be
        return (z * (z > th)).float()

    def _rows_step(self) -> int:
        n = int(self.base.W_enc_T.shape[0]) if self.spec["needs_codes"] else int(self.W.shape[0])
        return max(1, (256 * 2 ** 20) // max(1, n * 4 * 3))

    def addon(self, x32: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sp, W = self.spec, self.W.to(x32.device)
        if sp["oracle"]:
            return eps @ W                                # pcx / o<kind><k>: W = U U^T applied to the TRUE error
        if sp["kind"] in ("lin",) + SUBSPACE_KINDS:
            return x32 @ W + self.c.to(x32.device)        # lin: W d x d; var / fis / sens / rnd: W = W_lin U U^T (predictor)
        out = torch.empty_like(x32)
        step = self._rows_step()
        for a in range(0, len(x32), step):
            xs = x32[a:a + step]
            f = self.codes(xs) if sp["needs_codes"] else xs
            if sp["kind"] == "clin":
                out[a:a + step] = f @ W + self.c.to(x32.device)
            else:                                          # mlp / cmlp
                e = {k: v.to(x32.device) for k, v in self.extra.items()}
                h = torch.nn.functional.gelu((f / e["scale_in"]) @ W + e["b1"])
                out[a:a + step] = (h @ e["W2"] + e["b2"]) * e["scale_out"]
        return out

    @torch.inference_mode()
    def decompose_rows(self, X: torch.Tensor, block_latents=None, device=None):
        from harness.sae import decompose
        xh, eps, l0 = decompose(self.base, X, block_latents=block_latents, device=device)
        x32 = X.reshape(-1, X.shape[-1]).to(xh.device, torch.float32)
        e2 = eps.reshape(-1, eps.shape[-1])
        a = self.addon(x32, e2).reshape(eps.shape)
        return xh + a, eps - a, l0



# --------------------------------------------------------------------------- add-on fitting (P4 / P6; batch 2)
def _moments(base, chunks, device, need_ee: bool):
    """Streaming float64 moments of (x, eps) over generic residual chunks: XtX, XtE, [EtE], sum x, sum eps, n."""
    from harness.sae import decompose
    XtX = XtE = EtE = sx = se = None
    n = 0
    for X in chunks:
        x = X.reshape(-1, X.shape[-1]).float()
        _, eps, _ = decompose(base, x, device=device)
        x64, e64 = x.to(eps.device).double(), eps.double()
        XtX = x64.T @ x64 if XtX is None else XtX + x64.T @ x64
        XtE = x64.T @ e64 if XtE is None else XtE + x64.T @ e64
        if need_ee:
            EtE = e64.T @ e64 if EtE is None else EtE + e64.T @ e64
        sx = x64.sum(0) if sx is None else sx + x64.sum(0)
        se = e64.sum(0) if se is None else se + e64.sum(0)
        n += len(x)
    if n == 0:
        raise ValueError("fit_addon: no rows")
    return XtX, XtE, EtE, sx, se, n


def _top_eigvecs(M: torch.Tensor, k: int) -> torch.Tensor:
    ev, V = torch.linalg.eigh(M.cpu().double())
    return V[:, torch.argsort(ev, descending=True)[:k]]


def error_subspace(kind: str, k: int, Cee: torch.Tensor, F: Optional[torch.Tensor], seed: int = 0) -> torch.Tensor:
    """U [d, k] orthonormal: var = top eigenvectors of Cov(eps); fis = of the Fisher F; sens = of sym(Cov(eps) F); rnd = Haar."""
    d = Cee.shape[0]
    k = min(k, d)
    if kind == "var":
        return _top_eigvecs(Cee, k)
    if kind == "fis":
        return _top_eigvecs(F, k)
    if kind == "sens":
        S = Cee.cpu().double() @ F.cpu().double()
        return _top_eigvecs(0.5 * (S + S.T), k)
    g = torch.Generator().manual_seed(int(seed))
    Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    return Q


@torch.inference_mode()
def fit_addon(base, chunks: Iterable[torch.Tensor], mode: str, lam: float = ADDON_LAMBDA, device=None,
              F: Optional[torch.Tensor] = None, seed: int = 0) -> dict:
    """Fit the add-on map of `mode` from generic residual chunks [N, d] (one streaming pass, float64 sums).
    lin: W, c minimise sum |eps - x W - c|^2 + lam * mean(diag X^T X) * |W|^2 (bias not penalised).
    pcx<r>: W = U U^T with U the top-r eigenvectors of sum x x^T (uncentered).
    var / fis / sens / rnd <k>: U = error_subspace(...); W = W_lin U U^T, c = c_lin U U^T (the predictor of the k coefficients is
        the corresponding columns of the ridge solution: least squares decomposes per output direction); o<kind><k>: W = U U^T.
        fis / sens need F (the generic Fisher of harness.sensitivity); info records the share of Cov(eps) and of F in U and the
        predictor's R^2 on the generic fit rows.
    clin: W2, c2 = ridge of eps on the dense JumpReLU codes (the decoder refit on frozen codes).
    mlp<h> / cmlp<h>: one-hidden-layer GELU network eps ~ g(x) / g(codes), Adam, 10 % of rows held out for the reported R^2."""
    sp = parse_addon(mode)
    kind, k = sp["kind"], sp["k"]
    if sp["needs_F"] and F is None:
        raise ValueError(f"add-on {mode} needs the generic-text Fisher (scripts/fit_sensitivity.py) for layer {getattr(base, 'layer', '?')}")
    if kind == "pcx":
        XtX = None
        n = 0
        for X in chunks:
            x64 = X.reshape(-1, X.shape[-1]).double()
            XtX = x64.T @ x64 if XtX is None else XtX + x64.T @ x64
            n += len(x64)
        if n == 0:
            raise ValueError("fit_addon: no rows")
        ev, V = torch.linalg.eigh((XtX / n).cpu())
        U = V[:, torch.argsort(ev, descending=True)[:k]]
        share = float(ev.sort(descending=True).values[:k].sum() / ev.sum())
        return {"W": (U @ U.T).float(), "c": None, "info": {"mode": mode, "n_rows": n, "rank": k, "second_moment_share": share}}
    if kind in ("clin", "mlp", "cmlp"):
        return _fit_codes_or_mlp(base, chunks, mode, sp, lam, device, seed)
    # ---- linear-like: lin and the k-dimensional subspaces
    XtX, XtE, EtE, sx, se, n = _moments(base, chunks, device, need_ee=kind in SUBSPACE_KINDS)
    mx, me = sx / n, se / n
    C = XtX / n - torch.outer(mx, mx)                     # centered: the bias absorbs the means
    Ce = XtE / n - torch.outer(mx, me)
    reg = lam * float(torch.diagonal(C).mean())
    info = {"mode": mode, "n_rows": n, "lam": lam, "reg": reg}
    if not sp["oracle"] or kind == "lin":
        W = torch.linalg.solve(C + reg * torch.eye(C.shape[0], dtype=C.dtype, device=C.device), Ce)
        c = me - mx @ W
    if kind == "lin":
        return {"W": W.float().cpu(), "c": c.float().cpu(), "info": info}
    Cee = EtE / n - torch.outer(me, me)
    U = error_subspace(kind, k, Cee, F, seed=seed)
    P = (U @ U.T).cpu().double()
    Cee_c, tr = Cee.cpu().double(), float(torch.trace(Cee))
    info.update({"kind": kind, "k": int(U.shape[1]), "oracle": sp["oracle"], "eps_var_share": float(torch.trace(P @ Cee_c)) / tr})
    if F is not None:
        info["fisher_share"] = float(torch.trace(P @ F.cpu().double())) / float(torch.trace(F))
        info["cost_share"] = float(torch.trace(P @ (Cee_c @ F.cpu().double()))) / float(torch.trace(Cee_c @ F.cpu().double()))
    if sp["oracle"]:
        return {"W": P.float(), "c": None, "info": info}
    Wc, Cxe = W.cpu().double(), Ce.cpu().double()
    Cov_r = Cee_c - Wc.T @ Cxe - Cxe.T @ Wc + Wc.T @ C.cpu().double() @ Wc       # Cov(eps - x W)
    info["r2_predictor"] = 1.0 - float(torch.trace(P @ Cov_r)) / float(torch.trace(P @ Cee_c))
    return {"W": (W.cpu().double() @ P).float(), "c": (c.cpu().double() @ P).float(), "info": info}


@torch.inference_mode(False)                        # the networks train with autograd (fit_addon itself runs in inference mode)
def _fit_codes_or_mlp(base, chunks, mode: str, sp: dict, lam: float, device, seed: int) -> dict:
    """clin (ridge on codes) and mlp / cmlp (network on x / codes). The base must be JumpReLU for the codes-based ones."""
    from harness.sae import decompose
    dev = torch.device(device) if device is not None else base.W_enc_T.device
    tmp = AddonDict(base, mode, torch.zeros(1, 1))                # only for .codes()
    tmp.to_device(dev) if hasattr(base, "to_device") and sp["needs_codes"] else None
    kind = sp["kind"]
    if kind == "clin":
        n = int(base.W_enc_T.shape[0])
        FtF = FtE = sf = se = None
        rows = 0
        for X in chunks:
            x = X.reshape(-1, X.shape[-1]).float().to(dev)
            _, eps, _ = decompose(base, x, device=dev)
            step = max(1, (256 * 2 ** 20) // max(1, n * 4 * 3))
            for a in range(0, len(x), step):
                f = tmp.codes(x[a:a + step])
                e = eps[a:a + step]
                ftf, fte = (f.T @ f).double(), (f.T @ e).double()
                FtF = ftf if FtF is None else FtF + ftf
                FtE = fte if FtE is None else FtE + fte
                sf = f.double().sum(0) if sf is None else sf + f.double().sum(0)
                se = e.double().sum(0) if se is None else se + e.double().sum(0)
            rows += len(x)
        mf, me = sf / rows, se / rows
        C = FtF / rows - torch.outer(mf, mf)
        Ce = FtE / rows - torch.outer(mf, me)
        reg = lam * float(torch.diagonal(C).mean())
        C.diagonal().add_(reg)
        L = torch.linalg.cholesky(C)
        W2 = torch.cholesky_solve(Ce, L)
        c2 = me - mf @ W2
        return {"W": W2.float().cpu(), "c": c2.float().cpu(), "info": {"mode": mode, "n_rows": rows, "lam": lam, "reg": reg, "n_codes": n}}
    # mlp / cmlp: materialise (x, eps) on the CPU, train on the device
    xs, es = [], []
    for X in chunks:
        x = X.reshape(-1, X.shape[-1]).float()
        _, eps, _ = decompose(base, x.to(dev), device=dev)
        xs.append(x.cpu())
        es.append(eps.cpu())
    x, e = torch.cat(xs).clone(), torch.cat(es).clone()          # plain tensors: inference tensors cannot be saved for backward
    N, d = x.shape
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(N, generator=g)
    n_val = max(1, N // 10)
    va, tr = perm[:n_val], perm[n_val:]
    h = int(sp["k"])
    n_in = int(base.W_enc_T.shape[0]) if sp["needs_codes"] else d
    with torch.no_grad():
        sample = x[tr[:2048]].to(dev)
        s_in = float((tmp.codes(sample) if sp["needs_codes"] else sample).pow(2).mean().sqrt()) + 1e-6
    s_out = float(e[tr[:20000]].pow(2).mean().sqrt()) + 1e-6
    torch.manual_seed(int(seed))
    W1 = (torch.randn(n_in, h, device=dev) / n_in ** 0.5).requires_grad_(True)
    b1 = torch.zeros(h, device=dev, requires_grad=True)
    W2 = (torch.randn(h, d, device=dev) / h ** 0.5 * 0.1).requires_grad_(True)
    b2 = e[tr].mean(0).to(dev).div(s_out).clone().requires_grad_(True)
    params = [W1, b1, W2, b2]
    opt = torch.optim.Adam(params, lr=1e-3)
    epochs, bs = 6, 2048
    steps = epochs * max(1, (len(tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def fwd(xb):
        f = tmp.codes(xb) if sp["needs_codes"] else xb
        return torch.nn.functional.gelu((f / s_in) @ W1 + b1) @ W2 + b2

    with torch.enable_grad():
        for ep in range(epochs):
            order = tr[torch.randperm(len(tr), generator=g)]
            for a in range(0, len(order), bs):
                idx = order[a:a + bs]
                loss = (fwd(x[idx].to(dev)) - e[idx].to(dev) / s_out).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
    with torch.no_grad():
        num = den = 0.0
        for a in range(0, len(va), 4096):
            idx = va[a:a + 4096]
            ev_ = e[idx].to(dev)
            num += float(((fwd(x[idx].to(dev)) * s_out - ev_) ** 2).sum())
            den += float(((ev_ - e[tr].mean(0).to(dev)) ** 2).sum())
    info = {"mode": mode, "n_rows": int(N), "hidden": h, "epochs": epochs, "r2_heldout": 1.0 - num / den, "scale_in": s_in, "scale_out": s_out}
    extra = {"b1": b1.detach().cpu(), "W2": W2.detach().cpu(), "b2": b2.detach().cpu(),
             "scale_in": torch.tensor(s_in), "scale_out": torch.tensor(s_out)}
    return {"W": W1.detach().cpu(), "c": None, "extra": extra, "info": info}


# --------------------------------------------------------------------------- loading
def load_npz_dict(path, layer: int, width: str, dtype=torch.float32, sae_id: Optional[str] = None) -> Dictionary:
    """Our own params.npz (trained dictionaries; random test dictionaries): Gemma Scope's arrays (W_enc [d, n], W_dec [n, d],
    b_enc, b_dec, threshold) plus optional kind (str), k, mag_scale, b_mag."""
    with np.load(path, allow_pickle=False) as z:
        kind = str(z["kind"]) if "kind" in z.files else "jumprelu"
        t = lambda a: torch.from_numpy(np.ascontiguousarray(z[a])).to(dtype)
        n = z["W_dec"].shape[0]
        th = t("threshold") if "threshold" in z.files else torch.zeros(n, dtype=dtype)
        return Dictionary(t("W_enc").T.contiguous(), t("b_enc"), th, t("W_dec"), t("b_dec").float(), width, layer, sae_id,
                          kind=kind, k=int(z["k"]) if "k" in z.files else None,
                          mag_scale=t("mag_scale") if "mag_scale" in z.files else None,
                          b_mag=t("b_mag") if "b_mag" in z.files else None)


def save_npz_dict(path, d: Dictionary) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {"W_enc": d.W_enc_T.float().T.cpu().numpy(), "W_dec": d.W_dec.float().cpu().numpy(), "b_enc": d.b_enc.float().cpu().numpy(),
           "b_dec": d.b_dec.float().cpu().numpy(), "threshold": d.threshold.float().cpu().numpy(), "kind": np.array(d.kind)}
    if d.k:
        arr["k"] = np.array(int(d.k))
    if d.mag_scale is not None:
        arr["mag_scale"], arr["b_mag"] = d.mag_scale.float().cpu().numpy(), d.b_mag.float().cpu().numpy()
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arr)
    tmp.replace(path)
    return path


def random_dict(d_model: int, d_sae: int, kind: str = "jumprelu", seed: int = 0, k: int = 8, thr: float = 0.05,
                layer: int = 0, width: str = "?") -> Dictionary:
    """A random dictionary of any kind (tests and tiny runs only)."""
    g = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(d_sae, d_model, generator=g)
    W_dec /= W_dec.norm(dim=1, keepdim=True)
    W_enc_T = W_dec * 1.5
    b_enc = torch.randn(d_sae, generator=g) * 0.1
    th = thr * (0.5 + torch.rand(d_sae, generator=g))
    b_dec = torch.randn(d_model, generator=g) * 0.01
    if kind == "jumprelu":
        return Dictionary(W_enc_T, b_enc, th, W_dec, b_dec, width, layer, kind="jumprelu")
    if kind == "topk":
        return Dictionary(W_enc_T, b_enc, torch.zeros(d_sae), W_dec, b_dec, width, layer, kind="topk", k=k)
    return Dictionary(W_enc_T, b_enc, torch.zeros(d_sae), W_dec, b_dec, width, layer, kind="gated",
                      mag_scale=torch.exp(torch.randn(d_sae, generator=g) * 0.1), b_mag=torch.randn(d_sae, generator=g) * 0.1)


def trained_dir(cache_root: Path, name: str) -> Path:
    return Path(cache_root) / "trained_dicts" / name


def trained_digest(cache_root: Path, name: str) -> Optional[str]:
    """sha256 (16 hex) of a trained dictionary's params.npz, as recorded at training (None if it is not trained yet)."""
    d = trained_dir(cache_root, name)
    if not (d / "complete.json").exists():
        return None
    rec = json.loads((d / "complete.json").read_text())
    if rec.get("params_sha256"):
        return rec["params_sha256"][:16]
    import hashlib
    h = hashlib.sha256()
    with open(d / "params.npz", "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_trained(cache_root: Path, did: DictId, dtype=torch.float32) -> Dictionary:
    d = trained_dir(cache_root, did.name)
    if not (d / "complete.json").exists():
        raise FileNotFoundError(f"trained dictionary {did.name} is not complete in {d} (its training job must run first)")
    return load_npz_dict(d / "params.npz", did.layer, did.width, dtype, did.key)


def load_registry_dict(did: DictId, dtype=torch.float32) -> Dictionary:
    """Download (pinned revision) and convert a registry dictionary. Formats (`arch`):
        dl_<variant>   saprmarks/dictionary_learning checkpoints (ae.pt + config.json), variant one of
                       relu | topk | batchtopk | jumprelu | gated   (SAEBench, chanind Matryoshka-BatchTopK = batchtopk)
        gs2            Gemma Scope 2 JumpReLU (params.safetensors)
    """
    from huggingface_hub import hf_hub_download
    paths = [Path(hf_hub_download(did.repo, f, revision=did.revision)) for f in did.files]
    if did.arch.startswith("dl_"):
        from harness import dict_formats
        return dict_formats.from_dictionary_learning(paths, did.arch[3:], did.layer, did.width, did.key, dtype)
    if did.arch == "gs2":
        from harness import dict_formats
        return dict_formats.from_gemma_scope_2(paths, did.layer, did.width, did.key, dtype)
    raise ValueError(f"unknown registry arch {did.arch}")

"""Site-matched composite dictionaries (pilot P12/P13: Gemma 3 + Gemma Scope 2).

Every dictionary of P12 is scored at the SAME site as a residual SAE: the residual stream after layer L. A composite turns
sublayer dictionaries into a decomposition x_L = x_hat + eps of that residual, from activations of the same (real) run:

    x_L = embed + sum_{l <= L} attn_l + sum_{l <= L} mlp_l                       (exact; TransformerLens hooks)

    MLP replacement (attention kept exact; the circuit-tracing 'replacement model' convention)
        eps = sum_{l in S} (mlp_l - mlp_hat_l)
        tc      transcoder_all / transcoder:  mlp_hat_l = JumpReLU(mlp_in_l) W_dec + b_dec  (+ mlp_in_l A for *_affine)
        mlpsae  mlp_out_all SAE:              mlp_hat_l = SAE(mlp_l)
        clt     cross-layer transcoder:       mlp_hat_j = sum_{l <= j} f_l W_dec_l[:, j] + b_dec_j (+ mlp_in_j A_j)
                (decoders of layer l are zero for targets j < l; so sum_{j<=L} mlp_hat_j needs layers 0..L only)
        S = every layer 0..L (`*_all`, CLT) or {L} alone (`transcoder/` single-layer files)
    + attention SAEs (attn_out_all: o_proj input z, d = n_heads * d_head)
        eps += sum_{l <= L} (ln1_post(z_l W_O + b_O) - ln1_post(z_hat_l W_O + b_O))
    crosscoder (weakly causal, layers 7/13/17/22): latent block i reads the residual at layer L_i and decodes to layers
        >= L_i, so x_hat_L = sum_{i: L_i <= L} f_i W_dec_i[:, k(L)] + b_dec_k(L) uses layers <= L only; eps = x_L - x_hat_L.

x_hat := x_L - eps, so every downstream condition (feat_A_errB, capture along U, ...) is unchanged. Formats and the
conventions above were checked against real Gemma 3 1B activations (reconstruction FVU 0.01-0.37; docs/PILOT_G3_PLAN.md).

A composite is a registry dictionary with arch 'gs2c'; its `files` (Gemma Scope 2 paths) determine every part:
    transcoder_all/layer_<l>_<variant>/params.safetensors      tc at layer l (affine if the variant ends in _affine)
    transcoder/layer_<l>_<variant>/params.safetensors          tc at layer l (single-layer replacement)
    mlp_out_all/layer_<l>_<variant>/params.safetensors         mlpsae at layer l
    attn_out_all/layer_<l>_<variant>/params.safetensors        attention SAE at layer l
    clt/<variant>/params_layer_<l>.safetensors                 CLT source layer l
    crosscoder/layer_7_13_17_22_<variant>/params_layer_<i>.safetensors   crosscoder block i
Stage 1 caches the sublayer activations a composite reads (`needed_sites`); stage 2 (`eps_for_entries`) goes file by file
over all shards of a task (each weight file is read once per task), accumulating eps per shard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from harness.device import empty_cache

SITES = ("resid_post", "mlp_in", "mlp_out", "attn_z")
XC_LAYERS_RE = re.compile(r"crosscoder/layer_((?:\d+_)+)width")


@dataclass(frozen=True)
class Part:
    kind: str                  # tc | mlpsae | clt | attn | xc
    layer: int                 # sublayer index (tc / mlpsae / clt source / attn) or the block's source layer (xc)
    file: str
    affine: bool = False
    block: int = -1            # crosscoder block index
    xc_layers: Tuple[int, ...] = ()


def parse_parts(files: Sequence[str]) -> List[Part]:
    out = []
    for f in files:
        if not f.endswith(".safetensors"):
            continue
        top = f.split("/")[0]
        if top in ("transcoder_all", "transcoder", "mlp_out_all", "mlp_out", "attn_out_all", "attn_out"):
            m = re.match(r"[a-z_]+/layer_(\d+)_(\S+)/params\.safetensors$", f)
            if not m:
                raise ValueError(f"unrecognised Gemma Scope 2 path {f}")
            kind = {"transcoder_all": "tc", "transcoder": "tc", "mlp_out_all": "mlpsae", "mlp_out": "mlpsae",
                    "attn_out_all": "attn", "attn_out": "attn"}[top]
            out.append(Part(kind, int(m.group(1)), f, affine=m.group(2).endswith("_affine")))
        elif top == "clt":
            m = re.match(r"clt/(\S+)/params_layer_(\d+)\.safetensors$", f)
            out.append(Part("clt", int(m.group(2)), f, affine=m.group(1).endswith("_affine")))
        elif top == "crosscoder":
            m = re.match(r"crosscoder/layer_((?:\d+_)+)width\S*/params_layer_(\d+)\.safetensors$", f)
            ls = tuple(int(x) for x in m.group(1).strip("_").split("_"))
            i = int(m.group(2))
            out.append(Part("xc", ls[i], f, block=i, xc_layers=ls))
        else:
            raise ValueError(f"unrecognised Gemma Scope 2 path {f}")
    return out


def validate(parts: Sequence[Part], L: int) -> None:
    kinds = {p.kind for p in parts}
    if not parts:
        raise ValueError("composite without parts")
    if "xc" in kinds and kinds != {"xc"}:
        raise ValueError("a crosscoder composite cannot be mixed with other parts")
    if "xc" in kinds:
        ls = parts[0].xc_layers
        if L not in ls:
            raise ValueError(f"crosscoder layers {ls} do not include the cell layer {L}")
        need = {i for i, l in enumerate(ls) if l <= L}
        if {p.block for p in parts} != need:
            raise ValueError(f"crosscoder at layer {L} needs blocks {sorted(need)}")
        return
    mlp = [p for p in parts if p.kind in ("tc", "mlpsae", "clt")]
    if len({p.kind for p in mlp}) > 1:
        raise ValueError("one MLP dictionary kind per composite")
    if any(p.layer > L for p in parts):
        raise ValueError(f"composite parts above the cell layer {L}")
    if any(p.kind == "clt" for p in mlp) and sorted(p.layer for p in mlp) != list(range(L + 1)):
        raise ValueError(f"a CLT composite at layer {L} needs source layers 0..{L}")


def needed_sites(parts: Sequence[Part], L: int) -> Set[Tuple[int, str]]:
    """(layer, site) activations stage 1 must cache besides the cell's own (L, resid_post)."""
    s: Set[Tuple[int, str]] = set()
    for p in parts:
        if p.kind in ("tc", "clt"):
            s |= {(p.layer, "mlp_in"), (p.layer, "mlp_out")}
        elif p.kind == "mlpsae":
            s.add((p.layer, "mlp_out"))
        elif p.kind == "attn":
            s.add((p.layer, "attn_z"))
        elif p.kind == "xc":
            s.add((p.layer, "resid_post"))
    s.discard((L, "resid_post"))
    return s


def attn_layers(parts: Sequence[Part]) -> List[int]:
    return sorted({p.layer for p in parts if p.kind == "attn"})


def _jumprelu(x: torch.Tensor, w: dict) -> torch.Tensor:
    pre = x @ w["w_enc"] + w["b_enc"]
    return torch.relu(pre) * (pre > w["threshold"])


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """TransformerLens RMSNorm (float32): x / sqrt(mean(x^2) + eps) * w."""
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * w


def _load(path: Path, device) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return {k: v.to(device=device, dtype=torch.float32) for k, v in load_file(str(path)).items()}


class Composite:
    """A composite dictionary (see module doc). Not a JumpReLUSAE: stage 2 calls `eps_for_entries`, never decompose()."""
    composite = True
    streamed = False

    def __init__(self, parts: Sequence[Part], layer: int, width: str, sae_id: str,
                 fetch: Callable[[str], Path], row_budget_bytes: int = 512 * 2 ** 20):
        validate(parts, layer)
        self.parts, self.layer, self.width, self.sae_id = list(parts), int(layer), width, sae_id
        self.fetch = fetch                         # Gemma Scope 2 path -> local file (pinned revision)
        self.device = torch.device("cpu")
        self.row_budget = row_budget_bytes

    def to_device(self, device) -> "Composite":
        self.device = torch.device(device)
        return self

    def sites(self) -> Set[Tuple[int, str]]:
        return needed_sites(self.parts, self.layer)

    def _chunks(self, n: int, width: int):
        step = max(1, self.row_budget // max(1, 4 * 3 * width))
        for a in range(0, n, step):
            yield slice(a, min(n, a + step))

    @torch.inference_mode()
    def eps_for_entries(self, base, entries: Dict[Tuple[int, str], "object"], sub_params: Optional[dict] = None,
                        log: Optional[Callable[[str], None]] = None) -> Dict[int, Dict[str, torch.Tensor]]:
        """{shard: {'eps_A', 'eps_B' [rows, d] float32 (CPU), 'l0_A', 'l0_B' [rows]}} for every shard of the task's span
        entry `base` (the cell's (L, resid_post)); `entries` holds the other (layer, site) span entries (stage 1, same shard
        plan); `sub_params` the attention output maps {layer: {W_O, b_O, ln1_post_w, eps}} when attention SAEs are parts."""
        dev = self.device
        shards = list(base.shards())
        acc: Dict[int, Dict[str, torch.Tensor]] = {}
        pidx = {}
        for si in shards:
            s = base.read_shard(si)
            pidx[si] = s["pair_index"]
            acc[si] = {f"eps_{k}": torch.zeros(s[f"x_{k}"].shape, dtype=torch.float32) for k in ("A", "B")}
            acc[si].update({f"l0_{k}": torch.zeros(len(s[f"x_{k}"]), dtype=torch.long) for k in ("A", "B")})
            if self.parts[0].kind == "xc":
                for k in ("A", "B"):
                    acc[si][f"eps_{k}"] += s[f"x_{k}"].float()          # eps = x_L - x_hat_L, x_hat built below

        def rows(site_key, si):
            e = base if site_key == (self.layer, "resid_post") else entries[site_key]
            t = e.read_shard(si)
            if not torch.equal(t["pair_index"], pidx[si]):
                raise ValueError(f"span entry {site_key} is not aligned with the cell's span cache (shard {si})")
            return t

        L = self.layer
        for p in sorted(self.parts, key=lambda p: (p.kind, p.layer, p.block)):
            w = _load(self.fetch(p.file), dev)
            if log:
                log(f"composite {self.sae_id}: {p.kind} layer {p.layer} ({Path(p.file).parent.name})")
            width = int(w["b_enc"].shape[0])
            if p.kind == "clt":
                # sum over targets j in [l, L] of this source layer's decoder; target-j bias / affine come from file j
                Wsum = w["w_dec"][:, p.layer:L + 1, :].sum(1)
            for si in shards:
                if p.kind in ("tc", "clt"):
                    tin, tout = rows((p.layer, "mlp_in"), si), rows((p.layer, "mlp_out"), si)
                elif p.kind == "mlpsae":
                    tin = tout = rows((p.layer, "mlp_out"), si)
                elif p.kind == "attn":
                    tin = rows((p.layer, "attn_z"), si)
                else:
                    tin = rows((p.layer, "resid_post"), si)
                for k in ("A", "B"):
                    x_in = tin[f"x_{k}"]
                    eps, l0 = acc[si][f"eps_{k}"], acc[si][f"l0_{k}"]
                    for sl in self._chunks(len(x_in), width):
                        xi = x_in[sl].to(dev, torch.float32)
                        f = _jumprelu(xi, w)
                        l0[sl] += (f > 0).sum(-1).cpu()
                        if p.kind == "tc":
                            yh = f @ w["w_dec"] + w["b_dec"] + (xi @ w["affine_skip_connection"] if p.affine else 0)
                            eps[sl] += (tout[f"x_{k}"][sl].to(dev, torch.float32) - yh).cpu()
                        elif p.kind == "mlpsae":
                            eps[sl] += (xi - (f @ w["w_dec"] + w["b_dec"])).cpu()
                        elif p.kind == "clt":
                            d = -(f @ Wsum) + tout[f"x_{k}"][sl].to(dev, torch.float32) - w["b_dec"]
                            if p.affine:
                                d = d - xi @ w["affine_skip_connection"]
                            eps[sl] += d.cpu()
                        elif p.kind == "attn":
                            sp = sub_params[p.layer]
                            zh = f @ w["w_dec"] + w["b_dec"]
                            out = lambda z: _rms(z @ sp["W_O"].to(dev) + sp["b_O"].to(dev), sp["ln1_post_w"].to(dev), sp["eps"])
                            eps[sl] += (out(xi) - out(zh)).cpu()
                        else:                                       # crosscoder block: subtract its decode at layer L
                            k_L = p.xc_layers.index(L)
                            eps[sl] -= (f @ w["w_dec"][:, k_L, :]).cpu()
            if p.kind == "xc" and p.xc_layers[p.block] == L:
                for si in shards:                                   # b_dec of the target layer lives in its own file
                    for k in ("A", "B"):
                        acc[si][f"eps_{k}"] -= w["b_dec"].cpu()
            del w
            empty_cache()
        return acc


def from_registry(did, fetch: Callable[[str], Path]) -> Composite:
    return Composite(parse_parts(did.files), did.layer, did.width, did.key, fetch)

"""Experiment 3: hybrid low-rank repair. x_hybrid(r) = x_hat + U_r U_r^T eps, with U_r fit on TRAIN errors only.

Subspace methods (cfg 'exp3_methods'):
    unsupervised_svd          top-r right singular vectors of centered train errors (last position)
    supervised_centroid_svd   SVD of state centroids of train errors, padded by residual SVD
    random_subspace           Haar-random U_r (null baseline; averaged over cfg 'random_seeds')
    top_xhat_svd              top-r directions of the train *reconstructions* (wrong-space baseline)
    span_svd                  unsupervised SVD of train errors at every position from the first differing token
    generic_text_svd          unsupervised SVD of errors on generic web text (task-agnostic, no labels)
    transfer                  supervised U of every *other* task (method name transfer_from_<task>)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.experiments.common import RunContext, decompose_batched, generic_resid, run_patched, summarize
from src.experiments.exp2_geometry import _stack, centered_svd
from src.models.sae_loader import parse_width
from src.utils import save_json, save_npz, save_table

log = logging.getLogger("prism")
METHODS = ["unsupervised_svd", "supervised_centroid_svd", "random_subspace", "top_xhat_svd", "span_svd"]
SPLICE_METHODS = {"unsupervised_svd", "supervised_centroid_svd"}   # hybrid_splice is only run for these


def _orthonormalize(V: np.ndarray) -> np.ndarray:
    """Columns of V -> orthonormal columns spanning the same nested prefixes (QR with sign fix)."""
    Q, R = np.linalg.qr(V)
    return Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))


def fit_unsupervised(E_train: torch.Tensor, r_max: int) -> np.ndarray:
    _, Vt = centered_svd(E_train)
    return Vt[:r_max].T.copy()                                   # [d, r_max]


def fit_supervised(E_train: torch.Tensor, states: np.ndarray, r_max: int) -> np.ndarray:
    """Top singular vectors of the (centered) class centroids, padded with residual-SVD directions."""
    E = E_train.double().numpy()
    cls = np.unique(states)
    cent = np.stack([E[states == c].mean(0) for c in cls])
    w = np.array([(states == c).sum() for c in cls], dtype=float)
    cent = cent - (w[:, None] * cent).sum(0) / w.sum()
    _, S, Vt = np.linalg.svd(cent, full_matrices=False)
    k1 = int((S > 1e-8 * max(S[0], 1e-30)).sum())
    V = Vt[:min(k1, r_max)].T                                    # [d, k1]
    if V.shape[1] < r_max:
        resid = (E - E.mean(0)) - ((E - E.mean(0)) @ V) @ V.T
        _, _, Vr = np.linalg.svd(resid, full_matrices=False)
        V = np.concatenate([V, Vr[: r_max - V.shape[1]].T], 1)
    return _orthonormalize(V)


def fit_cov_eig(E: torch.Tensor, r_max: int) -> np.ndarray:
    """Top eigenvectors of the error covariance (= right singular vectors; cheap for many rows)."""
    E = E.double()
    Ec = E - E.mean(0, keepdim=True)
    w, V = torch.linalg.eigh(Ec.T @ Ec)
    return V[:, torch.argsort(w, descending=True)[:r_max]].numpy().copy()


def random_subspace(d: int, r_max: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    return _orthonormalize(g.standard_normal((d, r_max)))


def sae_param_count(d_model: int, d_sae: int) -> int:
    return 2 * d_model * d_sae + d_sae + d_model                 # W_enc, W_dec, b_enc, b_dec


def _fit_subspaces(ctx: RunContext, sae, task: str, layer: int, r_max_cfg: int, methods: Sequence[str]) -> Dict[str, np.ndarray]:
    tr = ctx.data[task]["train"]
    Xtr, ytr = _stack(tr, layer)
    xh_tr, E_tr, _ = decompose_batched(sae, Xtr)                                     # TRAIN only
    d = Xtr.shape[1]
    r_max = min(r_max_cfg, len(E_tr) - 1, d)
    U: Dict[str, np.ndarray] = {}
    if "unsupervised_svd" in methods:
        U["unsupervised_svd"] = fit_unsupervised(E_tr, r_max)
    if "supervised_centroid_svd" in methods or "transfer" in methods:
        U["supervised_centroid_svd"] = fit_supervised(E_tr, ytr, r_max)
    if "top_xhat_svd" in methods:
        U["top_xhat_svd"] = fit_unsupervised(xh_tr, r_max)
    if "span_svd" in methods and tr.span_clean:
        _, E_span, _ = decompose_batched(sae, torch.cat([tr.acts(layer, "clean", "from_diff"),
                                                          tr.acts(layer, "corrupt", "from_diff")]))
        U["span_svd"] = fit_cov_eig(E_span, min(r_max_cfg, d))
    if "random_subspace" in methods:
        for s in range(ctx.cfg.get("random_seeds", 3)):
            U[f"random_subspace#{s}"] = random_subspace(d, min(r_max_cfg, d), ctx.cfg["seed"] + 101 * s + layer)
    return U


def _unit(ctx: RunContext, sae, task: str, layer: int, width: str, positions: str, U_all: Dict[str, Dict[str, np.ndarray]],
          ranks: Sequence[int], methods: Sequence[str]) -> List[dict]:
    te = ctx.data[task]["test"]
    d = ctx.d_model
    d_sae = getattr(sae, "d_sae", None) or parse_width(width)
    xh_c, e_c, _ = decompose_batched(sae, te.acts(layer, "clean", positions))
    xh_k, e_k, _ = decompose_batched(sae, te.acts(layer, "corrupt", positions))
    n_boot = ctx.cfg.get("bootstrap", 1000)
    Us = dict(U_all[task])
    if "supervised_centroid_svd" not in methods:
        Us.pop("supervised_centroid_svd", None)          # only fitted to serve as a transfer source
    if "transfer" in methods:
        for other, Uo in U_all.items():
            if other != task and "supervised_centroid_svd" in Uo:
                Us[f"transfer_from_{other}"] = Uo["supervised_centroid_svd"]
    if "__generic__" in U_all:
        Us["generic_text_svd"] = U_all["__generic__"]["generic_text_svd"]
    acc: Dict[tuple, List[dict]] = {}
    for name, Uf in Us.items():
        method = name.split("#")[0]
        kinds = ("hybrid_patch", "hybrid_splice") if method in SPLICE_METHODS else ("hybrid_patch",)
        for r in ranks:
            if r > Uf.shape[1]:
                continue
            U = torch.from_numpy(Uf[:, :r]).float()                                     # [d, r]; r=0 -> empty
            proj = lambda e: (e @ U) @ U.T
            vecs = {"hybrid_patch": xh_c + e_k + proj(e_c - e_k),                         # xh_c + P e_c + (I-P) e_k
                    "hybrid_splice": xh_c + proj(e_c)}
            d_ck = e_c - e_k
            energy = {"eps_energy_captured": float((proj(e_c).norm(dim=-1) ** 2).sum() / (e_c.norm(dim=-1) ** 2).sum()),
                      "diff_energy_captured": float((proj(d_ck).norm(dim=-1) ** 2).sum() / (d_ck.norm(dim=-1) ** 2).sum().clamp_min(1e-12))}
            for kind in kinds:
                dl, top1 = run_patched(ctx.model, te, layer, vecs[kind], ctx.device, positions=positions)
                acc.setdefault((method, r, kind), []).append({**summarize(dl, top1, te, n_boot), **energy})
    base = dict(task=task, layer=layer, width=width, positions=positions)
    rows = []
    for (method, r, kind), rs in acc.items():
        agg = {k: float(np.nanmean([x[k] for x in rs])) for k in rs[0]}
        rows.append(dict(base, method=method, rank=r, kind=kind, n_seeds=len(rs), extra_params=r * d,
                         frac_of_sae_params=r * d / sae_param_count(d, d_sae), **agg))
    dl, top1 = run_patched(ctx.model, te, layer, xh_k + e_c, ctx.device, positions=positions)   # reference = error_only
    rows.append(dict(base, method="reference", rank=-1, kind="error_only", n_seeds=1, extra_params=0,
                     frac_of_sae_params=0.0, **summarize(dl, top1, te, n_boot)))
    log.info("exp3 %s L%d %s [%s]: supervised patch NLDR by rank %s", task, layer, width, positions,
             {r["rank"]: round(r["nldr"], 2) for r in rows if r["method"] == "supervised_centroid_svd" and r["kind"] == "hybrid_patch"})
    return rows


def run_exp3(ctx: RunContext, layers: Optional[Sequence[int]] = None, widths: Optional[Sequence[str]] = None,
             ranks: Optional[Sequence[int]] = None) -> List[dict]:
    layers = list(layers if layers is not None else ctx.cfg["geometry_layers"])
    widths = list(widths if widths is not None else ctx.cfg["geometry_widths"])
    ranks = sorted(ranks if ranks is not None else ctx.cfg["ranks"])
    modes = list(ctx.cfg.get("exp3_positions", ["last"]))
    methods = list(ctx.cfg.get("exp3_methods", ["unsupervised_svd", "supervised_centroid_svd"]))
    ctx.prepare(sorted(set(layers) | set(ctx.cached_layers)))
    d = ctx.d_model
    units = ctx.unit_log("exp3")
    todo = lambda l, w: [(t, m) for t in ctx.data for m in modes if not units.done(units.key(t, l, w, m))]
    combos = [(l, w) for l in layers for w in widths if todo(l, w)]
    for layer, width, sae in ctx.iter_saes("exp3", combos):
        try:
            U_all = {t: _fit_subspaces(ctx, sae, t, layer, max(ranks), methods) for t in ctx.data}
            for t, U in U_all.items():
                save_npz(ctx.out_dir / "tensors" / f"exp3_U_{t}_L{layer}_{width}.npz",
                         **{k.replace("#", "_seed"): v for k, v in U.items()})
            if "generic_text_svd" in methods:
                G = generic_resid(ctx, layer)
                if G is not None:
                    _, E_g, _ = decompose_batched(sae, G)
                    U_all["__generic__"] = {"generic_text_svd": fit_cov_eig(E_g, min(max(ranks), d))}
        except Exception as e:  # noqa: BLE001
            log.error("exp3: subspace fitting failed L%d %s: %s", layer, width, e)
            ctx.record_failure("exp3", f"fit|{layer}|{width}", e)
            del sae
            continue
        for task, mode in todo(layer, width):
            ctx.run_unit("exp3", units.key(task, layer, width, mode),
                         lambda: _unit(ctx, sae, task, layer, width, mode, U_all, ranks, methods), units)
        del sae
    rows = units.rows()
    sizes = {w: parse_width(w) for w in widths}
    overhead = {"d_model": d, "ranks": ranks, "rank_params": {int(r): r * d for r in ranks},
                "sae_params": {w: sae_param_count(d, n) for w, n in sizes.items()}}
    if "16k" in sizes and "65k" in sizes:
        overhead["width_scaling_extra_params_16k_to_65k"] = (sae_param_count(d, sizes["65k"]) - sae_param_count(d, sizes["16k"]))
    out = ctx.out_dir / "metrics"
    save_json({"rows": rows, "overhead": overhead}, out / "exp3_results.json")
    if rows:
        save_table(rows, out / "exp3_results")
    return rows

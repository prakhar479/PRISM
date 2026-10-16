"""Experiment 1: causal split. Patch the corrupt run at resid_post[layer] (last position, or every position from
the first differing token) with sae_only / error_only / sae_splice / noise_control vectors and measure NLDR.

Controls (last position only, since span lengths differ between pairs):
    clean_full                 x_clean                       (sanity: should give NLDR ~ 1)
    error_resample_same_state  x_hat_corrupt + eps_clean[j], pair j != i with the same clean state
    error_resample_diff_state  x_hat_corrupt + eps_clean[j], pair j with a different clean state
    mean_error                 x_hat_corrupt + mean train eps (removes all pair-specific error content)
If error_resample_same_state ~ error_only >> error_resample_diff_state, eps carries the *state*, not prompt identity.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.experiments.common import (RunContext, decompose_batched, rescaled_noise, run_patched, summarize)
from src.utils import free_memory, save_json, save_table
from src.models.sae_loader import load_sae

log = logging.getLogger("prism")

INTERVENTIONS = ["sae_only", "error_only", "sae_splice", "noise_control"]
CONTROLS = ["clean_full", "error_resample_same_state", "error_resample_diff_state", "mean_error"]


def build_vectors(xh_c, e_c, xh_k, e_k, eta=None) -> Dict[str, torch.Tensor]:
    out = {
        "sae_only": xh_c + e_k,        # sparse part from clean, error from corrupt
        "error_only": xh_k + e_c,      # error from clean, sparse part from corrupt
        "sae_splice": xh_c,            # plain SAE replacement of the clean activation (no error term)
    }
    if eta is not None:
        out["noise_control"] = xh_k + eta
    return out


def resample_partners(states: np.ndarray, same: bool, seed: int) -> np.ndarray:
    """For each i a partner j != i with equal (same=True) or different clean state; i itself if none exists."""
    rng = np.random.default_rng(seed)
    out = np.arange(len(states))
    for i, s in enumerate(states):
        pool = np.flatnonzero((states == s) if same else (states != s))
        pool = pool[pool != i]
        if len(pool):
            out[i] = rng.choice(pool)
    return out


def _unit(ctx: RunContext, sae, task: str, layer: int, width: str, positions: str) -> List[dict]:
    base = ctx.data[task]["test"]
    xc, xk = base.acts(layer, "clean", positions), base.acts(layer, "corrupt", positions)
    xh_c, e_c, l0_c = decompose_batched(sae, xc)
    xh_k, e_k, l0_k = decompose_batched(sae, xk)
    rel_err = float(((e_c.norm(dim=-1) ** 2 / xc.norm(dim=-1) ** 2).mean()
                     + (e_k.norm(dim=-1) ** 2 / xk.norm(dim=-1) ** 2).mean()) / 2)
    cos_err = float(torch.nn.functional.cosine_similarity(e_c, e_k, dim=-1).mean())
    common = dict(task=task, layer=layer, width=width, positions=positions, n=len(base), l0=(l0_c + l0_k) / 2,
                  rel_recon_err=rel_err, eps_norm=float(e_c.norm(dim=-1).mean()), eps_clean_corrupt_cos=cos_err,
                  dl_clean=float(base.dl_clean.mean()), dl_corrupt=float(base.dl_corrupt.mean()),
                  sae_id=getattr(sae, "sae_id", None))
    n_boot = ctx.cfg.get("bootstrap", 1000)
    results: Dict[str, List[dict]] = {}
    for seed in range(ctx.cfg.get("noise_seeds", 1)):
        gen = torch.Generator().manual_seed(ctx.cfg["seed"] + 1000 * seed + layer)
        vecs = build_vectors(xh_c, e_c, xh_k, e_k, rescaled_noise(e_c, gen))
        if seed == 0:
            vecs["clean_full"] = xc
            if positions == "last" and "controls" in ctx.cfg.get("exp1_extras", ["controls"]):
                s = base.states_clean
                vecs["error_resample_same_state"] = xh_k + e_c[resample_partners(s, True, ctx.cfg["seed"] + layer)]
                vecs["error_resample_diff_state"] = xh_k + e_c[resample_partners(s, False, ctx.cfg["seed"] + layer)]
                tr = ctx.data[task]["train"]
                _, e_tr, _ = decompose_batched(sae, torch.cat([tr.resid_clean[layer], tr.resid_corrupt[layer]]))
                vecs["mean_error"] = xh_k + e_tr.mean(0, keepdim=True)
        for name, v in vecs.items():
            if seed > 0 and name != "noise_control":
                continue  # deterministic interventions need one pass only
            dl, top1 = run_patched(ctx.model, base, layer, v, ctx.device, positions=positions)
            results.setdefault(name, []).append(summarize(dl, top1, base, n_boot))
    rows = []
    for name, rs in results.items():
        agg = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
        rows.append(dict(common, intervention=name, **agg))
    log.info("exp1 %s L%d %s [%s]: %s", task, layer, width, positions,
             {r["intervention"]: round(r["nldr"], 2) for r in rows})
    return rows


def run_exp1(ctx: RunContext, layers: Optional[Sequence[int]] = None, widths: Optional[Sequence[str]] = None) -> List[dict]:
    layers = list(layers if layers is not None else ctx.cfg["layers"])
    widths = list(widths if widths is not None else ctx.cfg["widths"])
    modes = list(ctx.cfg.get("exp1_positions", ["last"]))
    ctx.prepare(sorted(set(layers) | set(ctx.cached_layers)))
    units = ctx.unit_log("exp1")
    for layer in layers:
        for width in widths:
            todo = [(t, m) for t in ctx.data for m in modes if not units.done(units.key(t, layer, width, m))]
            if not todo or ctx.out_of_time():
                continue
            try:
                sae = load_sae(layer, width, model=ctx.model, mock=ctx.mock, device=ctx.device)   # one SAE at a time
            except Exception as e:  # noqa: BLE001 - missing SAE id / download failure: skip this (layer, width)
                log.error("exp1: cannot load SAE L%d %s: %s", layer, width, e)
                ctx.record_failure("exp1", f"load_sae|{layer}|{width}", e)
                continue
            for task, mode in todo:
                ctx.run_unit("exp1", units.key(task, layer, width, mode),
                             lambda: _unit(ctx, sae, task, layer, width, mode), units)
            del sae
            free_memory()
    rows = units.rows()
    out = ctx.out_dir / "metrics"
    save_json(rows, out / "exp1_results.json")
    if rows:
        save_table(rows, out / "exp1_results")
    return rows

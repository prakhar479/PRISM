#!/usr/bin/env python
"""Golden-fixture export (RUN machine). Captures pilot ground truth through the PILOT code paths (src/), so that
the new harness can later be checked against it independently of any refactor.

    python scripts/export_golden.py --out golden/run1            # real model + SAEs (GPU machine), ~30-40 min
    python scripts/export_golden.py --mock --out /tmp/golden     # CPU dry run of the plumbing (no weights)

What it does, per cell (default: entity_track L12 and add_chain L18, 16k SAE):
  1. Rebuilds the pilot train/test splits exactly as results/paper_20261026_001341 did (configs/paper_residue.yaml +
     that run's preflight.json task_overrides, prepare_task with from_diff spans) and records every scored candidate.
  2. Checks pair identity/order against the stored per-pair tensors (exp1_pairs_*_from_diff.npz: states and logit
     diffs) and re-runs the pilot conditions on all test pairs with the pilot code (run_patched + summarize),
     comparing NLDR and CIs with the stored exp1/exp3 CSVs and the refitted U with the stored U.
  3. Exports, for N fixed test pairs (ids recorded): token ids, first-diff index, answer ids, layer-L residuals
     (A at all positions; B from first_diff on, since earlier positions are identical), SAE pre-activations z for
     latents with z >= 0.5*threshold at patched positions (sparse), f, x_hat, eps, the pilot U matrices (r <= 8,
     both estimators + random) with the train pair ids, the noise rows used, and final-position logits for clean A,
     clean B (full vocab) and every pilot condition (candidate answers + logsumexp + top-k).
  4. Dumps all scored candidates of all four pilot tasks (prompts, token ids, filter scores, kept/split flags).

Output: <out>/{manifest.json, report.md, golden_<task>_L<layer>.npz, pairs_<task>.jsonl.gz,
pilot_reproduction.csv}, plus <out>.tar and <out>.tar.sha256. Nothing here is a result; it is a test fixture.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.data import synthetic_tasks as st
from src.experiments.common import (RunContext, decompose_batched, merge_task_overrides, prepare_task, rescaled_noise,
                                    run_patched, summarize)
from src.experiments.exp1_causal_split import build_vectors
from src.experiments.exp3_hybrid_repair import _fit_subspaces
from src.models import runtime
from src.models.sae_loader import hook_name, load_sae
from src.utils import get_device, get_dtype, load_config, load_json, save_json, set_seed, setup_logging

ROOT = Path(__file__).resolve().parents[1]
PILOT_RUN = "results/paper_20261026_001341"
PILOT_TIER = "paper_residue"
ALL_TASKS = ["entity_track", "add_chain", "var_binding", "ioi_control"]
CANDIDATES = {"entity_track": [f" {b}" for b in st.BOXES], "add_chain": [f" {d}" for d in range(10)],
              "var_binding": [f" {d}" for d in range(10)], "ioi_control": [f" {n}" for n in st.NAMES]}
Z_FRAC = 0.5          # keep pre-activations z >= Z_FRAC * threshold (E5 uses alpha >= 0.5)
R_EXPORT = 8
TOPK = 50
log = None


# --------------------------------------------------------------------------- helpers
def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _pkg(name: str):
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _hf_snapshots(repo: str) -> list:
    """Commit hashes of locally cached snapshots of `repo` (revision actually used when loading offline/cached)."""
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            if r.repo_id == repo:
                return sorted(rev.commit_hash for rev in r.revisions)
    except Exception:  # noqa: BLE001
        pass
    return []


def environment(device) -> dict:
    env = {"hostname": socket.gethostname(), "os": platform.platform(), "python": platform.python_version(),
           "packages": {p: _pkg(p) for p in ("torch", "transformers", "transformer-lens", "sae-lens", "numpy",
                                              "huggingface-hub", "safetensors")},
           "torch_cuda": torch.version.cuda, "device": str(device),
           "git": {"head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                   "dirty_files": _git("status", "--porcelain").splitlines()},
           "hf_snapshots": {r: _hf_snapshots(r) for r in ("google/gemma-2-2b", "google/gemma-scope-2b-pt-res")}}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name()
        env["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip()
    return env


class ScoreRecorder:
    """Wraps synthetic_tasks.score_pairs (called by the pilot filter) and keeps every candidate's scores."""

    def __init__(self) -> None:
        self.calls = []
        self._orig = st.score_pairs

    def __enter__(self):
        def rec(model, items, *a, **kw):
            out = self._orig(model, items, *a, **kw)
            self.calls.append((list(items), {k: np.asarray(v).copy() for k, v in out.items()}))
            return out
        st.score_pairs = rec
        return self

    def __exit__(self, *exc):
        st.score_pairs = self._orig

    def rows(self) -> dict:
        out = {}
        for items, sc in self.calls:
            for i, it in enumerate(items):
                out[it.pair.pair_id] = {k: (bool(v[i]) if v.dtype == bool else float(v[i])) for k, v in sc.items()}
        return out

    def items(self) -> dict:
        return {it.pair.pair_id: it for items, _ in self.calls for it in items}


def candidate_ids(tok, it: st.TokenizedPair, task: str) -> list:
    """Token ids of every candidate answer in this pair's answer format (None where not a single token)."""
    p = it.pair
    lead = p.clean_target_str.startswith(" ")
    return [st.answer_token_id(tok, p.clean_prompt, c if lead else c.lstrip()) for c in CANDIDATES[task]]


def logit_record(logits: torch.Tensor, cand: list) -> dict:
    """Final-position summary: candidate logits, logsumexp, top-k ids/values (logits post-softcap, float32)."""
    lg = logits.float()
    ids = torch.tensor([c if c is not None else 0 for c in cand])
    cl = lg[ids].clone()
    cl[torch.tensor([c is None for c in cand])] = float("nan")
    v, i = lg.topk(min(TOPK, lg.numel()))
    return {"cand": cl.numpy(), "lse": float(torch.logsumexp(lg, -1)), "topk_ids": i.numpy().astype(np.int32),
            "topk_vals": v.numpy()}


@torch.inference_mode()
def patched_last_logits(model, ids: list, layer: int, start: int, rows) -> torch.Tensor:
    """One prompt (batch 1) with resid_post[layer][start:] overwritten by rows [T-start, d] -> last logits [vocab].
    rows=None: unpatched. Full hooked forward (no resume), independent of how the pilot batched."""
    from src.utils import model_device

    dev = model_device(model)
    t = torch.tensor([ids], dtype=torch.long, device=dev)
    hooks = []
    if rows is not None:
        v = rows.to(dev)

        def patch(act, hook):
            act = act.clone()
            act[0, start:] = v.to(act.dtype)
            return act
        hooks = [(hook_name(layer), patch)]
    return runtime.last_logits(model, t, hooks)[0].cpu()


@torch.inference_mode()
def preacts(sae, x: torch.Tensor):
    """(z [N, d_sae] float32 on CPU, threshold [d_sae]) with the pilot SAE's own weights and arithmetic."""
    if hasattr(sae, "_chunks"):
        xw = x.to(sae.device, sae.dtype)
        zs, ths = [], []
        for We, be, th, _ in sae._chunks():
            zs.append((xw @ We.T + be).float().cpu())
            ths.append(th.float().cpu())
        return torch.cat(zs, 1), torch.cat(ths)
    z = (x.to(sae.device).float() @ sae.W_enc + sae.b_enc).cpu()       # MockSAE (TopK, no threshold)
    return z, torch.zeros(z.shape[1])


def sparse_z(z: torch.Tensor, th: torch.Tensor):
    keep = z >= Z_FRAC * th[None, :]
    keep &= z > 0
    r, c = keep.nonzero(as_tuple=True)
    vals = z[r, c]
    active = vals > th[c]
    return r.numpy().astype(np.int32), c.numpy().astype(np.int32), vals.numpy(), active.numpy()


# --------------------------------------------------------------------------- per cell
def load_cfg(a) -> dict:
    cfg = load_config(str(ROOT / a.config))
    pre = ROOT / a.pilot_run / "preflight" / "preflight.json"
    if pre.exists():
        ov = load_json(pre)
        cfg["task_overrides"] = merge_task_overrides(cfg.get("task_overrides"), ov.get("task_overrides"))
    if a.mock:   # same reductions as `run_server_gpu.sh --mock`
        cfg.update(n_train=a.n_train or 64, n_test=a.n_test or 32, min_pairs=8, bootstrap=200)
    return cfg


def stored(a, task: str, layer: int):
    d = ROOT / a.pilot_run / PILOT_TIER
    out = {}
    f = d / "tensors" / f"exp1_pairs_{task}_L{layer}_16k_from_diff.npz"
    if f.exists():
        with np.load(f) as z:
            out["pairs"] = {k: z[k] for k in z.files}
    f = d / "tensors" / f"exp3_U_{task}_L{layer}_16k.npz"
    if f.exists():
        with np.load(f) as z:
            out["U"] = {k: z[k] for k in z.files}
    for name in ("exp1", "exp3"):
        f = d / "metrics" / f"{name}_results.csv"
        if f.exists():
            import pandas as pd
            out[name] = pd.read_csv(f)
    return out


def vram(tag: str) -> None:
    """Log allocated/reserved VRAM at a stage boundary (OOM diagnosis)."""
    if torch.cuda.is_available():
        log.info("VRAM %s: %.2f GiB allocated, %.2f GiB reserved", tag, torch.cuda.memory_allocated() / 2 ** 30,
                 torch.cuda.memory_reserved() / 2 ** 30)


def dump_cuda_tensors(top: int = 25) -> None:
    """Log the largest live CUDA tensors (excluding model parameters) and who refers to them."""
    import gc

    if not torch.cuda.is_available():
        return
    params = {id(p) for p in ctx_model_params}
    rows = []
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda and id(o) not in params:
                refs = [type(r).__name__ for r in gc.get_referrers(o)[:4] if r is not rows]
                rows.append((o.numel() * o.element_size(), tuple(o.shape), str(o.dtype), refs))
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda r: -r[0])
    log.error("live non-parameter CUDA tensors: %d, %.2f GiB total", len(rows), sum(r[0] for r in rows) / 2 ** 30)
    for b, shape, dt, refs in rows[:top]:
        log.error("  %8.1f MiB %s %s <- %s", b / 2 ** 20, shape, dt, refs)
    log.error("%s", torch.cuda.memory_summary(abbreviated=True))


ctx_model_params: list = []


def check(checks: list, name: str, ok, detail: str) -> None:
    checks.append({"check": name, "status": "SKIP" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})
    log.info("[%s] %s: %s", checks[-1]["status"], name, detail)


def col_abs_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs((A * B).sum(0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0) + 1e-30))


def run_cell(a, ctx: RunContext, task: str, layer: int, out: Path, checks: list, repro: list, golden_ids_out: dict):
    cfg, model, tok = ctx.cfg, ctx.model, ctx.tok
    t0 = time.time()
    with ScoreRecorder() as rec:
        d = runtime.oom_retry(lambda: prepare_task(task, model, tok, cfg, ctx.mock, [layer], ctx.device, cache_spans=True),
                              f"prepare {task}")
    if not d:
        check(checks, f"{task}: splits built", False, "too few valid pairs")
        return {}, rec.rows(), rec.items()
    tr, te = d["train"], d["test"]
    log.info("%s: %d train / %d test pairs in %.0fs", task, len(tr), len(te), time.time() - t0)
    ctx.data = {task: d}
    ctx.cached_layers = (layer,)
    ctx.runtime_plan.pop("resume_check_max_abs_diff", None)
    vram("after prepare_task")
    ctx._check_resume()
    vram("after resume check")
    ref = stored(a, task, layer) if not a.mock else {}

    # ---- pair identity / order vs the stored pilot per-pair tensors
    if "pairs" in ref:
        P = ref["pairs"]
        same_n = len(P["dl_clean"]) == len(te)
        st_ok = same_n and np.array_equal(P["state_clean"], te.states_clean) and np.array_equal(P["state_corrupt"], te.states_corrupt)
        dmax = float(max(np.abs(P["dl_clean"] - te.dl_clean.numpy()).max(), np.abs(P["dl_corrupt"] - te.dl_corrupt.numpy()).max())) if same_n else float("nan")
        check(checks, f"{task} L{layer}: test pairs match the pilot (n, states, order)", st_ok,
              f"n={len(te)} (pilot {len(P['dl_clean'])}), states equal={st_ok}")
        check(checks, f"{task} L{layer}: clean/corrupt logit diffs match the pilot", same_n and dmax < 0.05,
              f"max |dD| = {dmax:.4g} (bf16 run-to-run tolerance 0.05)")
    else:
        check(checks, f"{task} L{layer}: test pairs match the pilot", None, "no stored per-pair tensors (mock)")

    # ---- SAE (pilot loader) and full-test-set decomposition
    sae = load_sae(layer, "16k", model=model, mock=ctx.mock, device=ctx.device,
                   release=cfg.get("sae_release", "gemma-scope-2b-pt-res-canonical"))
    sae_info = {"sae_id": getattr(sae, "sae_id", None), "dtype": str(getattr(sae, "dtype", "")),
                "streamed": bool(getattr(sae, "streamed", False)), "d_sae": int(getattr(sae, "d_sae", 0))}
    if not ctx.mock:
        try:
            from src.models.sae_loader import _sae_params_path, resolve_sae_id
            rel, sid = resolve_sae_id(layer, "16k", cfg.get("sae_release", "gemma-scope-2b-pt-res-canonical"))
            p = _sae_params_path(rel, sid)
            sae_info.update(params_path_tail="/".join(Path(p).parts[-4:]), params_sha256=_sha256(Path(p)))
        except Exception as e:  # noqa: BLE001
            sae_info["params_path_error"] = f"{type(e).__name__}: {e}"
    xc, xk = te.acts(layer, "clean", "from_diff"), te.acts(layer, "corrupt", "from_diff")
    xh_c, e_c, l0_c = decompose_batched(sae, xc)
    xh_k, e_k, l0_k = decompose_batched(sae, xk)
    n_boot = cfg.get("bootstrap", 1000)
    noise_seeds = int(cfg.get("noise_seeds", 1))
    etas = [rescaled_noise(e_c, torch.Generator().manual_seed(cfg["seed"] + 1000 * s + layer)) for s in range(noise_seeds)]

    # ---- pilot U (refit through the pilot function) vs stored
    vram("after SAE load + test decompose")
    U_all = _fit_subspaces(ctx, sae, task, layer, max(cfg["ranks"]), ["unsupervised_svd", "supervised_centroid_svd", "random_subspace"])
    if "U" in ref:
        for k, v in U_all.items():
            sk = k.replace("#", "_seed")
            if sk in ref["U"]:
                cos = col_abs_cos(v[:, :R_EXPORT], ref["U"][sk][:, :R_EXPORT])
                check(checks, f"{task} L{layer}: refit U '{k}' matches the pilot (r<=8)", float(cos.min()) > 0.999,
                      f"min |cos| per column = {cos.min():.6f}; max |dU| up to sign = "
                      f"{np.abs(np.abs(v[:, :R_EXPORT]) - np.abs(ref['U'][sk][:, :R_EXPORT])).max():.3g}")

    # ---- pilot conditions on the full test set through the pilot code
    def nldr_rows(name, vec_list, kind):
        rs = []
        for v in vec_list:
            dl, top1 = run_patched(model, te, layer, v, ctx.device, positions="from_diff")
            rs.append(summarize(dl, top1, te, n_boot))
        agg = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
        row = dict(task=task, layer=layer, condition=name, source=kind, n=len(te), n_seeds=len(rs), **agg)
        if ref:
            if kind == "exp1" and "exp1" in ref:
                q = ref["exp1"].query("task == @task and layer == @layer and width == '16k' and positions == 'from_diff' "
                                      "and intervention == @name")
            elif kind == "exp3" and "exp3" in ref:
                m, r = name.split("@")
                q = ref["exp3"].query("task == @task and layer == @layer and width == '16k' and positions == 'from_diff' "
                                      "and kind == 'hybrid_patch' and method == @m and rank == @int(r)")
            else:
                q = []
            if len(q):
                row.update(pilot_nldr=float(q.iloc[0]["nldr"]), pilot_nldr_lo=float(q.iloc[0]["nldr_lo"]),
                           pilot_nldr_hi=float(q.iloc[0]["nldr_hi"]))
                row["abs_diff_points"] = 100 * abs(row["nldr"] - row["pilot_nldr"])
        repro.append(row)
        return row

    vecs = build_vectors(xh_c, e_c, xh_k, e_k)
    for name in ("sae_only", "error_only", "sae_splice"):
        nldr_rows(name, [vecs[name]], "exp1")
    nldr_rows("noise_control", [xh_k + eta for eta in etas], "exp1")
    nldr_rows("clean_full", [xc], "exp1")
    for m in ("unsupervised_svd", "supervised_centroid_svd", "random_subspace"):
        Us = [v for k, v in U_all.items() if k.split("#")[0] == m]
        for r in (1, 2):
            vs = []
            for Uf in Us:
                U = torch.from_numpy(Uf[:, :r]).float()
                vs.append(xh_c + e_k + ((e_c - e_k) @ U) @ U.T)
            nldr_rows(f"{m}@{r}", vs, "exp3")
    for row in repro:
        if row["task"] == task and "abs_diff_points" in row:
            check(checks, f"{task} L{layer}: NLDR {row['condition']} reproduces the pilot (±1 point)",
                  row["abs_diff_points"] <= 1.0, f"{100 * row['nldr']:.1f} vs pilot {100 * row['pilot_nldr']:.1f}")

    # ---- golden pairs
    vram("after pilot conditions")
    from src.utils import free_memory
    free_memory()
    vram("after gc + empty_cache")
    rng = np.random.default_rng(a.golden_seed)
    gi = np.sort(rng.choice(len(te), size=min(a.n_pairs, len(te)), replace=False))
    golden_ids_out[f"{task}_L{layer}"] = [te.items[i].pair.pair_id for i in gi]
    arrs, meta_pairs = {}, []
    for k, v in U_all.items():
        arrs[f"U/{k.replace('#', '_seed')}"] = v[:, :R_EXPORT].astype(np.float32)
    arrs["U/train_pair_ids"] = np.array([it.pair.pair_id for it in tr.items])
    for j, i in enumerate(gi):
        it = te.items[i]
        r0, r1 = int(te.row_ptr[i]), int(te.row_ptr[i + 1])
        s, T = it.diff_pos, len(it.clean_ids)
        assert r1 - r0 == T - s
        cand = candidate_ids(tok, it, task)
        pre = f"p{j:02d}/"
        arrs[pre + "clean_ids"] = np.array(it.clean_ids, np.int32)
        arrs[pre + "corrupt_ids"] = np.array(it.corrupt_ids, np.int32)
        arrs[pre + "cand_ids"] = np.array([c if c is not None else -1 for c in cand], np.int32)
        # residuals: a fresh batch-1 forward (all positions) and the pilot's cached span rows (what it decomposed)
        dev = ctx.device
        rA = runtime.resid_at(model, torch.tensor([it.clean_ids], device=dev), layer)[0].float().cpu()
        rB = runtime.resid_at(model, torch.tensor([it.corrupt_ids], device=dev), layer)[0].float().cpu()
        arrs[pre + "resid_A"] = rA.numpy()
        arrs[pre + "resid_B_from_diff"] = rB[s:].numpy()
        arrs[pre + "prefix_A_minus_B_maxabs"] = np.float32((rA[:s] - rB[:s]).abs().max() if s else 0.0)
        arrs[pre + "span_A_pilot"] = xc[r0:r1].numpy()
        arrs[pre + "span_B_pilot"] = xk[r0:r1].numpy()
        for tag, X, XH, E in (("A", xc, xh_c, e_c), ("B", xk, xh_k, e_k)):
            z, th = preacts(sae, X[r0:r1])
            rr, cc, vv, act = sparse_z(z, th)
            arrs[pre + f"z_{tag}_row"], arrs[pre + f"z_{tag}_col"] = rr, cc
            arrs[pre + f"z_{tag}_val"], arrs[pre + f"z_{tag}_active"] = vv, act
            arrs[pre + f"xhat_{tag}"], arrs[pre + f"eps_{tag}"] = XH[r0:r1].numpy(), E[r0:r1].numpy()
        for si, eta in enumerate(etas):
            arrs[pre + f"eta_seed{si}"] = eta[r0:r1].numpy()
        # final-position logits
        conds = {"clean_A": None, "clean_B": None}
        sl = slice(r0, r1)
        conds.update({"full_A": xc[sl], "full_B": xk[sl], "feat_A_errB": xh_c[sl] + e_k[sl],
                      "feat_B_errA": xh_k[sl] + e_c[sl], "feat_A_zero": xh_c[sl]})
        for si, eta in enumerate(etas):
            conds[f"feat_B_noise_seed{si}"] = xh_k[sl] + eta[sl]
        for m, short in (("unsupervised_svd", "pca_pooled"), ("supervised_centroid_svd", "answer_sep")):
            for r in (1, 2):
                U = torch.from_numpy(U_all[m][:, :r]).float()
                conds[f"repair_{short}_r{r}"] = xh_c[sl] + e_k[sl] + ((e_c[sl] - e_k[sl]) @ U) @ U.T
        for k in [k for k in U_all if k.startswith("random_subspace")]:
            U = torch.from_numpy(U_all[k][:, :1]).float()
            conds[f"repair_random_r1_{k.split('#')[1]}"] = xh_c[sl] + e_k[sl] + ((e_c[sl] - e_k[sl]) @ U) @ U.T
        pm = {}
        for name, rows in conds.items():
            ids = it.clean_ids if name == "clean_A" else it.corrupt_ids
            lg = patched_last_logits(model, ids, layer, s, rows)
            if name in ("clean_A", "clean_B"):
                arrs[pre + f"logits_full_{name}"] = lg.numpy()
            rec_ = logit_record(lg, cand)
            for kk, vv in rec_.items():
                arrs[pre + f"logits/{name}/{kk}"] = np.asarray(vv)
            pm[name] = float(lg[it.clean_ans] - lg[it.corrupt_ans])
        meta_pairs.append({"slot": j, "test_index": int(i), "pair_id": it.pair.pair_id, "first_diff": s, "T": T,
                           "clean_ans": it.clean_ans, "corrupt_ans": it.corrupt_ans,
                           "state_clean": int(it.pair.clean_state_id), "state_corrupt": int(it.pair.corrupt_state_id),
                           "clean_prompt": it.pair.clean_prompt, "corrupt_prompt": it.pair.corrupt_prompt,
                           "clean_target": it.pair.clean_target_str, "corrupt_target": it.pair.corrupt_target_str,
                           "candidates": CANDIDATES[task], "D_batch1": pm,
                           "D_pilot_batched": {"clean_A": float(te.dl_clean[i]), "clean_B": float(te.dl_corrupt[i])}})
    d1 = max(abs(m["D_batch1"]["clean_A"] - m["D_pilot_batched"]["clean_A"]) for m in meta_pairs)
    check(checks, f"{task} L{layer}: batch-1 forward D matches the pilot's batched D (golden pairs)", d1 < 0.25,
          f"max |dD| = {d1:.4g} (bf16 batching tolerance 0.25)")
    pm_full = max(abs(m["D_batch1"]["full_A"] - m["D_batch1"]["clean_A"]) for m in meta_pairs)
    # the mock model re-reads the state from B's tokens after layer L, so this only holds for a real transformer
    check(checks, f"{task} L{layer}: full_A patch (pilot span rows) reproduces clean A's D",
          None if ctx.mock else pm_full < 0.25, f"max |dD| = {pm_full:.4g}" + (" (mock: not applicable)" if ctx.mock else ""))
    fname = out / f"golden_{task}_L{layer}.npz"
    np.savez_compressed(fname, **arrs)
    info = {"task": task, "layer": layer, "n_train": len(tr), "n_test": len(te), "golden_pairs": meta_pairs, "sae": sae_info,
            "l0_mean_test_spans": [l0_c, l0_k], "noise": {"seeds": noise_seeds, "torch_generator_seed": "cfg.seed + 1000*s + layer",
            "procedure": "src.experiments.common.rescaled_noise(eps_A over all test span rows in Baseline order)"},
            "z_rule": f"z >= {Z_FRAC} * threshold and z > 0 (sparse row/col/val; active = z > threshold)",
            "file": fname.name, "file_mb": round(fname.stat().st_size / 2 ** 20, 1), "seconds": round(time.time() - t0, 1)}
    del sae
    return info, rec.rows(), rec.items()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump_pairs(task: str, splits: dict, scores: dict, scored_items: dict, out: Path) -> dict:
    """All scored candidates (or all tokenized pairs if unfiltered) with their split membership, one JSON line each."""
    kept = {}
    for split, items in splits.items():
        for rank, it in enumerate(items):
            kept[it.pair.pair_id] = (split, rank)
    f = out / f"pairs_{task}.jsonl.gz"
    n = 0
    seen = set()
    with gzip.open(f, "wt") as fh:
        for split, items in splits.items():
            for it in items:
                seen.add(it.pair.pair_id)
        ids = sorted(set(scores) | seen)
        lookup = {**scored_items, **{it.pair.pair_id: it for items in splits.values() for it in items}}
        for pid in ids:
            it = lookup.get(pid)
            row = {"pair_id": pid, "scores": scores.get(pid), "split": kept.get(pid, (None, None))[0],
                   "split_rank": kept.get(pid, (None, None))[1]}
            if it is not None:
                row.update(clean_prompt=it.pair.clean_prompt, corrupt_prompt=it.pair.corrupt_prompt,
                           clean_target=it.pair.clean_target_str, corrupt_target=it.pair.corrupt_target_str,
                           state_clean=int(it.pair.clean_state_id), state_corrupt=int(it.pair.corrupt_state_id),
                           clean_ids=it.clean_ids, corrupt_ids=it.corrupt_ids, clean_ans=it.clean_ans,
                           corrupt_ans=it.corrupt_ans, first_diff=it.diff_pos, meta=it.pair.meta)
            fh.write(json.dumps(row, default=str) + "\n")
            n += 1
    return {"file": f.name, "rows": n, "kept": {s: len(v) for s, v in splits.items()},
            "scored": len(scores)}


# --------------------------------------------------------------------------- main
def main() -> int:
    global log
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/paper_residue.yaml")
    ap.add_argument("--pilot-run", default=PILOT_RUN, help="pilot run dir (preflight overrides + stored outputs)")
    ap.add_argument("--cells", default="entity_track:12,add_chain:18")
    ap.add_argument("--dump-tasks", default=",".join(ALL_TASKS), help="tasks whose candidate pairs are dumped")
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--golden-seed", type=int, default=20261030)
    ap.add_argument("--mock", action="store_true", help="CPU mock model/SAE: tests the plumbing only")
    ap.add_argument("--n-train", type=int, default=None, help="mock only")
    ap.add_argument("--n-test", type=int, default=None, help="mock only")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "export.log")
    t_start = time.time()
    cfg = load_cfg(a)
    set_seed(cfg["seed"])
    device = get_device("cpu" if a.mock else None)
    tok = st.get_tokenizer(a.mock, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
    if not a.mock and type(tok).__name__ == "FallbackTokenizer":   # pilot get_tokenizer falls back silently
        raise SystemExit("HF_TOKEN is not set: the pilot code would use the offline FallbackTokenizer with the "
                         "real model. export HF_TOKEN=hf_... and rerun.")
    from src.models.sae_loader import load_model

    model = load_model(mock=a.mock, device=device, dtype=get_dtype(device), mock_cfg=cfg.get("mock"), tokenizer=tok,
                       model_name=cfg["model_name"], runtime_cfg=cfg.get("runtime"))
    plan = runtime.setup_runtime(cfg, model, device)
    ctx_model_params.extend(list(model.parameters()) + list(model.buffers()))
    ctx = RunContext(model=model, tok=tok, cfg=cfg, mock=a.mock, device=device, out_dir=out, runtime_plan=plan)
    checks, repro, golden_ids, cells, dumps = [], [], {}, {}, {}
    cell_list = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",") if c]
    scores_by_task = {}
    for task, layer in cell_list:
        try:
            info, sc, scored = run_cell(a, ctx, task, layer, out, checks, repro, golden_ids)
        except torch.OutOfMemoryError:
            dump_cuda_tensors()
            raise
        cells[f"{task}_L{layer}"] = info
        scores_by_task[task] = sc
        if task in a.dump_tasks.split(","):
            dumps[task] = dump_pairs(task, {"train": ctx.data[task]["train"].items, "test": ctx.data[task]["test"].items}
                                     if ctx.data.get(task) else {}, sc, scored, out)
        ctx.data = {}
    for task in [t for t in a.dump_tasks.split(",") if t and t not in dumps]:
        c = {**cfg, **(cfg.get("task_overrides") or {}).get(task, {})}
        t0 = time.time()
        with ScoreRecorder() as rec:
            splits = st.build_task_splits(task, tok, c["n_train"], c["n_test"], c["seed"], model=model, mock=a.mock,
                                          candidate_multiplier=c.get("candidate_multiplier", 3),
                                          min_logit_diff=c.get("min_logit_diff", 2.0), n_shots=c.get("n_shots", 0),
                                          batch_size=c["batch_size"], answer_format=c.get("answer_format", "auto"),
                                          filter_mode=c.get("filter_mode", "strict"), gen_kwargs=c.get("gen_kwargs"))
        dumps[task] = dump_pairs(task, splits, rec.rows(), rec.items(), out)
        dumps[task]["seconds"] = round(time.time() - t0, 1)
    import pandas as pd

    pd.DataFrame(repro).to_csv(out / "pilot_reproduction.csv", index=False)
    manifest = {"kind": "golden_fixture", "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "mock": a.mock,
                "args": vars(a), "env": environment(device), "runtime_plan": ctx.runtime_plan,
                "pilot_run": a.pilot_run, "config": a.config, "cfg": cfg, "cells": cells, "golden_pair_ids": golden_ids,
                "pair_dumps": dumps, "checks": checks, "seconds": round(time.time() - t_start, 1),
                "notes": ["logits are post-softcap float32 at the last position (pilot head())",
                          "resid_B_from_diff: positions >= first_diff; earlier positions equal resid_A (see prefix_A_minus_B_maxabs)",
                          "span_*_pilot: the cached Baseline rows the pilot decomposed (model dtype, cast to float32)"]}
    save_json(manifest, out / "manifest.json")
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    lines = [f"# Golden export {'(MOCK: plumbing only) ' if a.mock else ''}{manifest['created']}", "",
             f"git {manifest['env']['git']['head'][:10]} on {manifest['env']['hostname']}; "
             f"{n_fail} FAIL / {len(checks)} checks; {manifest['seconds']:.0f} s", "",
             "| status | check | detail |", "|---|---|---|"]
    lines += [f"| {c['status']} | {c['check']} | {c['detail']} |" for c in checks]
    lines += ["", "## Pilot conditions re-run through the pilot code (all test pairs)", "",
              "| task | layer | condition | NLDR | 95% CI | pilot NLDR | |diff| (points) |", "|---|---|---|---|---|---|---|"]
    for r in repro:
        lines.append(f"| {r['task']} | {r['layer']} | {r['condition']} | {r['nldr']:.3f} | "
                     f"[{r.get('nldr_lo', float('nan')):.3f}, {r.get('nldr_hi', float('nan')):.3f}] | "
                     f"{r.get('pilot_nldr', float('nan')):.3f} | {r.get('abs_diff_points', float('nan')):.2f} |")
    lines += ["", "## Files", ""] + [f"- `{c['file']}`: {c['file_mb']} MB" for c in cells.values() if c]
    lines += [f"- `{d['file']}`: {d['rows']} rows (kept {d['kept']})" for d in dumps.values()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    tar = out.with_suffix(".tar")
    with tarfile.open(tar, "w") as tf:
        tf.add(out, arcname=out.name)
    tar.with_suffix(".tar.sha256").write_text(f"{_sha256(tar)}  {tar.name}\n")
    log.info("wrote %s (%.1f MB), %d FAIL of %d checks", tar, tar.stat().st_size / 2 ** 20, n_fail, len(checks))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

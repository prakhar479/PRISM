#!/usr/bin/env python
"""Preflight on the run machine (handoff §8.1). Writes preflight/<tag>_<stamp>.{json,md} and
preflight/throughput_<tag>_<stamp>.json (used by plan_runs.py). Exit 1 if any of checks 1-5 fails.

    python scripts/run_preflight.py --queue queue/release1.yaml --golden golden/R2 [--profile run] [--skip-smoke]

 1 environment vs lockfile, GPU / driver / free memory, RAM, disk at the cache root
 2 every prefetch target present in the local HF cache with a matching sha256
 3 model load (time, GPU memory) and last-position-only unembedding == full-forward last logits
 4 hook identity on the real model (4 object-tracking pairs, L12): full_B == clean_B, full_A == clean_A
 5 golden comparison (needs R2's golden export): prompt ids, residuals, SAE outputs and per-condition logits
 6 HF top-1 agreement on golden-dump pairs (informational; R3 is the full study)
 7 throughput: forward tokens/s at several batch sizes, SAE pass rate
 8 --smoke runs of E1 (split) and E2
 9 --per-gpu: every visible GPU in its own process: model load, hook identity, the largest token budget that really
   fits (one forward at the planned budget, halved on OOM) and throughput -> <runs_root>/_queue/<tag>/gpus.json,
   which the multi-GPU scheduler reads (budget pinning, failing GPUs dropped). At least one GPU must pass.
(The pilot's scripts/preflight.py, a prompt-format search, is unchanged.)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from harness import device as dv
from harness import tasks as T
from harness.batching import Budget
from harness.cache import Cache
from harness.model import ModelSpec, build_model
from harness.model import stages as S
from harness.profile import REPO_ROOT, load_profile
from harness.queue import Queue
from harness.tolerances import (
    D_BATCHING,
    HEAD_VS_FULL_ULPS,
    HOOK_IDENTITY_D,
    SPAN_RESID_REL,
    head_vs_full_atol,
    hook_identity_atol,
)

GiB = 2 ** 30
CHECKS = []


def check(n, name, ok, detail, **extra):
    CHECKS.append({"n": n, "check": name, "status": "SKIP" if ok is None else ("PASS" if ok else "FAIL"),
                   "detail": detail, **extra})
    print(f"[{CHECKS[-1]['status']}] {n}. {name}: {detail}", flush=True)


def ram_gb():
    try:
        for l in Path("/proc/meminfo").read_text().splitlines():
            if l.startswith("MemAvailable"):
                return round(int(l.split()[1]) / 2 ** 20, 1)
    except OSError:
        return None


def _near(p):
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def c1_env(prof):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_env", REPO_ROOT / "scripts" / "check_env.py")
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    lock = ce.default_lock()
    res = ce.check(ce.parse_lock(lock))
    check(1, f"packages match requirements/{lock.name}", not res["errors"],
          f"{len(res['errors'])} errors, {len(res['warnings'])} warnings", errors=res["errors"], warnings=res["warnings"])
    dev = dv.resolve_device(prof.device)
    info = dv.device_info(dev)
    check(1, "accelerator", dv.is_cuda(dev) if prof.device.startswith("cuda") else True,
          f"{info.get('gpu_name', info['device'])}, free {info.get('gpu_free_gb')} / {info.get('gpu_total_gb')} GB", info=info)
    cache = prof.root("cache_root")
    cache.mkdir(parents=True, exist_ok=True)
    tier_free = {str(d): shutil.disk_usage(_near(d)).free / GiB for d in [cache, *prof.spill_roots()]}
    free = sum(tier_free.values())                      # the cache spills across tiers, so their free space adds up
    runs = prof.root("runs_root")
    runs.mkdir(parents=True, exist_ok=True)
    rfree = shutil.disk_usage(runs).free / GiB
    check(1, "RAM and disk", free > prof.min_free_gb and rfree > 2,
          f"RAM available {ram_gb()} GB; disk free over cache tiers {free:.0f} GB "
          f"({', '.join(f'{k}: {v:.0f}' for k, v in tier_free.items())}; need > {prof.min_free_gb:g}); "
          f"at runs root {rfree:.0f} GB (need > 2)")
    return dev


def c2_hf(queue, tag):
    out = REPO_ROOT / "preflight" / f"{tag}_prefetch_verify.json"
    rc = subprocess.run([sys.executable, "scripts/run_prefetch.py", "--queue", queue, "--verify-only", "--out", str(out)],
                        cwd=REPO_ROOT, capture_output=True, text=True).returncode
    rep = json.loads(out.read_text()) if out.exists() else {"errors": ["no report"], "files": []}
    check(2, "prefetch targets cached with matching sha256", rc == 0,
          f"{len(rep['files'])} files, {rep.get('total_gb')} GB, {len(rep['errors'])} errors", errors=rep["errors"][:10])


def c3_model(prof, dev, revision):
    from src.models import runtime
    t0 = time.time()
    dv.reset_peak(dev)
    spec = ModelSpec(revision=revision, dtype=prof.model_dtype)
    model = build_model(spec, dev, offload_embed=prof.offload_embed)
    secs = time.time() - t0
    mem = dv.allocated_bytes(dev) / GiB
    toks = torch.tensor([[2, 651, 6037, 576, 6081, 603]], device=dev)      # "<bos>The capital of France is"
    with torch.inference_mode():
        a = runtime.last_logits(model, toks)
        inp, kw = runtime._input(model, toks)
        full = model.run_with_hooks(inp, fwd_hooks=[], tokens=toks, **kw)[:, -1].float()
    d = float((a - full).abs().max())
    tol = head_vs_full_atol(float(full.abs().max()), prof.model_dtype)
    same_top = bool((a.argmax(-1) == full.argmax(-1)).all())
    check(3, "model load", True, f"{secs:.0f} s, {mem:.2f} GB allocated", seconds=secs, gpu_gb=mem)
    check(3, "last-position head == full forward last logits", d <= tol and same_top,
          f"max |dlogit| {d:.3g} (tol {tol:.3g} = {HEAD_VS_FULL_ULPS} {prof.model_dtype} ulp at max |logit| "
          f"{float(full.abs().max()):.1f}); argmax equal {same_top}")
    return model, spec


def c4_hooks(model, spec, prof, dev):
    rows, meta = T.load_set(prof.root("data_root") / "prompt_sets" / "entity_track_pilot.jsonl.gz")
    sub = [r for r in rows if r["split"] == "test"][:4]
    cache = prof.cache()
    b = Budget(prof.max_batch_tokens, prof.max_batch_size)
    ent = S.stage1(model, spec, sub, T.set_hash(sub), [12], cache, b, dev)
    cl = S.load_clean(ent["clean_out"], len(sub))
    sp = S.load_spans(ent["spans"][12], sub, list(range(len(sub))))
    o = S.stage3(model, sub, list(range(len(sub))), 12, sp.row_ptr, {"clean_B": None, "full_B": sp.x_B, "full_A": sp.x_A}, b, dev)
    D = lambda x: x["logit_A"] - x["logit_B"]
    dB = float(np.abs(D(o["full_B"]) - D(cl["B"])).max())
    dA = float(np.abs(D(o["full_A"]) - D(cl["A"])).max())
    dR = float(np.abs(D(o["clean_B"]) - D(cl["B"])).max())
    tol = hook_identity_atol(prof.model_dtype)
    check(4, "hook identity (OT L12, 4 pairs): full_B == clean_B, resumed clean_B == stage 1",
          max(dB, dR) <= tol, f"max |dD| full_B {dB:.2g}, resumed clean_B {dR:.2g}")
    check(4, "full_A reproduces clean A (NLDR = 1)", dA <= tol, f"max |dD| {dA:.2g}")


def c5_golden(model, spec, prof, dev, golden):
    if not golden or not (Path(golden) / "manifest.json").exists():
        check(5, "golden comparison", False, f"no golden export at {golden} (R2 must be returned and unpacked first)")
        return
    from harness import conditions as C
    from harness.sae import SAEId, decompose, load_params, params_path
    man = json.loads((Path(golden) / "manifest.json").read_text())
    inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())
    cache = prof.cache()
    for cell, info in man["cells"].items():
        task, layer = info["task"], info["layer"]
        z = np.load(Path(golden) / info["file"])
        rows, _ = T.load_set(prof.root("data_root") / "prompt_sets" / f"{task}_pilot.jsonl.gz")
        by_id = {r["pair_id"]: r for r in rows}
        gp = info["golden_pairs"]
        sub = [by_id.get(g["pair_id"]) for g in gp]
        same = all(r is not None and r["clean_ids"] == z[f"p{k:02d}/clean_ids"].tolist()
                   and r["corrupt_ids"] == z[f"p{k:02d}/corrupt_ids"].tolist() for k, r in enumerate(sub))
        check(5, f"{cell}: golden pair ids and token ids == harness prompt set", same, f"{len(gp)} pairs")
        if not same:
            continue
        b = Budget(prof.max_batch_tokens, prof.max_batch_size)
        ent = S.stage1(model, spec, sub, T.set_hash(sub), [layer], cache, b, dev)
        sp = S.load_spans(ent["spans"][layer], sub, list(range(len(sub))))
        gA = torch.cat([torch.from_numpy(z[f"p{k:02d}/span_A_pilot"]) for k in range(len(sub))])
        relrows = (sp.x_A - gA).norm(dim=-1) / gA.norm(dim=-1)
        rel = float(relrows.max())
        # the pilot's own batching spread: its batch-1 forward (resid_A, all positions) vs its batched cached rows
        g1 = torch.cat([torch.from_numpy(z[f"p{k:02d}/resid_A"])[gp[k]["first_diff"]:] for k in range(len(sub))])
        pil = float(((g1 - gA).norm(dim=-1) / gA.norm(dim=-1)).max())
        check(5, f"{cell}: span residuals vs golden (relative)", rel <= SPAN_RESID_REL,
              f"max relative row error {rel:.3g} (median {float(relrows.median()):.3g}; tol {SPAN_RESID_REL}); "
              f"pilot's own batch-1 vs batched: max {pil:.3g}", rel_max=rel, pilot_batch_spread=pil)
        from harness.sae import canonical_l0
        sid = SAEId(layer, "16k", canonical_l0(inv["inventory"][str(layer)]["16k"]), revision=inv["revision"])
        tail = info["sae"].get("params_path_tail", "")
        check(5, f"{cell}: SAE variant == the one the golden export loaded", (sid.path_in_repo in tail) if tail else None,
              f"harness {sid.path_in_repo}; golden {tail or 'not recorded'}")
        sae = load_params(params_path(sid), layer, "16k").to_device(dev)
        xh, _, _ = decompose(sae, gA)
        gxh = torch.cat([torch.from_numpy(z[f"p{k:02d}/xhat_A"]) for k in range(len(sub))])
        d = float((xh.cpu() - gxh).abs().max())
        check(5, f"{cell}: SAE x_hat on golden residuals", d < 1e-3, f"max |dx_hat| {d:.3g} (GPU fp32 vs golden)")
        g = lambda key: torch.cat([torch.from_numpy(z[f"p{k:02d}/{key}"]) for k in range(len(sub))])
        sp.x_A, sp.x_B = gA, g("span_B_pilot")
        sp.xh_A, sp.xh_B, sp.e_A, sp.e_B = g("xhat_A"), g("xhat_B"), g("eps_A"), g("eps_B")
        U = torch.from_numpy(z["U/unsupervised_svd"])
        conds = {"feat_A_errB": C.build("feat_A_errB", sp), "feat_B_errA": C.build("feat_B_errA", sp),
                 "feat_A_zero": C.build("feat_A_zero", sp), "full_A": sp.x_A.clone(),
                 "repair_pca_pooled_r1": C.build("repair", sp, U=U[:, :1]),
                 "repair_pca_pooled_r2": C.build("repair", sp, U=U[:, :2])}
        o = S.stage3(model, sub, list(range(len(sub))), layer, sp.row_ptr, conds, b, dev)
        worst = {}
        for name, out in o.items():
            gD = np.array([gp[k]["D_batch1"][name] for k in range(len(sub))])
            worst[name] = float(np.abs((out["logit_A"] - out["logit_B"]) - gD).max())
        ok = max(worst.values()) <= D_BATCHING.atol
        check(5, f"{cell}: per-condition D vs golden", ok, "max |dD| " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()),
              worst=worst)
        del sae
        dv.empty_cache()


def c7_throughput(model, prof, dev, tag, stamp):
    from src.models import runtime
    rows, _ = T.load_set(prof.root("data_root") / "prompt_sets" / "entity_track_pilot.jsonl.gz")
    res = {"device": str(dev), "seq_len": rows[0]["T"], "by_rows": {}}
    for n in (1, 2, 4, max(1, prof.max_batch_tokens // rows[0]["T"])):
        t = torch.tensor([r["clean_ids"] for r in rows[:n]], device=dev)
        with torch.inference_mode():
            runtime.last_logits(model, t)
            dv.sync(dev)
            t0 = time.time()
            for _ in range(3):
                runtime.last_logits(model, t)
            dv.sync(dev)
        res["by_rows"][n] = round(3 * n * t.shape[1] / (time.time() - t0), 1)
    res["forward_tokens_per_s"] = max(res["by_rows"].values())
    out = REPO_ROOT / "preflight" / f"throughput_{tag}_{stamp}.json"
    out.write_text(json.dumps(res, indent=1))
    check(7, "throughput", True, f"forward tokens/s by rows {res['by_rows']} -> {out.name}")
    return res


def budget_probe(model, prof, dev, rows) -> dict:
    """Largest max_batch_tokens (from the profile's, halving on OOM, floor 1024) whose forward fits on this GPU."""
    from src.models import runtime
    T = rows[0]["T"]
    tokens, tried = int(prof.max_batch_tokens), []
    while tokens >= 1024:
        n = max(1, min(int(prof.max_batch_size), tokens // T))
        t = torch.tensor([rows[i % len(rows)]["clean_ids"] for i in range(n)], device=dev)
        dv.empty_cache()
        dv.reset_peak(dev)
        try:
            with torch.inference_mode():
                runtime.last_logits(model, t)
                dv.sync(dev)
                t0 = time.time()
                for _ in range(3):
                    runtime.last_logits(model, t)
                dv.sync(dev)
            peak = dv.peak_bytes(dev)
            tried.append({"tokens": tokens, "rows": n, "ok": True, "peak_gb": round(peak / GiB, 2)})
            return {"budget_ok": tokens, "rows": n, "tokens_per_s": round(3 * n * T / (time.time() - t0), 1),
                    "peak_gb": round(peak / GiB, 2), "tried": tried}
        except Exception as e:  # noqa: BLE001
            if not dv.is_oom(e):
                raise
            tried.append({"tokens": tokens, "rows": n, "ok": False})
            tokens //= 2
    return {"budget_ok": None, "tried": tried}


def gpu_probe(a, prof, revision) -> int:
    """One GPU (CUDA_VISIBLE_DEVICES set by the parent): prints one 'GPUPROBE {json}' line."""
    out = {"cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        dev = dv.resolve_device(prof.device)
        out["memory"] = dv.device_memory(dev)
        dv.cap_memory(dev, prof.gpu_mem_cap_gb)
        spec = ModelSpec(revision=revision, dtype=prof.model_dtype)
        t0 = time.time()
        model = build_model(spec, dev, offload_embed=prof.offload_embed)
        out["load_s"] = round(time.time() - t0, 1)
        c4_hooks(model, spec, prof, dev)
        out["hooks_ok"] = all(c["status"] == "PASS" for c in CHECKS if c["n"] == 4)
        rows, _ = T.load_set(prof.root("data_root") / "prompt_sets" / "entity_track_pilot.jsonl.gz")
        out.update(budget_probe(model, prof, dev, rows))
        out["ok"] = bool(out["hooks_ok"] and out.get("budget_ok"))
    except Exception as e:  # noqa: BLE001
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    print("GPUPROBE " + json.dumps(out, default=str), flush=True)
    return 0 if out["ok"] else 1


def per_gpu(a, prof, q, probe) -> dict:
    import os
    gpus = [str(g["index"]) for g in probe.get("gpus") or []]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip():
        gpus = [v.strip() for v in vis.split(",") if v.strip()]
    res = {}
    for g in gpus:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g, "HL_CONCURRENT_WORKERS": "1"}
        r = subprocess.run([sys.executable, "scripts/run_preflight.py", "--queue", a.queue, "--profile", prof.name,
                            "--gpu-probe"], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("GPUPROBE ")]
        res[g] = json.loads(line[-1][9:]) if line else {"ok": False, "error": (r.stdout + r.stderr)[-400:]}
        d = res[g]
        check(9, f"GPU {g}", d.get("ok"), f"budget {d.get('budget_ok')} tokens x {d.get('rows')} rows, peak "
              f"{d.get('peak_gb')} GB, {d.get('tokens_per_s')} tok/s" if d.get("ok") else d.get("error", "hook check failed"))
    ok = {g: d["budget_ok"] for g, d in res.items() if d.get("ok")}
    state = prof.root("runs_root") / "_queue" / q.tag
    state.mkdir(parents=True, exist_ok=True)
    (state / "gpus.json").write_text(json.dumps({"budget_ok": ok, "gpus_failed": sorted(set(res) - set(ok)),
                                                 "per_gpu": res}, indent=1, default=str))
    check(3, "per-GPU checks: at least one GPU usable", bool(ok) if res else None,
          f"{len(ok)}/{len(res)} GPUs passed -> {state.relative_to(REPO_ROOT) if state.is_relative_to(REPO_ROOT) else state}/gpus.json")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--hf-pairs", type=int, default=24)
    ap.add_argument("--per-gpu", action="store_true", help="check every visible GPU in its own process (check 9)")
    ap.add_argument("--gpu-probe", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    q = Queue.load(a.queue)
    prof = load_profile(a.profile or q.profile)
    prof.apply_env()
    from harness.memory import resolve_resources
    probe = dv.probe_machine(prof.root("cache_root"))
    try:
        dev_mem = dv.device_memory(dv.resolve_device(prof.device))
    except RuntimeError:
        dev_mem = None
    prof, resources = resolve_resources(prof, probe, dev_mem)          # `auto` values -> measured numbers
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (REPO_ROOT / "preflight").mkdir(exist_ok=True)
    t0 = time.time()
    from harness.config import ExpConfig
    revision = ExpConfig.load(REPO_ROOT / q.configs()[0]).model_revision
    if a.gpu_probe:
        return gpu_probe(a, prof, revision)
    print(f"resources: {json.dumps(resources.get('resolved'))} ({resources.get('rules')})", flush=True)
    model = None
    try:
        dev = c1_env(prof)
        c2_hf(a.queue, q.tag)
        model, spec = c3_model(prof, dev, revision)
        c4_hooks(model, spec, prof, dev)
        c5_golden(model, spec, prof, dev, a.golden)
        c7_throughput(model, prof, dev, q.tag, stamp)
    except Exception as e:  # noqa: BLE001
        check(0, "preflight crashed", False, f"{type(e).__name__}: {e}")
    model = None
    dv.empty_cache()
    if a.golden and (Path(a.golden) / "manifest.json").exists():
        out = REPO_ROOT / "preflight" / f"{q.tag}_{stamp}_hf"
        rc = subprocess.run([sys.executable, "scripts/hf_agreement.py", "--pairs-dir", a.golden, "--out", str(out),
                             "--max-per-task", str(a.hf_pairs), "--fp32-subset", "0"], cwd=REPO_ROOT, capture_output=True).returncode
        summ = json.loads((out / "manifest.json").read_text())["summary"] if (out / "manifest.json").exists() else []
        agree = {f"{s['task']}/{s['prompt']}": round(s["top1_agree"], 3) for s in summ}
        check(6, "HF top-1 agreement (informational)", None if rc == 0 else False, f"{agree} (see {out.name}/report.md)")
    else:
        check(6, "HF top-1 agreement", None, "needs the golden pair dumps")
    if a.per_gpu:
        per_gpu(a, prof, q, probe)
    if not a.skip_smoke:
        for cfg in ("configs/experiments/E1_split.yaml", "configs/experiments/E2.yaml"):
            rc = subprocess.run([sys.executable, "-m", "harness.run", "--config", cfg, "--profile", prof.name, "--smoke"],
                                cwd=REPO_ROOT, capture_output=True, text=True)
            check(8, f"smoke {Path(cfg).stem}", rc.returncode == 0, "ok" if rc.returncode == 0 else rc.stdout[-400:] + rc.stderr[-400:])
    failed = [c for c in CHECKS if c["status"] == "FAIL" and c["n"] <= 5]
    rep = {"tag": q.tag, "stamp": stamp, "seconds": round(time.time() - t0, 1), "checks": CHECKS, "ok": not failed,
           "profile": prof.name}
    base = REPO_ROOT / "preflight" / f"{q.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    md = [f"# Preflight {q.tag} {stamp}: {'OK' if not failed else 'FAILED'}", "", "| # | status | check | detail |", "|---|---|---|---|"]
    md += [f"| {c['n']} | {c['status']} | {c['check']} | {str(c['detail']).replace('|', '/')} |" for c in CHECKS]
    base.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

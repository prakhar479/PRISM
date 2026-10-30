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
from harness.tolerances import D_BATCHING, HOOK_IDENTITY_D

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


def c1_env(prof):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_env", REPO_ROOT / "scripts" / "check_env.py")
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    res = ce.check(ce.parse_lock(REPO_ROOT / "requirements" / "harness.lock"))
    check(1, "packages match requirements/harness.lock", not res["errors"],
          f"{len(res['errors'])} errors, {len(res['warnings'])} warnings", errors=res["errors"], warnings=res["warnings"])
    dev = dv.resolve_device(prof.device)
    info = dv.device_info(dev)
    check(1, "accelerator", dv.is_cuda(dev) if prof.device.startswith("cuda") else True,
          f"{info.get('gpu_name', info['device'])}, free {info.get('gpu_free_gb')} / {info.get('gpu_total_gb')} GB", info=info)
    cache = prof.root("cache_root")
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / GiB
    check(1, "RAM and disk", free > 20, f"RAM available {ram_gb()} GB; disk free at cache root {free:.0f} GB")
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
    check(3, "model load", True, f"{secs:.0f} s, {mem:.2f} GB allocated", seconds=secs, gpu_gb=mem)
    check(3, "last-position head == full forward last logits", d < 1e-2, f"max |dlogit| {d:.3g}")
    return model, spec


def c4_hooks(model, spec, prof, dev):
    rows, meta = T.load_set(prof.root("data_root") / "prompt_sets" / "entity_track_pilot.jsonl.gz")
    sub = [r for r in rows if r["split"] == "test"][:4]
    cache = Cache(prof.root("cache_root"))
    b = Budget(prof.max_batch_tokens, prof.max_batch_size)
    ent = S.stage1(model, spec, sub, T.set_hash(sub), [12], cache, b, dev)
    cl = S.load_clean(ent["clean_out"], len(sub))
    sp = S.load_spans(ent["spans"][12], sub, list(range(len(sub))))
    o = S.stage3(model, sub, list(range(len(sub))), 12, sp.row_ptr, {"clean_B": None, "full_B": sp.x_B, "full_A": sp.x_A}, b, dev)
    D = lambda x: x["logit_A"] - x["logit_B"]
    dB = float(np.abs(D(o["full_B"]) - D(cl["B"])).max())
    dA = float(np.abs(D(o["full_A"]) - D(cl["A"])).max())
    dR = float(np.abs(D(o["clean_B"]) - D(cl["B"])).max())
    check(4, "hook identity (OT L12, 4 pairs): full_B == clean_B, resumed clean_B == stage 1",
          max(dB, dR) <= HOOK_IDENTITY_D.atol, f"max |dD| full_B {dB:.2g}, resumed clean_B {dR:.2g}")
    check(4, "full_A reproduces clean A (NLDR = 1)", dA <= HOOK_IDENTITY_D.atol, f"max |dD| {dA:.2g}")


def c5_golden(model, spec, prof, dev, golden):
    if not golden or not (Path(golden) / "manifest.json").exists():
        check(5, "golden comparison", False, f"no golden export at {golden} (R2 must be returned and unpacked first)")
        return
    from harness import conditions as C
    from harness.sae import SAEId, decompose, load_params, params_path
    man = json.loads((Path(golden) / "manifest.json").read_text())
    inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())
    cache = Cache(prof.root("cache_root"))
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
        rel = float(((sp.x_A - gA).norm(dim=-1) / gA.norm(dim=-1)).max())
        check(5, f"{cell}: span residuals vs golden (relative)", rel < 2e-2, f"max relative row error {rel:.3g}")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--hf-pairs", type=int, default=24)
    a = ap.parse_args()
    q = Queue.load(a.queue)
    prof = load_profile(a.profile or q.profile)
    prof.apply_env()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (REPO_ROOT / "preflight").mkdir(exist_ok=True)
    t0 = time.time()
    from harness.config import ExpConfig
    revision = ExpConfig.load(REPO_ROOT / q.configs()[0]).model_revision
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

#!/usr/bin/env python
"""Print the plan for a queue on a profile: per job the estimated peak GPU memory (vs the profile cap), SAE
placement and, when a preflight throughput measurement exists, the estimated wall time. Exit 2 if a job cannot fit.

    python scripts/plan_runs.py --queue queue/release1.yaml --profile run [--throughput preflight/throughput.json]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from harness.config import ExpConfig
from harness.profile import REPO_ROOT, load_profile
from harness.queue import Queue

N_LAYERS = 26


def latest_throughput(path):
    if path:
        return json.loads(Path(path).read_text())
    cands = sorted((REPO_ROOT / "preflight").glob("throughput*.json")) + sorted((REPO_ROOT / "incoming").glob("**/throughput*.json"))
    return json.loads(cands[-1].read_text()) if cands else None


def estimate_seconds(R, tp) -> dict:
    """Rough time model: forward tokens / measured tokens-per-second (bf16, model on GPU)."""
    if not tp:
        return {}
    tps = tp.get("forward_tokens_per_s")
    if not tps:
        return {}
    s1 = sum(2 * sum(r["T"] for r in R.rows[t]) for t in R.cfg.layers_by_task())
    s3 = 0.0
    for c in R.cfg.cells:
        T = max(r["T"] for r in R.rows[c.task])
        n = R.n_test * len(R.cfg.filters)
        n_cond = len(R.condition_specs_count(c))
        s3 += n * T * (c.layer + 1) / N_LAYERS + n_cond * n * T * (N_LAYERS - c.layer - 1) / N_LAYERS
    return {"stage1_min": round(s1 / tps / 60, 1), "stage3_min": round(s3 / tps / 60, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--profile", default="run")
    ap.add_argument("--throughput", default=None)
    a = ap.parse_args()
    from harness.run import Runner

    q = Queue.load(a.queue)
    prof = load_profile(a.profile)
    tp = latest_throughput(a.throughput)
    ok, total = True, 0.0
    print(f"queue {q.tag} on profile {prof.name} (GPU cap {prof.gpu_mem_cap_gb} GB); throughput: "
          f"{tp.get('forward_tokens_per_s') if tp else 'none (no time estimates)'}")
    for job in q.jobs:
        if job.kind != "experiment":
            print(f"- {job.id}: script {job.script}")
            continue
        cfg = ExpConfig.load(REPO_ROOT / job.config)
        R = Runner(cfg, prof, SimpleNamespace(tiny=False, smoke="--smoke" in job.args, dry_run=True, config=job.config,
                                              allow_dirty=False))
        p = R.plan()
        t = estimate_seconds(R, tp)
        total += sum(t.values())
        peak = max(j["peak_gb"] for j in p["jobs"])
        print(f"- {job.id}: {cfg.exp_id} {cfg.name}: {len(cfg.cells)} cells, peak {peak:.2f} GB, "
              f"{'fits' if p['all_fit'] else 'DOES NOT FIT'}" + (f", ~{t['stage1_min']} + {t['stage3_min']} min" if t else ""))
        for j in p["jobs"]:
            if not j["fits"] or j["stage"] == 2:
                print(f"    stage {j['stage']}: {j['what']} -> {j.get('placement', '')} {j['peak_gb']} GB"
                      f"{'' if j['fits'] else '  <-- exceeds cap'}")
        ok &= p["all_fit"]
    if tp:
        print(f"estimated total (model stages only): ~{total / 60:.1f} h")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

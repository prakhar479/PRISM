#!/usr/bin/env python
"""E13 runner: base-model behaviour on the language prompt sets (harness/behaviour.py).

    python scripts/run_behaviour.py --langs hi,fr                         # real model, one GPU: all three tasks, test split
    python scripts/run_behaviour.py --tiny --tasks add_chain --langs en --limit 40    # tiny random model (plumbing test)
    python scripts/run_behaviour.py --dry-run

Reads data/prompt_sets/<task>_e13.jsonl.gz (scripts/build_lang_tasks.py). Writes <runs_root>/E13/<run_hash>/{behaviour.csv,
behaviour_pairs.parquet, manifest.json}. Every row says whether its template pack is human-validated (false until a native
speaker signs); gates A and B are in behaviour.csv (harness/behaviour.py).
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from harness import provenance
from harness.profile import REPO_ROOT, load_profile

MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TASKS = ("entity_track", "add_chain", "people_cities")
log = logging.getLogger("harness")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=os.environ.get("HL_PROFILE", "server"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--langs", default="en,hi,hi_latn,hinglish,fr")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="first N candidates (rank order) per family")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    from harness import langfamilies as LF
    from harness import tasks as T
    prof = load_profile(a.profile)
    prof.apply_env()
    tasks, langs = a.tasks.split(","), a.langs.split(",")
    data_root = prof.root("data_root") / "prompt_sets"
    sets = {t: T.load_set(data_root / f"{t}_e13.jsonl.gz") for t in tasks}
    fams = [(t, LF.family_id(t, l), l) for t in tasks for l in langs]
    missing = [f for t, f, l in fams if f not in sets[t][1]["families"]]
    if missing:
        raise SystemExit(f"families not in the prompt sets: {missing}")
    ident = {"exp": "E13", "model": MODEL_REVISION, "split": a.split, "limit": a.limit, "tiny": a.tiny,
             "sets": {t: sets[t][1]["hash"] for t in tasks}, "families": [f for _, f, _ in fams]}
    run_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = (REPO_ROOT / "tests" / "_tiny_runs" if a.tiny else prof.root("runs_root")) / "E13" / run_hash
    if a.dry_run:
        for t, f, l in fams:
            n = sum(1 for r in sets[t][0] if r["family"] == f and r["split"] == a.split)
            print(f"{t:14s} {f:12s} {min(n, a.limit or n)} candidates ({a.split})")
        return 0
    git = provenance.git_state()
    if git["dirty"] and not (a.allow_dirty or a.tiny):
        raise SystemExit("working tree is dirty: commit, or pass --allow-dirty\n  " + "\n  ".join(git["dirty_files"][:10]))
    run_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from harness import behaviour as B
    from harness import memory
    from harness.batching import Budget
    from harness.device import device_memory, probe_machine, resolve_device, set_threads
    from harness.model import ModelSpec, build_model

    device = torch.device("cpu") if a.tiny else resolve_device(prof.device)
    probe = probe_machine(prof.root("cache_root"))
    prof, resources = memory.resolve_resources(prof, probe, None if a.tiny else device_memory(device))
    set_threads(prof.cpu_threads if isinstance(prof.cpu_threads, int) else None)
    spec = ModelSpec(name="tiny", dtype="float32", tiny_vocab=512, tiny_layers=2) if a.tiny else \
        ModelSpec(revision=MODEL_REVISION, dtype=prof.model_dtype)
    budget = Budget(prof.max_batch_tokens if not a.tiny else 4096, prof.max_batch_size if not a.tiny else 8)
    csv_path = run_dir / "behaviour.csv"
    done = {(r["task"], r["family"]) for r in csv.DictReader(csv_path.open())
            if r.get("task") and r.get("family") and r.get("gate_A_viable") in ("True", "False")} if csv_path.exists() else set()
    todo = [x for x in fams if (x[0], x[1]) not in done]
    log.info("E13: %d families to do, %d done; device %s", len(todo), len(done), device)
    t_start = time.time()
    table = []
    if todo:
        model = build_model(spec, device, offload_embed=prof.offload_embed if not a.tiny else False)
        for task, fid, lang in todo:
            rows = sorted((r for r in sets[task][0] if r["family"] == fid and r["split"] == a.split), key=lambda r: r["rank"])
            rows = rows[:a.limit] if a.limit else rows
            if a.tiny:
                rows = [{**r, "clean_ids": [i % 509 for i in r["clean_ids"]], "corrupt_ids": [i % 509 for i in r["corrupt_ids"]],
                         "ans_A": r["ans_A"] % 509, "ans_B": r["ans_B"] % 509} for r in rows]
            t0 = time.time()
            s = B.score_rows(model, rows, budget, device)
            summ = B.summarize(task, rows, s)
            meta = sets[task][1]["families"][fid]
            row = {"task": task, "family": fid, "lang": lang, "script": meta["script"], "validated": LF.is_validated(fid), "pack_hash": LF.pack_hash(fid),
                   "split": a.split, "length_matching_success": meta["length_matching_success"][a.split], **summ}
            pairs = pd.DataFrame({"pair_id": [r["pair_id"] for r in rows], "family": fid, "rank": [r["rank"] for r in rows],
                                  "ans_A": [r["ans_A"] for r in rows], "ans_B": [r["ans_B"] for r in rows], **s})
            pairs.to_parquet(run_dir / f"behaviour_pairs_{task}_{fid}.parquet")
            new = not csv_path.exists()
            with csv_path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            log.info("%s %s: acc %.3f keep(clean_only) %.3f viable %s (%.0fs)", task, fid, row["acc_clean"], row["keep_clean_only"],
                     row["gate_A_viable"], time.time() - t0)
    allrows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    status = "complete" if len(allrows) == len(fams) else "partial"
    (run_dir / "manifest.json").write_text(json.dumps({
        "exp_id": "E13", "run_hash": run_hash, "status": status, "table": "behaviour.csv", "smoke": False, "tiny": a.tiny,
        "config": {"name": "behaviour_by_language", "identity": ident, "gates": {"A": "acc>=0.6 and clean_only keep>=0.3",
                                                                                "B": "|acc - English| <= 0.10 (applied in make_e13)"}},
        "profile": prof.to_dict(), "git": git, "packages": provenance.packages(), "machine": provenance.machine(device),
        "model": spec.key(), "sae_revision": None, "prompt_sets": {t: {"hash": sets[t][1]["hash"], "file": sets[t][1]["file"], "n": sets[t][1]["n"]} for t in tasks},
        "seeds": {}, "timings": {"total_s": round(time.time() - t_start, 1)}, "wall_time_s": round(time.time() - t_start, 1),
        "created": provenance.now(), "resources": {k: v for k, v in resources.items() if k != "probe"}, "budget_backoffs": budget.backoffs},
        indent=1, default=str))
    log.info("E13 %s -> %s (%s)", run_hash, run_dir, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Tier 0: find a prompt setting under which the base model actually solves each task, BEFORE long compute.

For every task (at the difficulty its config gen_kwargs set) it prints the tokenization of a sample prompt +
target and scores a grid (answer_format x n_shots) on a fresh probe set: tokenizable fraction, clean/corrupt
top-1, strict and clean_only filter retention, mean logit diffs. Per task it picks the setting that keeps the
most pairs (ties within 0.02 -> fewer shots), with the strict filter if that keeps >= --min-retention, else
clean_only, and writes it as task_overrides to <out>/preflight.json (consumed by run_full_suite.py --overrides).
Tasks keeping < --drop-below are listed in drop_tasks.

Gate: ioi_control is the positive control. If its best setting keeps < --ioi-gate of pairs, the model or the
pipeline is broken (the 2026-10-19 run scored IOI at 0% and burned the night), so the script exits 3 and the
launcher stops. preflight.json carries a fingerprint of the task code and config; `--check FILE` exits 0 only
if FILE still matches, so a stale preflight is never reused.

Exit codes: 0 ok, 1 no usable task, 3 IOI gate failed (or --check: 1 = stale / missing).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.synthetic_tasks import (_with_format, generate_pairs, generate_with_shots, get_tokenizer, score_pairs,
                                      tokenize_pairs, valid_mask)
from src.experiments.common import task_cfg
from src.models.runtime import setup_runtime
from src.models.sae_loader import load_model
from src.utils import get_device, get_dtype, load_config, load_json, save_json, setup_logging

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_FILES = ["src/data/synthetic_tasks.py", "scripts/preflight.py"]


def fingerprint(cfg: dict, mock: bool) -> str:
    """Hash of the task code + the config fields that change what preflight would choose."""
    h = hashlib.sha1()
    for f in FINGERPRINT_FILES:
        h.update((ROOT / f).read_bytes())
    keys = ("model_name", "tasks", "task_overrides", "seed", "min_logit_diff")
    h.update(json.dumps({**{k: cfg.get(k) for k in keys}, "mock": mock}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def choose(results: list, min_retention: float):
    """Most pairs kept by clean_only (to 0.02), then fewer shots; strict filter when it keeps enough."""
    best = max(results, key=lambda r: (round(r["clean_only"] / 0.02), -r["n_shots"]))
    mode = "strict" if best["strict"] >= min_retention else "clean_only"
    return best, mode, best[mode]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="results/preflight")
    ap.add_argument("--n-probe", type=int, default=96)
    ap.add_argument("--shots", default="0,2,4,8")
    ap.add_argument("--formats", default="auto", help="answer formats to try (auto picks per pair)")
    ap.add_argument("--min-retention", type=float, default=0.3, help="use the strict filter if it keeps this much")
    ap.add_argument("--drop-below", type=float, default=0.1, help="drop a task whose best setting keeps less")
    ap.add_argument("--ioi-gate", type=float, default=0.7, help="abort if the IOI control keeps less than this")
    ap.add_argument("--check", default=None, help="exit 0 iff this preflight.json matches the current code/config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.check:
        p = Path(a.check)
        ok = p.exists() and load_json(p).get("fingerprint") == fingerprint(cfg, a.mock)
        print(f"preflight {'current' if ok else 'stale or missing'}: {p}")
        return 0 if ok else 1
    out = Path(a.out)
    log = setup_logging(out / "preflight.log")
    device = get_device("cpu" if a.mock else None)
    tok = get_tokenizer(a.mock, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
    model = load_model(mock=a.mock, device=device, dtype=get_dtype(device), mock_cfg=cfg.get("mock"), tokenizer=tok,
                       model_name=cfg["model_name"], runtime_cfg=cfg.get("runtime"))
    setup_runtime(cfg, model, device)
    margin = cfg.get("min_logit_diff", 2.0)
    formats = a.formats.split(",")
    grid, overrides, drop, best_keep = [], {}, [], {}
    t0 = time.time()
    for task in cfg["tasks"]:
        gk = task_cfg(cfg, task).get("gen_kwargs")
        sample = generate_pairs(task, 1, cfg["seed"] + 4242, split="preflight", gen_kwargs=gk)[0]
        for fmt in ("space_target", "space_prompt"):
            p = _with_format(sample, fmt)
            ids = tok.encode(p.clean_prompt)
            tgt = tok.encode(p.clean_prompt + p.clean_target_str)[len(ids):]
            log.info("[%s/%s] prompt tail tokens %s | target %r -> %s",
                     task, fmt, [tok.decode([i]) for i in ids[-6:]], p.clean_target_str, [tok.decode([i]) for i in tgt])
        results = []
        for fmt in formats:
            for shots in [int(x) for x in a.shots.split(",")]:
                pairs = generate_with_shots(task, a.n_probe, shots, cfg["seed"] + 4242, cfg["seed"] + 77,
                                            split="preflight", gen_kwargs=gk)
                items = tokenize_pairs(tok, pairs, fmt)
                row = dict(task=task, answer_format=fmt, n_shots=shots, n_pairs=len(pairs),
                           tokenized=len(items) / max(len(pairs), 1))
                if items:
                    sc = score_pairs(model, items, cfg["batch_size"])
                    f = len(items) / len(pairs)
                    row.update(clean_top1=float(sc["clean_top1"].mean()), corrupt_top1=float(sc["corrupt_top1"].mean()),
                               dl_clean=float(sc["dl_clean"].mean()), dl_corrupt=float(sc["dl_corrupt"].mean()),
                               strict=f * float(valid_mask(sc, "strict", margin).mean()),
                               clean_only=f * float(valid_mask(sc, "clean_only", margin).mean()))
                else:
                    row.update(strict=0.0, clean_only=0.0)
                log.info("preflight %s", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
                results.append(row)
                grid.append(row)
        best, mode, ret = choose(results, a.min_retention)
        best_keep[task] = best["clean_only"]
        if best["clean_only"] < a.drop_below:
            log.warning("%s: best setting keeps only %.2f of pairs -> dropped", task, best["clean_only"])
            drop.append(task)
            continue
        overrides[task] = dict(n_shots=best["n_shots"], answer_format=best["answer_format"], filter_mode=mode,
                               candidate_multiplier=int(min(25, max(2, math.ceil(1.4 / ret)))),
                               expected_retention=round(ret, 3))
        log.info("==> %s: %s", task, overrides[task])
    gate = None
    if "ioi_control" in cfg["tasks"]:
        gate = {"ioi_best_keep": round(best_keep["ioi_control"], 3), "threshold": a.ioi_gate,
                "passed": bool(best_keep["ioi_control"] >= a.ioi_gate)}
        if not gate["passed"]:
            log.error("IOI GATE FAILED: the positive control keeps only %.2f of pairs (< %.2f). gemma-2-2b solves "
                      "IOI almost perfectly, so the model or the pipeline is broken: run scripts/check_forward.py",
                      best_keep["ioi_control"], a.ioi_gate)
    save_json({"task_overrides": overrides, "drop_tasks": drop, "ioi_gate": gate, "fingerprint": fingerprint(cfg, a.mock),
               "grid": grid, "seconds": time.time() - t0}, out / "preflight.json")
    print(f"\npreflight -> {out / 'preflight.json'}\n  overrides: {overrides}\n  dropped: {drop}\n  ioi gate: {gate}")
    if gate is not None and not gate["passed"]:
        return 3
    return 0 if overrides else 1


if __name__ == "__main__":
    sys.exit(main())

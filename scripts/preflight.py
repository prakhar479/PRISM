#!/usr/bin/env python
"""Tier 0: find a prompt setting under which the base model actually solves each task, BEFORE long compute.

For every task it prints the tokenization of a sample prompt + target and scores a grid
(answer_format x n_shots) on a fresh probe set: tokenizable fraction, clean/corrupt top-1, strict and
clean_only filter retention, mean logit diffs. The cheapest setting with enough retention is written as
task_overrides to <out>/preflight.json (consumed by run_full_suite.py --overrides). Tasks the model cannot
do at all are listed in drop_tasks.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.data.synthetic_tasks import (add_few_shot_prefix, generate_pairs, get_tokenizer, score_pairs, tokenize_pairs,
                                      valid_mask)
from src.models.sae_loader import load_model
from src.utils import get_device, get_dtype, load_config, save_json, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="results/preflight")
    ap.add_argument("--n-probe", type=int, default=96)
    ap.add_argument("--shots", default="0,2,4,8")
    ap.add_argument("--formats", default="space_target,space_prompt")
    ap.add_argument("--min-retention", type=float, default=0.3, help="strict retention considered 'good enough'")
    a = ap.parse_args()
    out = Path(a.out)
    log = setup_logging(out / "preflight.log")
    cfg = load_config(a.config)
    device = get_device("cpu" if a.mock else None)
    tok = get_tokenizer(a.mock, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
    model = load_model(mock=a.mock, device=device, dtype=get_dtype(device), mock_cfg=cfg.get("mock"), tokenizer=tok,
                       model_name=cfg["model_name"])
    margin = cfg.get("min_logit_diff", 2.0)
    grid, overrides, drop = [], {}, []
    t0 = time.time()
    for task in cfg["tasks"]:
        sample = generate_pairs(task, 1, cfg["seed"] + 4242, split="preflight")[0]
        for fmt in a.formats.split(","):
            from src.data.synthetic_tasks import _with_format
            p = _with_format(sample, fmt)
            ids = tok.encode(p.clean_prompt)
            tgt = tok.encode(p.clean_prompt + p.clean_target_str)[len(ids):]
            log.info("[%s/%s] prompt tail tokens %s | target %r -> %s",
                     task, fmt, [tok.decode([i]) for i in ids[-6:]], p.clean_target_str, [tok.decode([i]) for i in tgt])
        results = []
        for fmt in a.formats.split(","):
            for shots in [int(x) for x in a.shots.split(",")]:
                pairs = generate_pairs(task, a.n_probe, cfg["seed"] + 4242, split="preflight")
                add_few_shot_prefix(pairs, shots, cfg["seed"] + 77)
                items = tokenize_pairs(tok, pairs, fmt)
                row = dict(task=task, answer_format=fmt, n_shots=shots, n_pairs=len(pairs), tokenized=len(items) / max(len(pairs), 1))
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
        good = [r for r in results if r["strict"] >= a.min_retention]
        if good:
            best, mode, ret = min(good, key=lambda r: (r["n_shots"], -r["strict"])), "strict", None
            ret = best["strict"]
        else:
            best = max(results, key=lambda r: max(r["strict"], 0.5 * r["clean_only"]))
            mode = "strict" if best["strict"] >= 0.5 * best["clean_only"] else "clean_only"
            ret = best[mode]
        if ret < 0.05:
            log.warning("%s: model fails the task under every setting (best retention %.2f) -> dropped", task, ret)
            drop.append(task)
            continue
        mult = int(min(25, max(3, math.ceil(1.4 / ret))))
        overrides[task] = dict(n_shots=best["n_shots"], answer_format=best["answer_format"], filter_mode=mode,
                               candidate_multiplier=mult, expected_retention=round(ret, 3))
        log.info("==> %s: %s", task, overrides[task])
    save_json({"task_overrides": overrides, "drop_tasks": drop, "grid": grid, "seconds": time.time() - t0},
              out / "preflight.json")
    print(f"\npreflight -> {out / 'preflight.json'}\n  overrides: {overrides}\n  dropped: {drop}")
    return 0 if overrides else 1


if __name__ == "__main__":
    sys.exit(main())

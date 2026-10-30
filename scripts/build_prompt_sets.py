#!/usr/bin/env python
"""Stage 0: build prompt-set files (no model needed; real Gemma tokenizer required unless --fallback-tokenizer).

    python scripts/build_prompt_sets.py --pilot                       # the four pilot pools -> data/prompt_sets/
    python scripts/build_prompt_sets.py --pilot --tasks entity_track --out /tmp/ps

Each set is <out>/<task>_<family-set>.jsonl.gz with a .meta.json holding the content hash, settings and tokenizer
revision. Runs refuse a set whose hash does not match its meta file. Filter flags are added later from Stage 1.
"""
import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness import tasks as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="the pilot's candidate pools (paper_residue + preflight)")
    ap.add_argument("--tasks", default=",".join(T.PILOT_TASKS))
    ap.add_argument("--out", default=None, help="default: <data_root>/prompt_sets")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--fallback-tokenizer", action="store_true", help="offline stand-in tokenizer (tests only)")
    a = ap.parse_args()
    if not a.pilot:
        ap.error("only --pilot sets exist in release 1 (fresh families arrive with E3)")
    from harness.profile import load_profile

    prof = load_profile(a.profile)
    out = Path(a.out) if a.out else prof.root("data_root") / "prompt_sets"
    if a.fallback_tokenizer:
        from src.data.synthetic_tasks import FallbackTokenizer
        tok, tok_info = FallbackTokenizer(), {"name": "FallbackTokenizer"}
    else:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        rev = a.tokenizer_revision or HfApi().model_info("google/gemma-2-2b").sha
        tok, tok_info = AutoTokenizer.from_pretrained("google/gemma-2-2b", revision=rev), {"name": "google/gemma-2-2b", "revision": rev}
    cfg = T.pilot_cfg()
    for task in a.tasks.split(","):
        t0 = time.time()
        rows = T.build_pilot_candidates(task, tok, cfg)
        bad = T.check_invariants(rows)
        if bad:
            print(f"{task}: {len(bad)} invariant violations, e.g. {bad[:3]}")
            return 1
        info = {"task": task, "kind": "pilot_candidates", "settings": T.task_settings(cfg, task), "tokenizer": tok_info,
                "source": "harness.tasks.build_pilot_candidates (pilot generators, unfiltered, pilot order)",
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        p = T.save_set(rows, out / f"{task}_pilot.jsonl.gz", info)
        print(f"{task}: {len(rows)} candidates -> {p} ({p.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

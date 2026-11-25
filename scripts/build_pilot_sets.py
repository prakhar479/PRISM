#!/usr/bin/env python
"""Stage 0 for the breadth pilot (docs/PILOT_BREADTH_PLAN.md): prompt sets of the counterfactual variants in
harness/pilotfamilies.py, built like the E3/E13 sets (one file per task, every family's own few-shot prefix, prompt-disjoint
train/test pools, tokenizer checks: equal A/B length, single-token answers).

    python scripts/build_pilot_sets.py                     # -> data/prompt_sets/people_cities_p1.jsonl.gz, entity_track_p11.jsonl.gz
    python scripts/build_pilot_sets.py --tasks people_cities --n-train 40 --n-test 20 --out /tmp/ps

people_cities (P1): pc_en (content), pc_en_swap (binding swap), pc_en_query (query swap); E13's people-cities settings (8 shots,
depth 3, x3 candidates). entity_track (P11): ot_placed_dest, ot_bin_dest, ot_drawer_dest; the pilot's object-tracking settings
(depth 2, x6 candidates). add_chain (P8 only): rs_en, rs_plus, rs_story. Needs the real tokenizer (local HF cache is enough).
P8's Gemma 3 sets: --tokenizer google/gemma-3-1b-pt --tokenizer-revision <rev> --suffix _g3 --tasks people_cities,entity_track,add_chain
Batch 2: --tasks ioi_control (the pilot generator, family "pilot") -> ioi_control_x1[_g3].jsonl.gz
"""
import argparse
import copy
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness import pilotfamilies as PF
from harness import tasks as T

MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
SETS = {"people_cities": ("people_cities_p1", PF.P1_FAMILIES, {"n_shots": 8, "candidate_multiplier": 3, "gen_kwargs": {"depth": 3}}),
        "entity_track": ("entity_track_p11", PF.P11_FAMILIES, {"candidate_multiplier": 6}),
        # P8 (second model): running sum from the fresh / English families (the pilot's own generator is Gemma-2-specific)
        "add_chain": ("add_chain_p8", ("rs_en", "rs_plus", "rs_story"), {"candidate_multiplier": 3}),
        # batch 2 (atlas): IOI through the pilot's own generator (no template families): Gemma 3's token ids
        "ioi_control": ("ioi_control_x1", ("pilot",), {})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="people_cities,entity_track", help="any of people_cities, entity_track, add_chain")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default="google/gemma-2-2b")
    ap.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    ap.add_argument("--suffix", default="", help="appended to the file stem (e.g. _g3 for a Gemma 3 tokenizer, P8)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from harness.profile import load_profile
    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision)
    out = Path(a.out) if a.out else load_profile("dev").root("data_root") / "prompt_sets"
    for task in a.tasks.split(","):
        stem, fids, over = SETS[task]
        cfg = copy.deepcopy(T.pilot_cfg())
        cfg.setdefault("task_overrides", {})[task] = {**cfg.get("task_overrides", {}).get(task, {}), **over}
        t0 = time.time()
        rows, fams = [], {}
        for fid in fids:
            if fid == "pilot":
                fr = T.build_pilot_candidates(task, tok, cfg, n_train=a.n_train, n_test=a.n_test)
            else:
                fr = T.build_family_candidates(task, fid, tok, cfg, n_train=a.n_train, n_test=a.n_test)
            mult = T.task_settings(cfg, task)["candidate_multiplier"] or 3
            gen = {s: int(n * mult) for s, n in (("train", a.n_train), ("test", a.n_test))}
            kept = {s: sum(r["split"] == s for r in fr) for s in gen}
            fams[fid] = {"generated": gen, "kept": kept, "length_matching_success": {s: round(kept[s] / gen[s], 4) for s in gen}}
            rows += fr
        bad = T.check_invariants(rows)
        if bad:
            print(f"{task}: {len(bad)} invariant violations, e.g. {bad[:3]}")
            return 1
        info = {"task": task, "kind": "pilot_counterfactual_families", "settings": T.task_settings(cfg, task),
                "tokenizer": {"name": a.tokenizer, "revision": a.tokenizer_revision}, "families": fams,
                "source": "harness.tasks.build_family_candidates over harness.pilotfamilies (pilot tasks: build_pilot_candidates)",
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        p = T.save_set(rows, out / f"{stem}{a.suffix}.jsonl.gz", info)
        print(f"{task}: {len(rows)} candidates -> {p} ({p.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)")
        for fid, f in fams.items():
            print(f"   {fid:16s} kept {f['kept']}  length-matching success {f['length_matching_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

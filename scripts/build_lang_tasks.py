#!/usr/bin/env python
"""Stage 0 for E13: prompt sets of the language families (harness/langfamilies.py), one file per task holding every language's
family, like the E3 sets: data/prompt_sets/<task>_e13.jsonl.gz (+ .meta.json with per-family counts, language, validation stamp).

    python scripts/build_lang_tasks.py                       # entity_track, add_chain, people_cities; 150 train + 150 test per family
    python scripts/build_lang_tasks.py --tasks people_cities --n-train 50 --n-test 50 --out /tmp/ps

Needs the real Gemma tokenizer (local HF cache is enough). Generation uses the pilot's settings for OT (depth 2, x6 candidates)
and RS (depth 3, x3), the same 8 shots per family for people_cities (depth 3, x3). "Length-matching success" = candidates kept
after the tokenizer checks (equal A/B length, single-token answers) / candidates generated, recorded per family and split.
Every row of a family whose pack is not human-validated carries meta.validated = false.
"""
import argparse
import copy
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness import families as F
from harness import langfamilies as LF
from harness import tasks as T

MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
E13_SETTINGS = {"people_cities": {"n_shots": 8, "candidate_multiplier": 3, "gen_kwargs": {"depth": 3}}}
MULT = {"entity_track": 6, "add_chain": 3, "people_cities": 3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="entity_track,add_chain,people_cities")
    ap.add_argument("--langs", default=",".join(LF.LANGUAGES))
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="{task}_e13", help="file stem; {task} is filled in (E11 uses people_cities_en)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from harness.profile import load_profile
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", revision=MODEL_REVISION)
    out = Path(a.out) if a.out else load_profile("dev").root("data_root") / "prompt_sets"
    cfg = copy.deepcopy(T.pilot_cfg())
    for t, ov in E13_SETTINGS.items():
        cfg.setdefault("task_overrides", {})[t] = {**cfg["task_overrides"].get(t, {}), **ov}
    for task in a.tasks.split(","):
        t0 = time.time()
        rows, fams = [], {}
        for lang in a.langs.split(","):
            fid = LF.family_id(task, lang)
            fr = T.build_family_candidates(task, fid, tok, cfg, n_train=a.n_train, n_test=a.n_test)
            for r in fr:
                r["meta"] = {**r["meta"], "lang": lang, "validated": bool(LF.VALIDATED[fid])}
            gen = {s: int(n * MULT[task]) for s, n in (("train", a.n_train), ("test", a.n_test))}
            kept = {s: sum(r["split"] == s for r in fr) for s in gen}
            fams[fid] = {"lang": lang, "script": LF.LANG_OF[fid][1], "validated": bool(LF.VALIDATED[fid]), "generated": gen,
                         "kept": kept, "length_matching_success": {s: round(kept[s] / gen[s], 4) for s in gen}}
            rows += fr
        bad = T.check_invariants(rows)
        if bad:
            print(f"{task}: {len(bad)} invariant violations, e.g. {bad[:3]}")
            return 1
        info = {"task": task, "kind": "language_families", "settings": T.task_settings(cfg, task),
                "tokenizer": {"name": "google/gemma-2-2b", "revision": MODEL_REVISION}, "families": fams,
                "source": "harness.tasks.build_family_candidates over harness.langfamilies (DRAFT templates, unvalidated)",
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        p = T.save_set(rows, out / f"{a.name.format(task=task)}.jsonl.gz", info)
        print(f"{task}: {len(rows)} candidates -> {p} ({p.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)")
        for fid, f in fams.items():
            print(f"   {fid:12s} kept {f['kept']}  length-matching success {f['length_matching_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""tables/E13_behaviour_lang.csv from ingested E13 runs (runs/E13/*/behaviour.csv). DEV only; no model.

    python scripts/make_e13.py [--runs runs] [--out tables]

One row per (task, family, split; the gates of record are on the test split, the train split is an exploratory check): clean/corrupt accuracy with Wilson CIs, filter keep-rates, length-matching success, `validated`
(false until a native speaker signs the pack), Gate A (`gate_A_viable`: accuracy >= 0.6 and clean_only keep >= 0.3, the pilot's
VIABLE rule) and Gate B (`gate_B_within_10pts`: |accuracy - English| <= 0.10 on the same task, the preregistered rule). The gates
are separate columns and neither is a verdict on the research question.
"""
import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from harness import behaviour as B
from harness.profile import REPO_ROOT, load_profile

NUM = ("n", "acc_clean", "acc_corrupt", "keep_f_A_correct", "keep_f_both_correct", "keep_pilot_rule", "keep_clean_only",
       "median_D_clean", "median_D_corrupt", "chance_acc", "length_matching_success")


def gather(runs_root: Path):
    rows = {}
    for f in sorted(runs_root.glob("E13/*/behaviour.csv"), key=lambda p: p.stat().st_mtime):
        for r in csv.DictReader(f.open()):
            r["run_hash"] = f.parent.name
            rows[(r["task"], r["family"], r.get("split", "test"))] = r
    return list(rows.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "tables"))
    a = ap.parse_args()
    runs_root = Path(a.runs) if a.runs else load_profile("dev").root("runs_root")
    rows = gather(runs_root)
    if not rows:
        print(f"no E13 runs under {runs_root}/E13")
        return 1
    for r in rows:
        for k in NUM:
            r[k] = float(r[k])
        r["gate_A_viable"] = r["gate_A_viable"] == "True"
    order = {"entity_track": 0, "add_chain": 1, "people_cities": 2}
    rows.sort(key=lambda r: (r.get("split", "test") != "test", order[r["task"]], ["en", "hi", "hi_latn", "hinglish", "fr"].index(r["lang"])))
    B.add_gate_b(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "E13_behaviour_lang.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out / 'E13_behaviour_lang.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""DEV: verify and merge a *published* results directory (results_server/<tag>/, as pushed on a results/* branch) into a runs
root. The directory form of ingest_results.py (which takes a bundle tarball): same checks, same log.

    python scripts/ingest_published.py <results_server/<tag>> --runs <runs_root> [--allow-untagged] [--dry-run]

Checks: every file listed in PUBLISHED.json exists with its sha256; manifest and per_pair schemas; the run's commit carries a
run-* tag in this repository (or, with --allow-untagged, is a descendant of one: the gap is logged per run); no tiny/smoke runs;
a run directory that already exists must be byte-identical. Queue state goes to <runs_root>/_queue/<tag>/.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from harness import provenance
from harness.schema import validate_manifest, validate_per_pair
from ingest_results import sha256, tagged


def nearest_tag(commit: str) -> str:
    try:
        return provenance.git("describe", "--tags", "--match", "run-*", commit).strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("published", help="results_server/<tag> directory containing PUBLISHED.json")
    ap.add_argument("--runs", required=True, help="runs root to merge into")
    ap.add_argument("--allow-untagged", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    top, runs_root = Path(a.published), Path(a.runs)
    pub = json.loads((top / "PUBLISHED.json").read_text())
    errors, notes = [], []
    for rel, info in pub["files"].items():
        f = top / rel
        if not f.exists() or sha256(f) != info["sha256"]:
            errors.append(f"checksum mismatch or missing: {rel}")
    run_dirs = sorted({(top / rel).parent for rel in pub["files"] if rel.startswith("runs/E") and rel.endswith("/manifest.json")})
    for d in run_dirs:
        m = json.loads((d / "manifest.json").read_text())
        errs = validate_manifest(m)
        if m.get("smoke") or m.get("tiny"):
            errs.append("smoke/tiny run")
        c = m["git"]["commit"]
        if not tagged(c):
            if a.allow_untagged:
                notes.append(f"{d.relative_to(top)}: commit {c[:10]} untagged (describe: {nearest_tag(c) or 'none'})")
            else:
                errs.append(f"commit {c[:10]} has no run-* tag here")
        if m["git"].get("dirty"):
            errs.append("dirty tree")
        if (d / "per_pair.parquet").exists():
            errs += validate_per_pair(pd.read_parquet(d / "per_pair.parquet"))
        elif m["status"] == "complete" and m.get("table"):
            if not (d / m["table"]).exists():
                errs.append(f"complete run without {m['table']}")
        elif m["status"] == "complete":
            errs.append("complete run without per_pair.parquet")
        errors += [f"{d.relative_to(top)}: {e}" for e in errs]
    if errors:
        print("REFUSED:\n  " + "\n  ".join(errors[:50]))
        return 1
    merged, skipped = [], []
    for d in run_dirs:
        rel = d.relative_to(top / "runs")
        dest = runs_root / rel
        if dest.exists():
            same = all((dest / f.relative_to(d)).exists() and sha256(dest / f.relative_to(d)) == sha256(f)
                       for f in d.rglob("*") if f.is_file())
            if not same:
                errors.append(f"duplicate run {rel} with different content")
            skipped.append(str(rel))
            continue
        merged.append(str(rel))
        if not a.dry_run:
            shutil.copytree(d, dest)
    q = top / "runs" / "_queue" / pub["tag"]
    if q.exists() and not a.dry_run:
        shutil.copytree(q, runs_root / "_queue" / pub["tag"], dirs_exist_ok=True)
        if (top / "preflight").exists():
            shutil.copytree(top / "preflight", runs_root / "_queue" / pub["tag"] / "preflight", dirs_exist_ok=True)
    if not a.dry_run:
        with open(runs_root / "_ingested.jsonl", "a") as f:
            f.write(json.dumps({"published": pub["tag"], "host": pub.get("hostname"), "commit": pub.get("git_commit"),
                                "time": provenance.now(), "merged": merged, "skipped": skipped, "errors": errors,
                                "allow_untagged": a.allow_untagged, "notes": notes}) + "\n")
    print(f"{'(dry run) ' if a.dry_run else ''}merged {len(merged)}, already present {len(skipped)}, errors {len(errors)}")
    for n in sorted(set(x.split(': ', 1)[1] for x in notes)):
        print("  note:", n)
    if errors:
        print("  " + "\n  ".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

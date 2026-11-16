#!/usr/bin/env python
"""Compact, read-only inventory of a runs/ tree (what a results branch holds), for checking before any analysis.

    python scripts/inventory_runs.py --runs <dir>/runs [--json out.json]

Per run directory: finalized or not (manifest.json / per_pair.parquet), the commits and statuses recorded by every partial
manifest (stage 1, stage-2 slices, stage-3 workers), smoke/tiny flags, stage-3 units done vs claimed vs expected (from the
worker logs and parts/), skipped SAEs, and the queue's per-unit status. Prints a short report; --json writes everything.
Never opens per-pair data beyond its row count, so the output stays small enough to read in full.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def run_info(d: Path) -> dict:
    parts = d / "parts"
    done = sorted(p.name for p in parts.glob("*.done")) if parts.exists() else []
    claims = sorted(p.name for p in parts.glob("*.claim")) if parts.exists() else []
    pq = [p.name for p in parts.glob("*.parquet")] if parts.exists() else []
    cells = collections.Counter(re.sub(r"__\d+\.done$", "", n) for n in done)
    partial = {}
    for m in sorted(d.glob("manifest.*.json")):
        j = load(m) or {}
        partial[m.name] = {"status": j.get("status"), "commit": (j.get("git") or {}).get("commit", "")[:12],
                           "dirty": (j.get("git") or {}).get("dirty"), "smoke": j.get("smoke"), "tiny": j.get("tiny"),
                           "stages": j.get("stages"), "wall_s": j.get("wall_time_s"), "created": j.get("created"),
                           "machine": (j.get("machine") or {}).get("gpu_name") or (j.get("machine") or {}).get("device"),
                           "backoffs": (j.get("budget") or {}).get("backoffs")}
    final = load(d / "manifest.json")
    info = {"exp": d.parent.name, "run_hash": d.name, "finalized": final is not None,
            "final_status": final and final.get("status"), "per_pair": (d / "per_pair.parquet").exists(),
            "summary": (d / "summary.json").exists(), "units_done": len(done), "claims_left": len(claims),
            "part_parquets": len(pq), "cells_with_units": len(cells), "units_per_cell": sorted(set(cells.values())),
            "skipped_saes": load(d / "skipped_saes.json"), "partial_manifests": partial,
            "commits": sorted({v["commit"] for v in partial.values() if v["commit"]}),
            "any_dirty": any(v["dirty"] for v in partial.values()) or bool(final and (final.get("git") or {}).get("dirty")),
            "any_smoke": any(v["smoke"] for v in partial.values()) or bool(final and final.get("smoke")),
            "tiny": bool(final and final.get("tiny")) or any(v["tiny"] for v in partial.values()),
            "final_commit": final and (final.get("git") or {}).get("commit", "")[:12],
            "statuses": dict(collections.Counter(v["status"] for v in partial.values()))}
    logs = "\n".join(p.read_text(errors="replace")[-4000:] for p in sorted(d.glob("log.*.txt")))
    info["log_errors"] = sorted(set(re.findall(r"(?:Error|Traceback|FAILED|missing)[^\n]{0,140}", logs)))[:12]
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = Path(a.runs)
    runs = [run_info(d) for d in sorted(root.glob("E*/*")) if d.is_dir()]
    queues = {}
    for q in sorted((root / "_queue").glob("*/status.json")):
        st = load(q) or {}
        units = st.get("units", st)
        by = collections.defaultdict(list)
        for uid, u in (units.items() if isinstance(units, dict) else []):
            if isinstance(u, dict):
                by[u.get("status")].append(f"{uid}({u.get('attempts')})")
        queues[q.parent.name] = {k: v for k, v in by.items()}
    out = {"runs": runs, "queues": queues}
    for r in runs:
        print(f"{r['exp']}/{r['run_hash']}: finalized={r['finalized']} ({r['final_status']}) per_pair={r['per_pair']} "
              f"units_done={r['units_done']} cells={r['cells_with_units']} units/cell={r['units_per_cell']} claims_left={r['claims_left']} "
              f"commits={r['commits'] or [r['final_commit']]} dirty={r['any_dirty']} smoke={r['any_smoke']} partial={r['statuses']} skipped={bool(r['skipped_saes'])}")
        for e in r["log_errors"][:4]:
            print("    log:", e)
    for name, by in queues.items():
        print(f"queue {name}: " + "; ".join(f"{k}: {len(v)}" for k, v in by.items()))
        for k, v in by.items():
            if k not in ("done",):
                print(f"   {k}: {', '.join(v)[:400]}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

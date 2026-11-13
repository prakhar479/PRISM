#!/usr/bin/env python
"""Copy the finished, small results of a queue into one directory that can be committed and pushed as it is:

    python scripts/publish_results.py --tag release2-server [--out results_server/<tag>] [--all-commits]

Contents (same files as the bundle, never the parts/ intermediates or any cache): per run the manifest.json,
summary.json/csv, per_pair.parquet, skipped_saes.json and logs; the queue state (queue_status.md, status.json,
machine.json, gpus.json, the unit logs); the preflight reports. Files over 90 MB are split into .part-NN pieces (cat
them back together). PUBLISHED.json lists every file with its sha256 and the source commit. The directory is git-ignored
(so the run checkout stays clean while it exists): commit it with `git add -f`, see README.md.
"""
import argparse
import hashlib
import json
import shutil
import socket
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from bundle_results import collect, sha256
from harness import provenance
from harness.profile import REPO_ROOT, load_profile

PART = 90 * 2 ** 20


def publish(prof, tag: str, out: Path, all_commits: bool = False) -> dict:
    files, runs, head = collect(prof, tag, all_commits)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    listing = {}
    for src, arc in files:
        dst = out / arc
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest, size = sha256(src), src.stat().st_size
        if size > PART:
            with open(src, "rb") as f:
                for i, chunk in enumerate(iter(lambda: f.read(PART), b"")):
                    (dst.parent / f"{dst.name}.part-{i:02d}").write_bytes(chunk)
            listing[str(arc)] = {"sha256": digest, "bytes": size, "split": True}
        else:
            shutil.copy2(src, dst)
            listing[str(arc)] = {"sha256": digest, "bytes": size}
    meta = {"tag": tag, "created": provenance.now(), "hostname": socket.gethostname(), "git_commit": head,
            "git_tags": provenance.git_state()["tags"], "runs": runs, "files": listing,
            "note": "split files: cat <file>.part-* > <file>, then check the sha256 listed here"}
    (out / "PUBLISHED.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-commits", action="store_true")
    a = ap.parse_args()
    prof = load_profile(a.profile)
    out = Path(a.out) if a.out else REPO_ROOT / "results_server" / a.tag
    meta = publish(prof, a.tag, out, a.all_commits)
    total = sum(v["bytes"] for v in meta["files"].values())
    print(f"published {len(meta['files'])} files ({total / 2**20:.1f} MiB), {len(meta['runs'])} runs -> {out}")
    return 0 if meta["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())

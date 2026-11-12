#!/usr/bin/env python
"""RUN -> DEV results bundle: run directories (manifest, per_pair.parquet, summaries, logs, directions), preflight
reports and the queue state, never caches. Writes bundles/<tag>_<stamp>.tar.gz + .sha256 with BUNDLE_MANIFEST.json.

    python scripts/bundle_results.py --tag release1 [--all-commits]
By default only runs made at the current HEAD commit are included.
"""
import argparse
import hashlib
import io
import json
import socket
import sys
import tarfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness import provenance
from harness.profile import REPO_ROOT, load_profile

SKIP_DIRS = {"parts"}                      # per_pair.parquet already holds every row


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def collect(prof, tag: str, all_commits: bool):
    runs_root = prof.root("runs_root")
    head = provenance.git_state()["commit"]
    files, runs = [], []
    for man in sorted(runs_root.glob("E*/*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("tiny") or (not all_commits and m["git"]["commit"] != head):
            continue
        d = man.parent
        runs.append({"exp_id": m["exp_id"], "run_hash": m["run_hash"], "status": m["status"], "smoke": m["smoke"],
                     "commit": m["git"]["commit"]})
        for f in d.rglob("*"):
            if f.is_file() and not (set(f.relative_to(d).parts) & SKIP_DIRS):
                files.append((f, Path("runs") / f.relative_to(runs_root)))
    q = runs_root / "_queue" / tag
    if q.exists():
        files += [(f, Path("runs") / f.relative_to(runs_root)) for f in q.rglob("*") if f.is_file()]
    pf = REPO_ROOT / "preflight"
    if pf.exists():
        files += [(f, Path("preflight") / f.relative_to(pf)) for f in pf.rglob("*") if f.is_file()]
    return files, runs, head


def default_out_dir(prof) -> Path:
    """<repo>/bundles, or <runs_root>/../bundles when runs_root lives outside the repo (a scratch disk): bundles are
    as big as the runs, so they go on the same disk."""
    runs = prof.root("runs_root").resolve()
    return REPO_ROOT / "bundles" if runs.is_relative_to(REPO_ROOT) else runs.parent / "bundles"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--all-commits", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    prof = load_profile(a.profile)
    files, runs, head = collect(prof, a.tag, a.all_commits)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out_dir) if a.out_dir else default_out_dir(prof)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{a.tag}_{stamp}"
    manifest = {"bundle": name, "tag": a.tag, "created": provenance.now(), "hostname": socket.gethostname(),
                "git_commit": head, "git_tags": provenance.git_state()["tags"], "runs": runs,
                "files": {str(arc): {"sha256": sha256(f), "bytes": f.stat().st_size} for f, arc in files}}
    tar = out_dir / f"{name}.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        data = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo(f"{name}/BUNDLE_MANIFEST.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        for f, arc in files:
            tf.add(f, arcname=f"{name}/{arc}")
    (out_dir / f"{name}.tar.gz.sha256").write_text(f"{sha256(tar)}  {tar.name}\n")
    print(f"{len(runs)} runs, {len(files)} files -> {tar} ({tar.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

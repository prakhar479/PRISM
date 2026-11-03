#!/usr/bin/env python
"""DEV: build the clean `run` branch (runtime files only) from the current commit, for GPU servers to clone.

    python scripts/make_run_branch.py --golden-tar incoming/R2/R2.tar [--tag run-YYYYMMDD-N] [--branch run]
    python scripts/make_run_branch.py --no-golden --dry-run           # list what would go in, change nothing

The run tree is the ALLOW list below taken from HEAD (tracked files only), README_RUN.md as README.md, a
RUN_SOURCE.json naming the source commit, and the golden fixtures split into < 95 MB parts (golden_parts/), which
scripts/setup_server.sh reassembles and verifies. It is built with git plumbing (a temporary index + commit-tree), so
the working tree and the current branch are untouched and .gitignore does not apply. The commit's parent is the
previous tip of the branch (the first one has none); the commit and the tag are GPG-signed. Nothing is pushed.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = [
    "harness/", "src/", "configs/", "data/", "profiles/", "queue/", "requirements/", "requirements.txt", ".gitignore",
    "results/paper_20261026_001341/preflight/preflight.json",
    "releases/release2-server.md",
    "scripts/_bootstrap.py", "scripts/run_queue.py", "scripts/run_prefetch.py", "scripts/run_preflight.py",
    "scripts/check_env.py", "scripts/hf_agreement.py", "scripts/bundle_results.py", "scripts/plan_runs.py",
    "scripts/probe_env.py", "scripts/setup_server.sh",
]
RENAME = {"README_RUN.md": "README.md"}
PART = 90 * 2 ** 20


def git(*args, input=None, env=None) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], input=input, capture_output=True,
                       env={**os.environ, **(env or {})}, text=not isinstance(input, bytes))
    if r.returncode != 0:
        err = r.stderr if isinstance(r.stderr, str) else r.stderr.decode()
        raise SystemExit(f"git {' '.join(args)} failed: {err.strip()}")
    return r.stdout if isinstance(r.stdout, str) else r.stdout.decode()


def blob(data: bytes) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "hash-object", "-w", "--stdin"], input=data, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode())
    return r.stdout.decode().strip()


def tracked(src: str):
    """(mode, sha, path) of every file of HEAD under the allow list."""
    out = git("ls-tree", "-r", "-l", src, "--", *ALLOW, *RENAME)
    rows = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, sha, size = meta.split()
        if typ == "blob":
            rows.append((mode, sha, path, int(size)))
    return rows


def next_tag() -> str:
    day = dt.date.today().strftime("%Y%m%d")
    have = set(git("tag", "--list", f"run-{day}-*").split())
    n = 1
    while f"run-{day}-{n}" in have:
        n += 1
    return f"run-{day}-{n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-tar", default=None, help="the R2 golden export tar (golden/R2.tar from the run machine)")
    ap.add_argument("--no-golden", action="store_true", help="build without golden parts (preflight check 5 will fail)")
    ap.add_argument("--branch", default="run")
    ap.add_argument("--tag", default=None, help="default: the next free run-YYYYMMDD-N")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-sign", action="store_true", help="unsigned commit and lightweight tag (tests only)")
    a = ap.parse_args()
    if not a.golden_tar and not a.no_golden:
        ap.error("pass --golden-tar PATH (or --no-golden for a build whose preflight golden check will fail)")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("working tree has uncommitted changes to tracked files: commit them first")
    src = git("rev-parse", "HEAD").strip()
    src_branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    rows = tracked(src)
    missing = [p for p in ALLOW if not any(r[2] == p or r[2].startswith(p) for r in rows)]
    if missing:
        raise SystemExit(f"allow-listed paths missing at {src[:10]}: {missing}")
    entries = [(m, sha, RENAME.get(p, p)) for m, sha, p, _ in rows]
    size = sum(r[3] for r in rows)
    golden = []
    if a.golden_tar:
        data = Path(a.golden_tar).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        sha_file = Path(a.golden_tar).with_suffix(".tar.sha256")
        if sha_file.exists() and sha_file.read_text().split()[0] != digest:
            raise SystemExit(f"{a.golden_tar}: sha256 does not match {sha_file.name}")
        for i in range(0, len(data), PART):
            golden.append((f"golden_parts/R2.tar.part-{i // PART:02d}", data[i:i + PART]))
        golden.append(("golden_parts/R2.tar.sha256", f"{digest}  R2.tar\n".encode()))
        size += len(data)
    source = {"source_commit": src, "source_branch": src_branch, "created": dt.datetime.now().isoformat(timespec="seconds"),
              "files": len(entries) + len(golden), "golden": bool(golden),
              "note": "generated by scripts/make_run_branch.py; edit on the source branch, not here"}
    print(f"source {src_branch}@{src[:10]}: {len(entries)} files + {len(golden)} golden parts, {size / 2**20:.1f} MiB")
    if a.dry_run:
        for _, _, p in sorted(entries, key=lambda e: e[2]):
            print("  ", p)
        for p, _ in golden:
            print("  ", p)
        return 0
    with tempfile.TemporaryDirectory() as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        git("read-tree", "--empty", env=env)
        for mode, sha, path in entries:
            git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}", env=env)
        for path, data in golden:
            git("update-index", "--add", "--cacheinfo", f"100644,{blob(data)},{path}", env=env)
        git("update-index", "--add", "--cacheinfo",
            f"100644,{blob(json.dumps(source, indent=1).encode())},RUN_SOURCE.json", env=env)
        tree = git("write-tree", env=env).strip()
    parent = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", f"refs/heads/{a.branch}"],
                            capture_output=True, text=True).stdout.strip()
    tag = a.tag or next_tag()
    msg = (f"Run checkout {tag} from {src_branch}@{src[:10]}\n\nRuntime files only (scripts/make_run_branch.py)."
           f"{' Golden fixtures in golden_parts/.' if golden else ' No golden fixtures.'}\n\nSource-Commit: {src}\n")
    cmd = ["commit-tree", tree, "-m", msg] + (["-p", parent] if parent else []) + ([] if a.no_sign else ["-S"])
    commit = git(*cmd).strip()
    git("update-ref", f"refs/heads/{a.branch}", commit, *( [parent] if parent else []))
    if a.no_sign:
        git("tag", tag, commit)
    else:
        git("tag", "-s", tag, commit, "-m", f"Run checkout {tag} (source {src[:10]})")
    print(f"branch {a.branch} -> {commit[:10]}, tag {tag}")
    print(f"push with: git push origin {a.branch} {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

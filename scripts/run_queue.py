#!/usr/bin/env python
"""Execute a run queue sequentially and unattended. Safe to kill and restart: a job is skipped once its .done
sentinel exists; experiments themselves also resume at unit granularity.

    python scripts/run_queue.py --queue queue/release1.yaml [--profile run] [--only E0,E2] [--dry-run]

State: <runs_root>/_queue/<tag>/{<job>.done, <job>.failed, <job>.log, heartbeat, queue_status.md}.
A failed job is retried once. Later jobs run unless they depend on a failed job.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness.profile import REPO_ROOT, load_profile
from harness.queue import Queue


def command(job, profile: str, allow_dirty: bool = False) -> list:
    py = sys.executable
    if job.kind == "experiment":
        cmd = [py, "-m", "harness.run", "--config", job.config, "--profile", job.profile or profile, "--stage", job.stage,
                *job.args]
        if allow_dirty and "--allow-dirty" not in cmd:
            cmd.append("--allow-dirty")
        return cmd
    return [py, job.script, *job.args]


def write_status(state: Path, q: Queue, status: dict, current: str = None) -> None:
    lines = [f"# Queue {q.tag}", "", f"updated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; current: {current or '-'}", "",
             "| job | status | attempts | minutes | note |", "|---|---|---|---|---|"]
    for j in q.jobs:
        s = status.get(j.id, {})
        lines.append(f"| {j.id} | {s.get('status', 'pending')} | {s.get('attempts', 0)} | {s.get('minutes', '')} | {s.get('note', '')} |")
    (state / "queue_status.md").write_text("\n".join(lines) + "\n")


def heartbeat(path: Path, stop: threading.Event) -> None:
    while not stop.wait(30):
        path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--only", default=None, help="comma list of job ids")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--allow-dirty", action="store_true", help="pass --allow-dirty to experiment runs")
    ap.add_argument("--retry-failed", action="store_true", help="retry failed jobs (reset failure state)")
    a = ap.parse_args()
    q = Queue.load(a.queue)
    prof_name = a.profile or q.profile
    prof = load_profile(prof_name)
    state = prof.root("runs_root") / "_queue" / q.tag
    state.mkdir(parents=True, exist_ok=True)
    status = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {}
    only = set(a.only.split(",")) if a.only else None
    if a.retry_failed:
        for j in q.jobs:
            if only and j.id not in only:
                continue
            if not (state / f"{j.id}.done").exists():
                status.pop(j.id, None)
                (state / f"{j.id}.failed").unlink(missing_ok=True)
        (state / "status.json").write_text(json.dumps(status, indent=1))
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(state / "heartbeat", stop), daemon=True).start()
    failed = {j for j, s in status.items() if s.get("status") == "failed"}
    try:
        for job in q.jobs:
            if only and job.id not in only:
                continue
            if (state / f"{job.id}.done").exists():
                status.setdefault(job.id, {})["status"] = "done"
                continue
            s = status.setdefault(job.id, {"attempts": 0})
            if s.get("status") == "failed" and (a.retry_failed or (only and job.id in only)):
                s["attempts"] = 0
                s["status"] = "pending"
                (state / f"{job.id}.failed").unlink(missing_ok=True)
                failed.discard(job.id)
            blocked = [d for d in job.depends_on if d in failed]
            if blocked:
                status[job.id] = {"status": "skipped", "note": f"depends on failed {blocked}"}
                failed.add(job.id)
                continue
            cmd = command(job, prof_name, allow_dirty=a.allow_dirty)
            if a.dry_run:
                print(job.id, " ".join(cmd))
                continue
            ok = False
            while s["attempts"] < 2 and not ok:
                s["attempts"] += 1
                s["status"] = "running"
                write_status(state, q, status, job.id)
                t0 = time.time()
                with open(state / f"{job.id}.log", "a") as log:
                    log.write(f"\n=== attempt {s['attempts']} {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
                    log.flush()
                    try:
                        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=job.timeout_h * 3600 if job.timeout_h else None, env=os.environ.copy()).returncode
                    except subprocess.TimeoutExpired:
                        rc = -9
                s["minutes"] = round((time.time() - t0) / 60, 1)
                ok = rc == 0
                s["note"] = "" if ok else f"exit {rc}"
            if ok:
                s["status"] = "done"
                (state / f"{job.id}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                (state / f"{job.id}.failed").unlink(missing_ok=True)
            else:
                s["status"] = "failed"
                (state / f"{job.id}.failed").write_text(s["note"])
                failed.add(job.id)
            (state / "status.json").write_text(json.dumps(status, indent=1))
            write_status(state, q, status)
    finally:
        stop.set()
        (state / "status.json").write_text(json.dumps(status, indent=1))
        write_status(state, q, status)
    print((state / "queue_status.md").read_text())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

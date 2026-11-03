#!/usr/bin/env python
"""Execute a run queue unattended, on every GPU of the machine. Safe to kill and restart: finished units are skipped
(<unit>.done) and experiments resume at stage-3 unit granularity.

    python scripts/run_queue.py --queue queue/release2-server.yaml [--gpus auto|0,1,2,3|cpu] [--workers-per-gpu 1]
                                [--profile P] [--only E3,E4] [--dry-run] [--allow-dirty] [--retry-failed]

The machine is measured first (harness.device.probe_machine -> <state>/machine.json). With one GPU (or one slot) the
queue runs in order exactly as before; with several, harness/scheduler.py runs units side by side. A GPU that fails
preflight's per-GPU check (<state>/gpus.json) is dropped from the pool.

State: <runs_root>/_queue/<tag>/{machine.json, status.json, queue_status.md, heartbeat, <unit>.done, <unit>.log}.
"""
import argparse
import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness import parallel as Par
from harness.device import probe_machine
from harness.profile import REPO_ROOT, load_profile
from harness.queue import Queue
from harness.scheduler import Scheduler, expand, slots_for


def command(job, profile: str, allow_dirty: bool = False, extra=()) -> list:
    py = sys.executable
    if job.kind == "experiment":
        cmd = [py, "-m", "harness.run", "--config", job.config, "--profile", job.profile or profile, "--stage", job.stage,
               *job.args, *extra]
        if allow_dirty and "--allow-dirty" not in cmd:
            cmd.append("--allow-dirty")
        return cmd
    return [py, job.script, *job.args, *extra]


def heartbeat(path: Path, stop: threading.Event) -> None:
    while not stop.wait(30):
        path.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def pick_gpus(arg: str, probe: dict) -> list:
    if arg == "cpu":
        return ["cpu"]
    if arg and arg != "auto":
        return [g.strip() for g in arg.split(",") if g.strip()]
    gpus = probe.get("gpus") or []
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip():
        return [v.strip() for v in vis.split(",") if v.strip()]
    return [str(g["index"]) for g in gpus] or ["cpu"]


def pinned_tokens(q: Queue, prof_name: str, probe: dict, gpus: list, state: Path) -> dict:
    """One max_batch_tokens per experiment config: the smallest over the GPUs in use (measured by preflight when it
    has run, else estimated from the probe). Empty when the profile sets a fixed number (nothing to pin)."""
    from dataclasses import replace

    from harness.memory import resolve_resources
    prof = load_profile(prof_name)
    if not prof.is_auto("max_batch_tokens"):
        return {}
    health = json.loads((state / "gpus.json").read_text()) if (state / "gpus.json").exists() else {}
    vals = []
    for g in probe.get("gpus") or []:
        if str(g["index"]) not in gpus:
            continue
        ok = (health.get("budget_ok") or {}).get(str(g["index"]))
        if ok:
            vals.append(int(ok))
            continue
        p2, _ = resolve_resources(replace(prof), {**probe, "gpus": [g]})
        vals.append(int(p2.max_batch_tokens))
    return {j.config: min(vals) for j in q.jobs if j.kind == "experiment" and j.config} if vals else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--gpus", default="auto", help="auto (every visible GPU), a list like 0,1,2,3, or cpu")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of job ids")
    ap.add_argument("--dry-run", action="store_true", help="print the units and their commands only")
    ap.add_argument("--allow-dirty", action="store_true", help="pass --allow-dirty to experiment runs")
    ap.add_argument("--retry-failed", action="store_true", help="retry failed units (reset failure state)")
    ap.add_argument("--poll", type=float, default=2.0)
    a = ap.parse_args()
    q = Queue.load(a.queue)
    prof_name = a.profile or q.profile
    prof = load_profile(prof_name)
    state = prof.root("runs_root") / "_queue" / q.tag
    state.mkdir(parents=True, exist_ok=True)
    lock = open(state / ".scheduler.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run_queue.py is already running queue {q.tag} ({state})")
    probe = probe_machine(prof.root("cache_root"))
    gpus = pick_gpus(a.gpus, probe)
    slots = slots_for(gpus, a.workers_per_gpu)
    (state / "machine.json").write_text(json.dumps({**probe, "gpus_used": gpus, "slots": slots}, indent=1))
    units = expand(q, len(slots))
    only = set(a.only.split(",")) if a.only else None
    status = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {}
    for u in units:
        if only and u.job.id not in only:
            u.status, u.note = "done", "not selected (--only)"     # treated as satisfied for ordering
            continue
        if (state / f"{u.id}.done").exists() or (u.n == 1 and (state / f"{u.job.id}.done").exists()):
            u.status = "done"
        elif status.get(u.id, {}).get("status") == "failed" and not (a.retry_failed or (only and u.job.id in only)):
            u.status, u.attempts, u.note = "failed", 2, status[u.id].get("note", "")
    if a.dry_run:
        for u in units:
            print(f"{u.id:24} gpu={u.job.gpu:4} deps={u.job.depends_on} {' '.join(command(u.job, prof_name, a.allow_dirty, u.extra_args()))}")
        print(f"slots: {slots}")
        return 0
    n_cleared = Par.clear_stale_claims(prof.root("runs_root"))
    env = {**os.environ, "HL_CPU_PHYSICAL": str(probe.get("cpu_physical") or os.cpu_count() or 1),
           "HL_RAM_BUDGET_BYTES": str(int(0.8 * (probe.get("ram_available_bytes") or 2 ** 40))),
           "HL_PROFILE": prof_name}
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(state / "heartbeat", stop), daemon=True).start()
    log = lambda msg: print(time.strftime("%H:%M:%S"), msg, flush=True)
    log(f"queue {q.tag}: {len(units)} units on slots {slots}; cleared {n_cleared} stale claims")

    class S(Scheduler):
        applied_health = False

        def start(self, u):
            if u.job.kind == "experiment":
                self.pinned = pinned_tokens(q, prof_name, probe, [s for s in set(self.free) | set(slots)], state)
            super().start(u)

        def reap(self, u):
            super().reap(u)
            h = state / "gpus.json"
            if not self.applied_health and h.exists():
                bad = {str(g) for g in json.loads(h.read_text()).get("gpus_failed", [])}
                if bad:
                    self.free = [s for s in self.free if s not in bad]
                    self.n_slots = len([s for s in slots if s not in bad])
                    log(f"dropping GPUs {sorted(bad)} after preflight")
                self.applied_health = True
            if u.status == "done":
                (state / f"{u.id}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                if all(x.status == "done" for x in self.units if x.job.id == u.job.id):
                    (state / f"{u.job.id}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    sched = S(q, units, slots, state, lambda u: command(u.job, prof_name, a.allow_dirty, u.extra_args()), env, {},
              REPO_ROOT, log=log, poll_s=a.poll)

    def write():
        st = {u.id: {"status": u.status, "attempts": u.attempts, "note": u.note, "slots": u.slots} for u in units}
        (state / "status.json").write_text(json.dumps(st, indent=1))
        lines = [f"# Queue {q.tag}", "", f"updated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; slots {slots}", "",
                 "| unit | status | attempts | slots | note |", "|---|---|---|---|---|"]
        lines += [f"| {u.id} | {u.status} | {u.attempts} | {','.join(u.slots)} | {u.note} |" for u in units]
        (state / "queue_status.md").write_text("\n".join(lines) + "\n")

    try:
        sched.run(write)
    finally:
        stop.set()
        write()
    print((state / "queue_status.md").read_text())
    return 1 if any(u.status in ("failed", "skipped") for u in units) else 0


if __name__ == "__main__":
    sys.exit(main())

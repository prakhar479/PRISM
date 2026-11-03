"""Multi-GPU queue scheduler (plan step 6). Pure orchestration: never initialises CUDA itself.

A queue job expands into units:
  - `slices: N|auto` (stage 2): N units, unit i runs `--stage2-slice i/N` (disjoint SAE groups);
  - `workers: N|auto` (stage 3): N units, each a `--worker` that claims stage-3 units (harness.parallel);
  - otherwise one unit.
A job needs one GPU slot per unit (`gpu: one`, experiments), every slot (`gpu: all`, e.g. preflight), or none
(`gpu: none`: prefetch, finalize, bundle). A unit starts when every dependency of its job is done (or finished, for
`after_failed` jobs such as the bundle). Higher `priority` first, then queue order. Each unit gets
CUDA_VISIBLE_DEVICES, HL_CPU_THREADS, HL_RAM_BUDGET_BYTES, HL_CONCURRENT_WORKERS, HL_WORKER_ID and the experiment's
pinned HL_MAX_BATCH_TOKENS. A unit is retried once. This process is the only writer of the queue state.

A queue without `parallel: true` runs exactly as the old sequential runner did: one job at a time in queue order,
the environment passed through unchanged, a job skipped only when one of its dependencies failed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from harness.queue import Job, Queue

MAX_CPU_UNITS = 2                     # concurrent gpu:none units (finalize, bundle, prefetch)


@dataclass
class Unit:
    job: Job
    k: int                            # index within the job
    n: int                            # units of the job
    attempts: int = 0
    status: str = "pending"           # pending | running | done | failed | skipped
    slots: List[str] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = None
    t0: float = 0.0
    note: str = ""

    @property
    def id(self) -> str:
        return self.job.id if self.n == 1 else f"{self.job.id}[{self.k}]"

    def extra_args(self) -> List[str]:
        if self.job.slices:
            return ["--stage2-slice", f"{self.k}/{self.n}"]
        if self.job.workers:
            return ["--worker", f"{self.job.id}-{self.k}"]
        return []


def n_units(job: Job, n_slots: int) -> int:
    v = job.slices or job.workers
    if not v:
        return 1
    return max(1, n_slots if str(v) == "auto" else int(v))


def expand(q: Queue, n_slots: int) -> List[Unit]:
    out = []
    for j in q.jobs:
        n = n_units(j, n_slots)
        out += [Unit(j, k, n) for k in range(n)]
    return out


def slots_for(gpus: List[str], per_gpu: int) -> List[str]:
    """Slot ids are GPU ids repeated per_gpu times (several workers may share one large GPU)."""
    return [g for g in gpus for _ in range(max(1, per_gpu))]


class Scheduler:
    def __init__(self, q: Queue, units: List[Unit], slots: List[str], state: Path, command: Callable[[Unit], List[str]],
                 env_base: dict, pinned_tokens: Dict[str, int], cwd: Path, log=print, poll_s: float = 1.0) -> None:
        self.q, self.units, self.state, self.command = q, units, state, command
        self.free = list(slots)
        self.n_slots = len(slots)
        self.env_base, self.pinned, self.cwd, self.log, self.poll_s = env_base, pinned_tokens, cwd, log, poll_s
        self.order = {j.id: i for i, j in enumerate(q.jobs)}

    # ---- job-level status from its units
    def job_status(self, jid: str) -> str:
        st = [u.status for u in self.units if u.job.id == jid]
        if not st:
            return "missing"
        if all(s == "done" for s in st):
            return "done"
        if any(s in ("failed", "skipped") for s in st) and all(s in ("done", "failed", "skipped") for s in st):
            return "failed"
        return "running" if any(s == "running" for s in st) else "pending"

    def ready(self, u: Unit) -> Optional[bool]:
        """True = may start now; False = wait; None = can never run (a dependency failed)."""
        for d in u.job.depends_on:
            s = self.job_status(d)
            if s == "failed" and not u.job.after_failed:
                return None
            if s not in ("done",) and not (u.job.after_failed and s == "failed"):
                return False
        return True

    def env_for(self, u: Unit, slots: List[str]) -> dict:
        env = dict(self.env_base)
        if not self.q.parallel:
            return env                                     # sequential queues: nothing changes for the child
        gpu = u.job.gpu
        env["CUDA_VISIBLE_DEVICES"] = ",".join(dict.fromkeys(slots)) if gpu != "none" else ""
        env["HL_CONCURRENT_WORKERS"] = str(max(1, self.n_slots))
        phys = int(self.env_base.get("HL_CPU_PHYSICAL", os.cpu_count() or 1))
        env["HL_CPU_THREADS"] = str(max(1, phys // max(1, self.n_slots if gpu != "all" else 1)))
        if u.job.workers:
            env["HL_WORKER_ID"] = f"{u.job.id}-{u.k}"
        if u.job.kind == "experiment" and u.job.config in self.pinned:
            env["HL_MAX_BATCH_TOKENS"] = str(self.pinned[u.job.config])
        return env

    def start(self, u: Unit) -> None:
        g = u.job.gpu if self.q.parallel else "none"       # sequential: no slot bookkeeping
        if g == "all":
            take, self.free = list(self.free), []          # every slot (the child sees each GPU once)
        elif g == "none":
            take = []
        else:
            take = [self.free.pop(0)]
        u.slots, u.attempts, u.status, u.t0 = take, u.attempts + 1, "running", time.time()
        cmd = self.command(u)
        logf = open(self.state / f"{u.id.replace('[', '_').replace(']', '')}.log", "a")
        logf.write(f"\n=== attempt {u.attempts} {time.strftime('%Y-%m-%d %H:%M:%S')} slots={take}: {' '.join(cmd)}\n")
        logf.flush()
        u.proc = subprocess.Popen(cmd, cwd=self.cwd, stdout=logf, stderr=subprocess.STDOUT, env=self.env_for(u, take))
        u.proc._logf = logf                                # noqa: SLF001 - closed when the unit ends
        self.log(f"start {u.id} on {take or 'cpu'} (attempt {u.attempts})")

    def reap(self, u: Unit) -> None:
        rc = u.proc.poll()
        if rc is None:
            return
        u.proc._logf.close()                               # noqa: SLF001
        if self.q.parallel:
            self.free += u.slots
        mins = round((time.time() - u.t0) / 60, 1)
        if rc == 0:
            u.status, u.note = "done", f"{mins} min"
        elif u.attempts < 2:
            u.status, u.note = "pending", f"exit {rc}, retrying"
        else:
            u.status, u.note = "failed", f"exit {rc}"
        self.log(f"end {u.id}: {u.status} ({u.note})")
        u.proc = None

    def pick(self) -> Optional[Unit]:
        if not self.q.parallel:
            if any(u.status == "running" for u in self.units):
                return None
            for u in self.units:                           # queue order, one at a time
                if u.status != "pending":
                    continue
                if any(self.job_status(d) == "failed" for d in u.job.depends_on):
                    u.status, u.note = "skipped", "a dependency failed"
                    continue
                return u
            return None
        running_cpu = sum(1 for u in self.units if u.status == "running" and u.job.gpu == "none")
        cands = []
        for u in self.units:
            if u.status != "pending":
                continue
            r = self.ready(u)
            if r is None:
                u.status, u.note = "skipped", "a dependency failed"
                continue
            if not r:
                continue
            g = u.job.gpu
            if g == "none" and running_cpu >= MAX_CPU_UNITS:
                continue
            if g == "one" and not self.free:
                continue
            if g == "all" and (len(self.free) < self.n_slots):
                continue                                   # waits until every slot is free
            cands.append(u)
        if not cands:
            return None
        return sorted(cands, key=lambda u: (-u.job.priority, self.order[u.job.id], u.k))[0]

    def run(self, write: Callable[[], None]) -> None:
        while True:
            for u in self.units:
                if u.status == "running":
                    self.reap(u)
            while (u := self.pick()) is not None:
                self.start(u)
            write()
            if all(u.status in ("done", "failed", "skipped") for u in self.units):
                return
            if not any(u.status == "running" for u in self.units) and self.pick() is None:
                # nothing runs and nothing can start: everything left waits on something that cannot happen
                for u in self.units:
                    if u.status == "pending":
                        u.status, u.note = "skipped", "unsatisfiable dependencies"
                write()
                return
            time.sleep(self.poll_s)

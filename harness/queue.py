"""Run queues: queue/<tag>.yaml lists jobs in order. Shared by run_queue.py, plan_runs.py and run_prefetch.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Job:
    id: str
    kind: str = "experiment"          # experiment | script
    config: Optional[str] = None      # experiment config (kind=experiment)
    stage: str = "all"
    args: List[str] = field(default_factory=list)       # extra args (script: full argv after the script path)
    script: Optional[str] = None      # kind=script: path of the script to run with the queue's python
    profile: Optional[str] = None
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)
    timeout_h: Optional[float] = None
    required: bool = True             # a failed required job stops jobs that depend on it
    # multi-GPU scheduling (harness/scheduler.py); defaults keep the sequential behaviour
    gpu: Optional[str] = None         # one | all | none (default: one for experiments, none for scripts)
    slices: Optional[object] = None   # stage 2: N or "auto" disjoint SAE-group units (--stage2-slice i/N)
    workers: Optional[object] = None  # stage 3: N or "auto" claiming workers (--worker)
    after_failed: bool = False        # run once dependencies have finished, even if some failed (bundle)

    def __post_init__(self) -> None:
        if self.gpu is None:
            self.gpu = "one" if self.kind == "experiment" else "none"
        if self.gpu not in ("one", "all", "none"):
            raise ValueError(f"{self.id}: gpu must be one | all | none")
        if self.slices and self.workers:
            raise ValueError(f"{self.id}: set slices or workers, not both")


@dataclass
class Queue:
    tag: str
    profile: str
    jobs: List[Job]
    parallel: bool = False            # True: harness/scheduler.py runs units side by side over the GPUs
    prefetch: dict = field(default_factory=dict)   # extra downloads for script jobs: {saes: [[layer, width, l0], ...]}

    @classmethod
    def load(cls, path) -> "Queue":
        d = yaml.safe_load(Path(path).read_text())
        q = cls(d["tag"], d.get("profile", "run"), [Job(**j) for j in d["jobs"]], bool(d.get("parallel", False)),
                dict(d.get("prefetch") or {}))
        ids = [j.id for j in q.jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate job ids")
        for j in q.jobs:
            for dep in j.depends_on:
                if dep not in ids or ids.index(dep) >= ids.index(j.id):
                    raise ValueError(f"{j.id}: dependency {dep} must be an earlier job")
        return q

    def configs(self) -> List[str]:
        return [j.config for j in self.jobs if j.kind == "experiment" and j.config]


def commit_free(job: "Job") -> bool:
    """Units whose result does not depend on the code commit: downloads, preflight, and stage 1/2 of experiments
    (content-addressed caches). Stage-3 workers and finalize write into a run dir whose hash includes the commit; script jobs,
    bundle and publish record the commit; those must be redone (cheaply: they resume) after a re-tag."""
    return job.id in ("prefetch", "preflight") or (job.kind == "experiment" and str(job.stage) in ("1", "2"))


def done_valid(path, job: "Job", n: int, head: str) -> bool:
    """Is the `.done` marker at `path` still valid for this commit and unit count?"""
    import json as _json
    from pathlib import Path as _P
    path = _P(path)
    if not path.exists():
        return False
    try:
        d = _json.loads(path.read_text())
        d = d if isinstance(d, dict) else {}
    except ValueError:
        d = {}                                             # legacy marker: a timestamp only
    if job.slices and d.get("n") not in (None, n):
        return False                                       # stage-2 slice i/N from a run with a different N
    return commit_free(job) or d.get("commit") == head

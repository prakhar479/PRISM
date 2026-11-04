"""Coordination between processes of one machine (one worker per GPU): stage-3 unit claims and a host-RAM byte budget.

Claims: a worker owns a stage-3 unit while `<unit>.claim` exists (created with O_EXCL, so exactly one creator wins).
A claim whose process is gone (same host, dead pid) is stale and may be taken over; finished units have `.done`.

RAM budget: large host-memory users (an SAE's weights, a model load) reserve bytes before allocating them. The total
is HL_RAM_BUDGET_BYTES (set by the scheduler: a share of MemAvailable measured when it started) or, if unset, 80% of
MemAvailable now. A reservation that does not fit waits; the record of reservations is guarded by flock.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("harness")
HOST = socket.gethostname()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parallel_mode() -> bool:
    """True when this process is one of several workers the scheduler started."""
    return int(os.environ.get("HL_CONCURRENT_WORKERS", "1")) > 1 or bool(os.environ.get("HL_WORKER_ID"))


# --------------------------------------------------------------------------- unit claims
def claim(path: Path, worker: str) -> bool:
    """Try to own a unit. True = this worker owns it now; False = another live worker does."""
    path = Path(path)
    body = json.dumps({"worker": worker, "pid": os.getpid(), "host": HOST, "t": time.time()})
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if not stale(path):
                return False
            with contextlib.suppress(FileNotFoundError):
                path.unlink()                    # take over a dead worker's claim, then retry once
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        return True
    return False


def stale(path: Path) -> bool:
    """A claim is stale when its process is gone (same host). Unreadable or foreign-host claims are left alone."""
    try:
        c = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return c.get("host") == HOST and not pid_alive(int(c.get("pid", 0)))


def release(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        Path(path).unlink()


def clear_stale_claims(root: Path) -> int:
    """Remove claims of dead workers under root (the scheduler calls this at start)."""
    n = 0
    for p in Path(root).rglob("*.claim"):
        if stale(p):
            release(p)
            n += 1
    return n


# --------------------------------------------------------------------------- host-RAM budget
def _available_ram() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 2 ** 50


def ram_budget() -> int:
    v = os.environ.get("HL_RAM_BUDGET_BYTES")
    return int(v) if v else int(0.8 * _available_ram())


@contextlib.contextmanager
def reserve_ram(root: Path, nbytes: int, what: str, timeout_s: float = 6 * 3600, poll_s: float = 5.0) -> Iterator[None]:
    """Hold `nbytes` of the machine's host-RAM budget while the block runs (no-op outside parallel mode)."""
    if not parallel_mode() or nbytes <= 0:
        yield
        return
    d = Path(root) / ".ram"
    d.mkdir(parents=True, exist_ok=True)
    me = d / f"{HOST}_{os.getpid()}_{time.time_ns()}.json"
    budget, t0, warned = ram_budget(), time.time(), False
    if nbytes > budget:
        raise MemoryError(f"{what} needs {nbytes / 2**30:.1f} GiB host RAM, the whole budget is {budget / 2**30:.1f} GiB")
    while True:
        with open(d / ".lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            held = 0
            for p in d.glob("*.json"):
                try:
                    r = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                if r.get("host") == HOST and not pid_alive(int(r.get("pid", 0))):
                    p.unlink(missing_ok=True)            # a dead worker's reservation
                    continue
                held += int(r.get("bytes", 0))
            if held + nbytes <= budget:
                me.write_text(json.dumps({"host": HOST, "pid": os.getpid(), "bytes": int(nbytes), "what": what}))
                break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"waited {timeout_s:.0f}s for {nbytes / 2**30:.1f} GiB host RAM ({what})")
        if not warned:
            log.info("waiting for %.1f GiB host RAM for %s (%.1f of %.1f GiB reserved)", nbytes / 2**30, what,
                     held / 2**30, budget / 2**30)
            warned = True
        time.sleep(poll_s)
    try:
        yield
    finally:
        me.unlink(missing_ok=True)


def worker_id() -> Optional[str]:
    return os.environ.get("HL_WORKER_ID")

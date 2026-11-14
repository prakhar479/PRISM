"""Content-addressed caches. A cache entry lives in <cache_root>/<kind>/<key>/ where key = sha256 of its canonical
key parts (model revision, prompt-set hash, layer, site, dtype [, SAE id] ...). key.json stores the parts; a reader
re-derives the key and refuses an entry whose stored parts differ (never silently reuses a stale cache).

Writers fill an entry shard by shard (resumable: finished shards have a .done sentinel) and mark it complete with
_COMPLETE. Tensors are safetensors; small tables are parquet / json.

Concurrency (several processes on one machine, e.g. one per GPU): every file is written to a per-process temporary
name and published with os.replace, so a reader never sees a half-written file and two writers never share a temp
file. `writer_lock()` serialises writers of ONE entry with flock; the scheduler additionally never schedules two
writers of the same entry. flock is unreliable on network filesystems, so there it only warns.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from safetensors.torch import load_file, save_file

CACHE_FORMAT = 1
NETWORK_FS = ("nfs", "nfs4", "cifs", "smbfs", "fuse", "fuse.sshfs", "lustre", "gpfs", "ceph", "9p")
log = logging.getLogger("harness")


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a per-process temporary file and os.replace (atomic on POSIX)."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def fs_type(path: Path) -> str:
    """Filesystem type of the mount holding `path` (from /proc/mounts; '' if unknown)."""
    try:
        p = os.path.realpath(path)
        best, typ = "", ""
        for line in Path("/proc/mounts").read_text().splitlines():
            f = line.split()
            if len(f) >= 3 and (p == f[1] or p.startswith(f[1].rstrip("/") + "/")) and len(f[1]) > len(best):
                best, typ = f[1], f[2]
        return typ
    except OSError:
        return ""


def key_of(parts: dict) -> str:
    blob = json.dumps({"format": CACHE_FORMAT, **parts}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class StaleCacheError(RuntimeError):
    pass


class Entry:
    def __init__(self, root: Path, kind: str, parts: dict) -> None:
        self.parts = {"format": CACHE_FORMAT, **parts}
        self.key = key_of(parts)
        self.dir = Path(root) / kind / self.key
        self.kind = kind

    # ---- lifecycle
    def open(self) -> "Entry":
        self.dir.mkdir(parents=True, exist_ok=True)
        kp = self.dir / "key.json"
        if kp.exists():
            stored = json.loads(kp.read_text())
            if stored != json.loads(json.dumps(self.parts, sort_keys=True, default=str)):
                raise StaleCacheError(f"{self.dir}: stored key parts differ from requested ones")
        else:
            atomic_write_text(kp, json.dumps(self.parts, sort_keys=True, indent=1, default=str))
        return self

    @contextlib.contextmanager
    def writer_lock(self):
        """Exclusive lock on this entry while filling it (released when the process exits, even on a crash).
        Callers re-check shard_done() / complete after acquiring it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if fs_type(self.dir).startswith(NETWORK_FS):
            log.warning("cache %s is on %s: flock is unreliable there, relying on the scheduler alone",
                        self.dir, fs_type(self.dir))
            yield self
            return
        with open(self.dir / ".lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield self
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    @property
    def complete(self) -> bool:
        return (self.dir / "_COMPLETE").exists()

    def mark_complete(self, info: Optional[dict] = None) -> None:
        atomic_write_text(self.dir / "_COMPLETE", json.dumps(info or {}, default=str))

    def require_complete(self) -> "Entry":
        if not self.complete:
            raise FileNotFoundError(f"cache {self.kind}/{self.key} is missing or incomplete ({self.dir})")
        stored = json.loads((self.dir / "key.json").read_text())
        if stored != json.loads(json.dumps(self.parts, sort_keys=True, default=str)):
            raise StaleCacheError(f"{self.dir}: key parts differ")
        return self

    def reset(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # ---- shards
    def shard_done(self, i: int) -> bool:
        return (self.dir / f"shard_{i:05d}.done").exists()

    def write_shard(self, i: int, tensors: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> None:
        tmp = self.dir / f"shard_{i:05d}.safetensors.tmp.{os.getpid()}"
        save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(tmp), metadata={"meta": json.dumps(meta or {})})
        os.replace(tmp, self.dir / f"shard_{i:05d}.safetensors")
        atomic_write_text(self.dir / f"shard_{i:05d}.done", "")

    def shards(self) -> List[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("shard_*.done"))

    def read_shard(self, i: int) -> Dict[str, torch.Tensor]:
        return load_file(str(self.dir / f"shard_{i:05d}.safetensors"))

    def read_all(self, names: Iterable[str]) -> Dict[str, torch.Tensor]:
        names = list(names)
        parts = {n: [] for n in names}
        for i in self.shards():
            s = self.read_shard(i)
            for n in names:
                parts[n].append(s[n])
        return {n: torch.cat(v) if v else torch.zeros(0) for n, v in parts.items()}

    def write_json(self, name: str, obj) -> None:
        atomic_write_text(self.dir / name, json.dumps(obj, indent=1, default=str))

    def read_json(self, name: str):
        return json.loads((self.dir / name).read_text())

    def rel(self) -> str:
        """Path relative to the cache root (what manifests record)."""
        return f"{self.kind}/{self.key}"


def free_gib(path: Path) -> float:
    """Free GiB on the filesystem holding `path` (nearest existing parent if it does not exist yet; 0 if unmeasurable)."""
    p = Path(path)
    try:
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free / 2 ** 30
    except OSError:
        return 0.0


class Cache:
    """`root` is the fast tier. `spill` lists slower roots (e.g. an HDD) used only for entries that no longer fit:
    an entry lives in exactly one tier, found again by every process by looking for <tier>/<kind>/<key> (a complete
    copy wins, then a partial one with room to finish). A new entry goes to the first tier with >= min_free_gb free,
    so the fast disk is used until it is nearly full and nothing is ever moved or evicted behind a reader's back."""

    def __init__(self, root: Path, spill=(), min_free_gb: float = 4.0) -> None:
        self.root = Path(root)
        self.roots = [self.root] + [Path(r) for r in spill]
        self.min_free_gb = float(min_free_gb)

    def locate(self, kind: str, key: str) -> Path:
        if len(self.roots) == 1:
            return self.root
        found = [r for r in self.roots if (r / kind / key).is_dir()]
        for r in found:
            if (r / kind / key / "_COMPLETE").exists():
                return r
        for r in found:
            if free_gib(r) >= self.min_free_gb:
                return r                                    # resumable partial entry with room to finish
        for r in self.roots:
            if r in found or free_gib(r) < self.min_free_gb:
                continue
            try:
                r.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue                                    # tier not mounted / not writable
            return r
        usable = [r for r in self.roots if self._usable(r)] or [self.root]
        return max(usable, key=free_gib)                    # everything is low: use the roomiest, the queue's disk guard pauses

    @staticmethod
    def _usable(r: Path) -> bool:
        try:
            r.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    def entry(self, kind: str, **parts) -> Entry:
        return Entry(self.locate(kind, key_of(parts)), kind, parts)

    def where(self) -> Dict[str, int]:
        """Entries per tier (for reports)."""
        return {str(r): sum(1 for _ in r.glob("*/*/_COMPLETE")) if r.exists() else 0 for r in self.roots}


@contextlib.contextmanager
def lock_many(entries: Iterable[Entry]):
    """Writer locks on several entries, always taken in key order (two processes cannot deadlock)."""
    with contextlib.ExitStack() as stack:
        for e in sorted({e.dir: e for e in entries}.values(), key=lambda e: (e.kind, e.key)):
            stack.enter_context(e.writer_lock())
        yield

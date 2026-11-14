"""What each disk is and how it is used: filesystem, SSD or HDD, free space, a short speed test, and the size of every
item the harness keeps on it. Read-only except the speed test, which writes and deletes one temporary file.
Used by scripts/storage_report.py and by the queue's periodic disk log."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from harness.cache import free_gib, fs_type

GiB = 2 ** 30


def _nearest(path: Path) -> Path:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _mount(path: Path):
    """(mount point, device) of the longest /proc/mounts prefix of `path`."""
    p, best, dev = os.path.realpath(_nearest(path)), "", ""
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            f = line.split()
            if len(f) >= 3 and (p == f[1] or p.startswith(f[1].rstrip("/") + "/")) and len(f[1]) > len(best):
                best, dev = f[1], f[0]
    except OSError:
        pass
    return best, dev


def rotational(dev: str) -> Optional[bool]:
    """True = spinning disk, False = SSD/NVMe, None = unknown (network or virtual device)."""
    if not dev.startswith("/dev/"):
        return None
    try:
        node = Path("/sys/class/block") / os.path.basename(os.path.realpath(dev))
        real = node.resolve()
        for d in (real, real.parent):                       # a partition keeps `queue/` in its parent disk
            q = d / "queue" / "rotational"
            if q.exists():
                return q.read_text().strip() == "1"
    except OSError:
        pass
    return None


def disk_info(path: Path) -> dict:
    import shutil
    where = _nearest(path)
    mnt, dev = _mount(path)
    du = shutil.disk_usage(where)
    rot = rotational(dev)
    return {"path": str(path), "mount": mnt, "device": dev, "fs": fs_type(where),
            "kind": {True: "HDD", False: "SSD", None: "network/unknown"}[rot], "rotational": rot,
            "total_gib": round(du.total / GiB, 1), "free_gib": round(du.free / GiB, 1)}


def speed(path: Path, mb: int = 256) -> dict:
    """Sequential write and read MB/s on the disk holding `path` (one temp file; the read bypasses the page cache
    where the OS honours posix_fadvise, otherwise it overstates and the report says so)."""
    d = _nearest(path)
    f = d / f".speedtest.{os.getpid()}"
    buf = os.urandom(1 << 20)
    try:
        t = time.time()
        with open(f, "wb") as fh:
            for _ in range(mb):
                fh.write(buf)
            fh.flush()
            os.fsync(fh.fileno())
        w = mb / max(time.time() - t, 1e-6)
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            t = time.time()
            while os.read(fd, 1 << 22):
                pass
            r = mb / max(time.time() - t, 1e-6)
        finally:
            os.close(fd)
        return {"write_mb_s": round(w), "read_mb_s": round(r)}
    except OSError as e:
        return {"error": str(e)}
    finally:
        f.unlink(missing_ok=True)


def tree_bytes(path: Path) -> int:
    n = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                n += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return n


def cache_usage(root: Path) -> Dict[str, int]:
    """Bytes per cache kind (spans, sae_out, ...) under one cache root."""
    root = Path(root)
    return {k.name: tree_bytes(k) for k in sorted(root.iterdir()) if k.is_dir() and not k.name.startswith(".")} if root.is_dir() else {}


def tiers(prof) -> List[dict]:
    """The profile's cache tiers, fast first, with their disks."""
    return [{"role": "cache" if i == 0 else "cache-spill", **disk_info(r)}
            for i, r in enumerate([prof.root("cache_root"), *prof.spill_roots()])]


def snapshot(prof) -> str:
    """One line for logs: free GiB on every disk the run writes to."""
    parts = [("cache", prof.root("cache_root"))] + [("spill", p) for p in prof.spill_roots()] + [("runs", prof.root("runs_root"))]
    if prof.hf_home:
        parts.append(("hf", Path(prof.hf_home)))
    seen, out = set(), []
    for name, p in parts:
        out.append(f"{name} {free_gib(p):.0f}G free @ {p}")
    return "; ".join(out)

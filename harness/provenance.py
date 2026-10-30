"""Run provenance: git state, packages, machine, resolved profile. Paths in manifests are relative to roots."""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path

from harness.profile import REPO_ROOT

PACKAGES = ("torch", "transformers", "transformer-lens", "numpy", "pandas", "pyarrow", "safetensors", "huggingface-hub",
            "scipy")


def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_state() -> dict:
    dirty = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tags": git("tag", "--points-at", "HEAD").split(), "dirty": bool(dirty), "dirty_files": dirty[:50]}


def packages() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def machine(device) -> dict:
    from harness.device import device_info

    info = {"hostname": socket.gethostname(), "os": platform.platform(), "python": platform.python_version(),
            **device_info(device)}
    try:
        info["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        info["nvidia_driver"] = None
    return info


def canonical_hash(obj, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def assert_relative(obj, roots) -> None:
    """Raise if any string in obj is an absolute path (manifests must stay portable)."""
    roots = [str(r) for r in roots]

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and (o.startswith("/") or any(o.startswith(r) for r in roots)) and len(o) > 1:
            raise ValueError(f"absolute path in manifest: {o}")
    walk(obj)

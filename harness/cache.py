"""Content-addressed caches. A cache entry lives in <cache_root>/<kind>/<key>/ where key = sha256 of its canonical
key parts (model revision, prompt-set hash, layer, site, dtype [, SAE id] ...). key.json stores the parts; a reader
re-derives the key and refuses an entry whose stored parts differ (never silently reuses a stale cache).

Writers fill an entry shard by shard (resumable: finished shards have a .done sentinel) and mark it complete with
_COMPLETE. Tensors are safetensors; small tables are parquet / json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from safetensors.torch import load_file, save_file

CACHE_FORMAT = 1


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
            kp.write_text(json.dumps(self.parts, sort_keys=True, indent=1, default=str))
        return self

    @property
    def complete(self) -> bool:
        return (self.dir / "_COMPLETE").exists()

    def mark_complete(self, info: Optional[dict] = None) -> None:
        (self.dir / "_COMPLETE").write_text(json.dumps(info or {}, default=str))

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
        tmp = self.dir / f"shard_{i:05d}.safetensors.tmp"
        save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(tmp), metadata={"meta": json.dumps(meta or {})})
        tmp.rename(self.dir / f"shard_{i:05d}.safetensors")
        (self.dir / f"shard_{i:05d}.done").write_text("")

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
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=str))

    def read_json(self, name: str):
        return json.loads((self.dir / name).read_text())

    def rel(self) -> str:
        """Path relative to the cache root (what manifests record)."""
        return f"{self.kind}/{self.key}"


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def entry(self, kind: str, **parts) -> Entry:
        return Entry(self.root, kind, parts)

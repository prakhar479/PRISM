"""Machine profiles: profiles/<name>.yaml, then profiles/local.yaml (git-ignored), then environment variables.

    HL_PROFILE      profile name when none is passed explicitly (default: dev)
    HL_CACHE_ROOT   cache_root      HL_DATA_ROOT  data_root      HL_RUNS_ROOT  runs_root
    HL_DEVICE       device          HF_HOME       hf_home (also exported for huggingface_hub)
Relative roots resolve against the repo root. The resolved profile is stored in every run manifest.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "profiles"
ENV_KEYS = {"HL_CACHE_ROOT": "cache_root", "HL_DATA_ROOT": "data_root", "HL_RUNS_ROOT": "runs_root",
            "HL_DEVICE": "device", "HF_HOME": "hf_home"}


@dataclass
class Profile:
    name: str
    device: str = "auto"
    model_dtype: str = "float32"
    sae_dtype: str = "float32"
    sae_bf16_min_width: str = "524k"
    sae_placement: str = "auto"
    cache_root: str = "cache"
    data_root: str = "data"
    runs_root: str = "runs"
    hf_home: Optional[str] = None
    offline: bool = False
    batch_size: int = 8
    max_batch_size: int = 64
    max_batch_tokens: int = 4096
    gpu_mem_cap_gb: Optional[float] = None
    cpu_workers: int = 2
    offload_embed: bool = False
    allowed_experiments: List[str] = field(default_factory=list)
    allow_real_model: bool = False
    sources: List[str] = field(default_factory=list)      # files / env vars that set values (provenance)

    def root(self, key: str) -> Path:
        """Absolute path of cache_root / data_root / runs_root."""
        p = Path(getattr(self, key))
        return p if p.is_absolute() else REPO_ROOT / p

    def to_dict(self) -> dict:
        return asdict(self)

    def check_experiment(self, exp_id: str) -> None:
        if self.allowed_experiments and exp_id not in self.allowed_experiments:
            raise PermissionError(f"profile '{self.name}' does not allow experiment {exp_id}")

    def apply_env(self) -> None:
        """Export HF settings this profile asks for (call once per process before importing HF loaders)."""
        if self.hf_home:
            os.environ["HF_HOME"] = str(self.hf_home)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"


def _read(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_profile(name: Optional[str] = None, env: Optional[dict] = None, profile_dir: Path = PROFILE_DIR) -> Profile:
    env = os.environ if env is None else env
    name = name or env.get("HL_PROFILE") or "dev"
    path = profile_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no profile {path}")
    data = _read(path)
    sources = [str(path.relative_to(profile_dir.parent)) if path.is_relative_to(profile_dir.parent) else str(path)]
    local = profile_dir / "local.yaml"
    if local.exists():
        loc = _read(local)
        loc = loc.get(name, loc) if isinstance(loc.get(name), dict) else loc   # local.yaml may be keyed by profile
        data.update({k: v for k, v in loc.items() if not isinstance(v, dict)})
        sources.append("profiles/local.yaml")
    for var, key in ENV_KEYS.items():
        if env.get(var):
            data[key] = env[var]
            sources.append(f"env:{var}")
    known = {f for f in Profile.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown profile keys {sorted(unknown)} in {sources}")
    data["name"] = data.get("name", name)
    data["sources"] = sources
    return Profile(**data)

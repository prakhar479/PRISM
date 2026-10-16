"""Device detection, seeding, memory cleanup, config and JSON/NPZ/table I/O."""
from __future__ import annotations

import gc
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on CUDA, float32 everywhere else."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def model_device(model) -> torch.device:
    """HookedTransformer has no `.device`; fall back to cfg.device / first parameter."""
    d = getattr(model, "device", None)
    if d is None:
        d = getattr(getattr(model, "cfg", None), "device", None)
    if d is None:
        d = next(model.parameters()).device
    return torch.device(d)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("prism")


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path is None:
        path = str(Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml")
    cfg = _load_yaml_with_base(Path(path))
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def _load_yaml_with_base(path: Path) -> Dict[str, Any]:
    """YAML config; a top-level `base: other.yaml` (relative to this file) is loaded first and updated
    (dict values such as `mock` are merged one level deep)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    merged = _load_yaml_with_base(path.parent / base)
    for k, v in cfg.items():
        merged[k] = {**merged[k], **v} if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
    return merged


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_table(rows: list, path_stem: Path) -> None:
    """Write list-of-dicts as CSV (always) and Parquet (if pyarrow is installed)."""
    import pandas as pd

    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path_stem.with_suffix(".parquet"), index=False)
    except Exception:  # pyarrow missing -> CSV only
        pass


def load_table(path_csv: Path):
    import pandas as pd
    return pd.read_csv(path_csv)

"""Orchestration shared by run_full_suite.py and run_smoke_test.py."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

from src.analysis.plot_figures import make_all_figures
from src.analysis.summary import write_summary
from src.data.synthetic_tasks import get_tokenizer
from src.experiments.common import RunContext
from src.experiments.exp1_causal_split import run_exp1
from src.experiments.exp2_geometry import run_exp2
from src.experiments.exp3_hybrid_repair import run_exp3
from src.models.runtime import setup_runtime
from src.models.sae_loader import load_model
from src.utils import get_device, get_dtype, save_json, set_seed

log = logging.getLogger("prism")


def run_suite(cfg: Dict, mock: bool, out_dir: Path, experiments: Sequence[str] = ("exp1", "exp2", "exp3", "plots"),
              resume: bool = False, deadline: Optional[float] = None) -> Dict:
    import time

    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = get_device("cpu" if mock else None)
    dtype = get_dtype(device)
    log.info("device=%s dtype=%s mock=%s out=%s", device, dtype, mock, out_dir)
    tok = get_tokenizer(mock, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
    model = load_model(mock=mock, device=device, dtype=dtype, mock_cfg=cfg.get("mock"), tokenizer=tok,
                       model_name=cfg["model_name"], runtime_cfg=cfg.get("runtime"))
    plan = setup_runtime(cfg, model, device)
    ctx = RunContext(model=model, tok=tok, cfg=cfg, mock=mock, device=device, out_dir=out_dir, resume=resume,
                     deadline=deadline, runtime_plan=plan)
    layers = set()
    if "exp1" in experiments:
        layers |= set(cfg["layers"])
    if "exp2" in experiments or "exp3" in experiments:
        layers |= set(cfg["geometry_layers"])
    info = {"cfg": cfg, "mock": mock, "device": str(device), "dtype": str(dtype), "experiments": list(experiments),
            "runtime": ctx.runtime_plan}
    if layers:
        ctx.prepare(sorted(layers))
        info.update(tasks_run=list(ctx.data), task_info=ctx.task_info)
        log.info("task info: %s", ctx.task_info)
        save_json(info, out_dir / "run_info.json")
    res: Dict = {}
    for name, fn in (("exp1", run_exp1), ("exp2", run_exp2), ("exp3", run_exp3)):
        if name in experiments and ctx.data:
            t0 = time.time()
            res[name] = fn(ctx)
            info.setdefault("timings", {})[name] = round(time.time() - t0, 1)
            save_json(info, out_dir / "run_info.json")
    if "plots" in experiments:
        try:
            res["figures"] = make_all_figures(out_dir)
        except Exception as e:  # noqa: BLE001 - never lose metrics to a plotting bug
            log.error("plotting failed: %s", e)
        try:
            write_summary(out_dir)
        except Exception as e:  # noqa: BLE001
            log.error("summary failed: %s", e)
    info["total_seconds"] = round(time.time() - t_start, 1)
    save_json(info, out_dir / "run_info.json")
    return res

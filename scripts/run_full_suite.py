#!/usr/bin/env python
"""Full PRISM pipeline: exp1 -> exp2 -> exp3 -> figures."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.experiments.common import merge_task_overrides
from src.pipeline import run_suite
from src.utils import load_config, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="CPU mock model/SAE (no weights, no GPU)")
    ap.add_argument("--experiments", default="exp1,exp2,exp3,plots")
    ap.add_argument("--tasks", default=None, help="comma list")
    ap.add_argument("--layers", default=None, help="comma list of exp1 layers")
    ap.add_argument("--widths", default=None, help="comma list, e.g. 16k,65k")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="skip work units already recorded in <out>/metrics/*_units.jsonl")
    ap.add_argument("--overrides", default=None, help="JSON with task_overrides (written by scripts/preflight.py)")
    ap.add_argument("--deadline", type=float, default=None, help="unix time after which no new work unit starts")
    ap.add_argument("--set", action="append", default=[], help="extra config override key=yaml_value")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",")] if s else None
    strs = lambda s: s.split(",") if s else None
    cfg = load_config(a.config, dict(tasks=strs(a.tasks), layers=ints(a.layers), widths=strs(a.widths),
                                     n_train=a.n_train, n_test=a.n_test, seed=a.seed))
    import yaml
    from src.utils import load_json
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if a.overrides and Path(a.overrides).exists():
        ov = load_json(Path(a.overrides))
        cfg["task_overrides"] = merge_task_overrides(cfg.get("task_overrides"), ov.get("task_overrides"))
        if ov.get("drop_tasks"):
            cfg["tasks"] = [t for t in cfg["tasks"] if t not in ov["drop_tasks"]]
    out = Path(a.out or f"results/{'mock_' if a.mock else ''}{cfg['run_name']}")
    setup_logging(out / "logs" / "run.log")
    run_suite(cfg, a.mock, out, a.experiments.split(","), resume=a.resume, deadline=a.deadline)


if __name__ == "__main__":
    main()

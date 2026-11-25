#!/usr/bin/env python
"""Breadth pilot (P2 / P4): ordinary-text fidelity of every dictionary a pilot config uses, on the held-out generic set.

    python scripts/run_dict_fidelity.py --config configs/pilot/P4.yaml                 # every distinct cell dictionary
    python scripts/run_dict_fidelity.py --config configs/pilot/P2.yaml --extras        # + norm-matched noise controls (E23)
    python scripts/run_dict_fidelity.py --config <cfg> --tiny                          # tests

Dictionaries are resolved and loaded by the experiment runner itself (harness.run.Runner.get_sae: Gemma Scope, registry,
trained, add-ons), so this measures exactly what the patching cells use; an add-on reads the map that the config's stage 2
fitted (run this job after the config's stage 2). Metrics are harness/fidelity.py's (CE clean / spliced / mean- and
zero-ablated, CE recovered, FVU, |eps|/|x|, L0), per window with bootstrap CIs over windows. The evaluation set
(data/generic/pile10k_100x128.json) shares no document with the fit set used by add-ons and training.
Output: <runs_root>/PF/<run>/{fidelity.csv, per_window_<dict>.npz, manifest.json}; finished dictionaries are skipped.
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import types

import numpy as np

import _bootstrap  # noqa: F401

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from harness import provenance
from harness.profile import REPO_ROOT, load_profile

log = logging.getLogger("harness")
FIELDS = ["dict", "layer", "source", "name", "addon", "width", "l0_variant", "n_windows", "n_tokens", "ce_clean", "ce_splice",
          "ce_mean_ablate", "ce_zero_ablate", "recovered_mean", "recovered_zero", "ce_delta", "fvu", "eps_over_x", "l0"]
CI = ["ce_delta", "recovered_mean", "recovered_zero", "fvu", "eps_over_x", "l0"]
EXTRA = ["ce_noise", "ce_perturb", "noise_delta", "perturb_delta", "splice_over_noise"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", default=os.environ.get("HL_PROFILE", "server"))
    ap.add_argument("--eval-set", default="generic/pile10k_100x128.json")
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--extras", action="store_true", help="norm-matched noise controls (x_hat + eta, x + eta)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--only-sae", action="append", default=None,
                    help="smoke subsets: only the dictionaries of cells matching this selector (harness.run.select_cells; repeatable)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import torch
    from harness import fidelity as Fd
    from harness.batching import Budget
    from harness.config import ExpConfig
    from harness.device import empty_cache, resolve_device
    from harness.run import Runner

    cfg = ExpConfig.load(a.config)
    if a.only_sae:
        import dataclasses
        from harness.run import select_cells
        keep = select_cells(cfg.cells, a.only_sae)
        if not keep:
            raise SystemExit(f"--only-sae {a.only_sae}: no cell matches")
        cfg = dataclasses.replace(cfg, cells=keep)
    prof = load_profile("dev" if a.tiny else a.profile)
    prof.apply_env()
    args = types.SimpleNamespace(tiny=a.tiny, smoke=False, dry_run=False, finalize_only=False, worker=None, config=a.config,
                                 allow_dirty=a.allow_dirty)
    R = Runner(cfg, prof, args)
    ev = json.loads((REPO_ROOT / "data" / a.eval_set).read_text())
    ids = ev["ids"][:a.n_windows] if a.n_windows else ev["ids"]
    if a.tiny:
        ids = [[min(t, 511) for t in r] for r in ids[:8]]
    sids = {}
    for c in cfg.cells:
        sid = R.sae_id(c.layer, c.sae)
        sids.setdefault(sid.key, sid)
    ident = {"exp": "PF", "config": cfg.exp_id, "config_hash": provenance.canonical_hash(cfg.to_dict()),
             "dicts": sorted(sids), "eval": ev["hash"], "n_windows": a.n_windows, "extras": a.extras, "tiny": a.tiny}
    run_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = (REPO_ROOT / "tests" / "_tiny_runs" if a.tiny else prof.root("runs_root")) / "PF" / run_hash
    git = provenance.git_state()
    if git["dirty"] and not (a.allow_dirty or a.tiny):
        raise SystemExit("working tree is dirty: commit, or pass --allow-dirty")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "fidelity.csv"
    done = set()
    if csv_path.exists():
        done = {r["dict"] for r in csv.DictReader(csv_path.open()) if r.get("dict")}
    todo = [s for k, s in sids.items() if k not in done]
    log.info("PF: %d dictionaries to do, %d done", len(todo), len(done))
    device = torch.device("cpu") if a.tiny else resolve_device(prof.device)
    if not a.tiny:
        from harness import memory
        from harness.device import device_memory, probe_machine
        prof, _ = memory.resolve_resources(prof, probe_machine(prof.root("cache_root")), device_memory(device))
    budget = Budget(prof.max_batch_tokens if not a.tiny else 4096, prof.max_batch_size if not a.tiny else 8)
    t_start = time.time()
    if todo:
        model = R.load_model()
        need = sorted({s.layer for s in todo if getattr(s, "addon", "")})
        if need:                                    # add-on maps are fitted on the generic residuals: fill any missing layer
            from harness.model import stages as S
            S.stage1_generic(model, R.spec, R.generic["ids"], R.generic["hash"], need, R.generic["skip"], R.cache,
                             R.budget, device)
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS + [c + s for c in CI for s in ("_lo", "_hi")]
                               + (EXTRA + [c + s for c in EXTRA for s in ("_lo", "_hi")] if a.extras else []), extrasaction="ignore")
            if new:
                w.writeheader()
            for sid in todo:
                t0 = time.time()
                sae = R.get_sae(sid)
                per = Fd.measure(model, sae, sid.layer, ids, ev["skip_positions"], budget, device, extras=a.extras)
                s = Fd.summarize(per, len(ids) * (len(ids[0]) - ev["skip_positions"] - 1), a.n_boot)
                np.savez_compressed(run_dir / f"per_window_{sid.key}.npz", **per)
                w.writerow({"dict": sid.key, "layer": sid.layer, "source": getattr(sid, "source", "gemma_scope"),
                            "name": getattr(sid, "name", ""), "addon": getattr(sid, "addon", ""), "width": sid.width,
                            "l0_variant": sid.l0, **s})
                fh.flush()
                log.info("%s: CE delta %.4f, recovered(mean) %.4f, FVU %.4f, L0 %.1f (%.0fs)", sid.key, s["ce_delta"],
                         s["recovered_mean"], s["fvu"], s["l0"], time.time() - t0)
                del sae
                empty_cache()
    status = "complete" if len(done) + len(todo) == len(sids) else "partial"
    (run_dir / "manifest.json").write_text(json.dumps({
        "exp_id": "PF", "run_hash": run_hash, "status": status, "smoke": False, "tiny": a.tiny, "table": "fidelity.csv",
        "config": {"name": "dict_fidelity", "identity": ident, "source_config": a.config}, "git": git,
        "packages": provenance.packages(), "machine": provenance.machine(device), "model": R.spec.key(),
        "dicts": {k: s.to_dict() for k, s in sids.items()}, "created": provenance.now(),
        "wall_time_s": round(time.time() - t_start, 1)}, indent=1, default=str))
    log.info("PF %s -> %s (%s)", run_hash, run_dir, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""E12 runner: SAE fidelity on ordinary text per language/script (harness/fidelity.py).

    python scripts/run_fidelity.py --sets en_pile,hi_deva_web --layers 12 18          # real model, one GPU
    python scripts/run_fidelity.py --tiny --sets en_pile --layers 1 --n-windows 16     # tiny random model + random SAE (tests)
    python scripts/run_fidelity.py --sets all --dry-run                                # what would run

Reads data/lang/<set>.json (scripts/build_lang_sets.py), uses the pilot SAE variants (16k, L0 82 at L12, 74 at L18; the same
as the confirmatory cells) and writes <runs_root>/E12/<run>/{fidelity.csv, per_window_<set>_L<layer>.npz, manifest.json}.
Finished (set, layer) pairs are skipped on restart. Table assembly and plots are on DEV (scripts/make_e12.py).
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from harness import provenance
from harness.profile import REPO_ROOT, load_profile

PILOT_L0 = {12: 82, 18: 74}                  # 16k pilot variants (configs/experiments/E4.yaml confirmatory cells)


def l0_for(layer: int, width: str) -> int:
    """Gemma Scope's canonical variant (average L0 closest to 100) from configs/sae_inventory.json; equals the pilot's variant
    at L12 and L18 (checked), so other layers (E12 depth profile) use the same rule."""
    from harness.sae import canonical_l0
    inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())["inventory"]
    l0 = canonical_l0(inv[str(layer)][width])
    if layer in PILOT_L0 and width == "16k" and l0 != PILOT_L0[layer]:
        raise SystemExit(f"inventory canonical L0 {l0} != pilot {PILOT_L0[layer]} at L{layer}")
    return l0
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
FIELDS = ["set", "lang", "script", "genre", "licence", "layer", "sae", "n_windows", "n_tokens", "tokens_per_word",
          "ce_clean", "ce_splice", "ce_mean_ablate", "ce_zero_ablate", "recovered_mean", "recovered_zero", "ce_delta",
          "fvu", "eps_over_x", "l0"]
CI = ["ce_clean", "ce_splice", "recovered_mean", "recovered_zero", "ce_delta", "fvu", "eps_over_x", "l0"]
log = logging.getLogger("harness")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=os.environ.get("HL_PROFILE", "server"))
    ap.add_argument("--sets", default="all", help="comma list of data/lang set names, or all")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 18])
    ap.add_argument("--width", default="16k")
    ap.add_argument("--n-windows", type=int, default=None, help="use only the first N windows of each set")
    ap.add_argument("--lang-dir", default=str(REPO_ROOT / "data" / "lang"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    prof = load_profile(a.profile)
    prof.apply_env()
    lang_dir = Path(a.lang_dir)
    man = json.loads((lang_dir / "MANIFEST.json").read_text())
    names = sorted(man) if a.sets == "all" else [s for s in a.sets.split(",") if s]
    missing = [n for n in names if n not in man]
    if missing:
        raise SystemExit(f"unknown sets {missing}; have {sorted(man)}")
    import hashlib
    sae_rev = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())["revision"]
    ident = {"exp": "E12", "model": MODEL_REVISION, "sae_revision": sae_rev, "width": a.width, "layers": a.layers,
             "sets": {n: man[n]["hash"] for n in names}, "n_windows": a.n_windows, "tiny": a.tiny}
    run_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = (REPO_ROOT / "tests" / "_tiny_runs" if a.tiny else prof.root("runs_root")) / "E12" / run_hash
    todo = [(n, L) for n in names for L in a.layers]
    if a.dry_run:
        for n, L in todo:
            print(f"{n:20s} L{L} {a.width} l0={l0_for(L, a.width)}  {man[n]['n_docs']} windows, {man[n]['tokens_scored']} tokens")
        return 0
    git = provenance.git_state()
    if git["dirty"] and not (a.allow_dirty or a.tiny):
        raise SystemExit("working tree is dirty: commit, or pass --allow-dirty\n  " + "\n  ".join(git["dirty_files"][:10]))
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from harness import fidelity as F
    from harness.batching import Budget
    from harness.device import device_memory, empty_cache, probe_machine, resolve_device, set_threads
    from harness import memory
    from harness.model import ModelSpec, build_model
    from harness.sae import SAEId, load_params, params_path, write_random_params

    device = torch.device("cpu") if a.tiny else resolve_device(prof.device)
    probe = probe_machine(prof.root("cache_root"))
    prof, resources = memory.resolve_resources(prof, probe, None if a.tiny else device_memory(device))
    set_threads(prof.cpu_threads if isinstance(prof.cpu_threads, int) else None)
    spec = ModelSpec(name="tiny", dtype="float32", tiny_vocab=512, tiny_layers=max(a.layers) + 2) if a.tiny else \
        ModelSpec(revision=MODEL_REVISION, dtype=prof.model_dtype)
    budget = Budget(prof.max_batch_tokens if not a.tiny else 4096, prof.max_batch_size if not a.tiny else 8)
    csv_path = run_dir / "fidelity.csv"
    done = set()
    if csv_path.exists():
        done = set()
        for r in csv.DictReader(csv_path.open()):          # a row cut short by a kill is skipped, so the pair is redone
            try:
                done.add((r["set"], int(r["layer"])))
            except (TypeError, ValueError, KeyError):
                continue
    todo = [t for t in todo if t not in done]
    log.info("E12: %d (set, layer) pairs to do, %d already done; device %s", len(todo), len(done), device)
    if not todo:
        return 0
    model = build_model(spec, device, offload_embed=prof.offload_embed if not a.tiny else False)
    new = not csv_path.exists()
    t_start = time.time()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + [c + s for c in CI for s in ("_lo", "_hi")], extrasaction="ignore")
        if new:
            w.writeheader()
        for layer in a.layers:
            sets = [n for n, L in todo if L == layer]
            if not sets:
                continue
            sid = SAEId(layer, a.width, l0_for(layer, a.width) if not a.tiny else 1, revision=sae_rev)
            if a.tiny:
                sae = load_params(write_random_params(run_dir / "tiny_saes" / f"{sid.key}.npz", 64, 512, seed=layer, thr=0.05), layer, a.width)
            else:
                from harness.memory import sae_dtype_for
                sae = load_params(params_path(sid), layer, a.width, sae_dtype_for(a.width, prof), sid.key)
            for n in sets:
                d = json.loads((lang_dir / man[n]["file"]).read_text())
                ids = d["ids"][:a.n_windows] if a.n_windows else d["ids"]
                if a.tiny:
                    ids = [[min(t, 511) for t in r] for r in ids]
                t0 = time.time()
                per = F.measure(model, sae, layer, ids, d["skip_positions"], budget, device)
                n_scored = len(ids) * (len(ids[0]) - d["skip_positions"] - 1)
                s = F.summarize(per, n_scored, a.n_boot)
                np.savez_compressed(run_dir / f"per_window_{n}_L{layer}.npz", **per)
                row = {"set": n, "lang": d["lang"], "script": d["script"], "genre": d["genre"], "licence": d["licence"],
                       "layer": layer, "sae": sid.key, "tokens_per_word": d["tokens_per_word"], **s}
                w.writerow(row)
                fh.flush()
                log.info("%s L%d: CE clean %.3f splice %.3f, recovered(mean) %.3f [%.3f, %.3f], FVU %.3f  (%.0fs)", n, layer,
                         s["ce_clean"], s["ce_splice"], s["recovered_mean"], s["recovered_mean_lo"], s["recovered_mean_hi"],
                         s["fvu"], time.time() - t0)
            del sae
            empty_cache()
    status = "complete" if all((run_dir / f"per_window_{n}_L{L}.npz").exists() for n in names for L in a.layers) else "partial"
    (run_dir / "manifest.json").write_text(json.dumps({
        "exp_id": "E12", "run_hash": run_hash, "status": status, "table": "fidelity.csv", "smoke": False, "tiny": a.tiny,
        "config": {"name": "fidelity_by_language", "identity": ident, "layers": a.layers, "width": a.width,
                   "l0": {str(L): l0_for(L, a.width) for L in a.layers} if not a.tiny else {}},
        "profile": prof.to_dict(), "git": git, "packages": provenance.packages(), "machine": provenance.machine(device),
        "model": spec.key(), "sae_revision": sae_rev, "prompt_sets": {n: {"hash": man[n]["hash"], "file": man[n]["file"], "n": man[n]["n_docs"]} for n in names},
        "seeds": {"bootstrap_seed": 0, "n_boot": a.n_boot}, "timings": {"total_s": round(time.time() - t_start, 1)},
        "wall_time_s": round(time.time() - t_start, 1), "created": provenance.now(), "sets_sha256": {n: man[n]["sha256"] for n in names},
        "resources": {k: v for k, v in resources.items() if k != "probe"}, "budget_backoffs": budget.backoffs},
        indent=1, default=str))
    log.info("E12 %s -> %s (%s)", run_hash, run_dir, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())

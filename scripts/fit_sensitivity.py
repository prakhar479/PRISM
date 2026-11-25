#!/usr/bin/env python
"""Batch 2 (branch repair): the generic-text Fisher of the LM loss at chosen layers (harness/sensitivity.py), cached.

    python scripts/fit_sensitivity.py --model gemma-2-2b --revision c5ebcd40... --layers 6 12 18 24
    python scripts/fit_sensitivity.py --model gemma-3-1b-pt --revision fcf18a2a... --layers 7 13 17 22 \\
        --generic generic/pile10k_6000x128_skip100_g3.json
    python scripts/fit_sensitivity.py ... --smoke        # the 4-document corpus the runner uses with --smoke (isolated caches)

One forward + backward pass per batch of windows serves every requested layer. The first harness.sensitivity.SENS_WINDOWS (2000)
documents of the generic fit set are used, i.e. the same documents the add-on fitter's first 262k rows come from. The cache
entry's key holds the model spec, the corpus hash, the layer and the estimator constants: a config's add-ons find it through
Runner.sensitivity_entry. Finished layers are skipped. Run it BEFORE the config's stage 2 (queue jobs S_*).
Output: <cache_root>/sensitivity/<key>/{shard_00000.safetensors (F, float32 [d, d]), _COMPLETE (trace, n_rows)}.
"""
import argparse
import json
import logging
import os
import sys

import _bootstrap  # noqa: F401

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from harness import provenance
from harness.profile import REPO_ROOT, load_profile

log = logging.getLogger("harness")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--revision", default=None, help="pinned model commit (not needed with --tiny)")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--generic", default="generic/pile10k_6000x128_skip100.json")
    ap.add_argument("--profile", default=os.environ.get("HL_PROFILE", "server"))
    ap.add_argument("--batch", type=int, default=4, help="windows per backward pass (halved on out-of-memory)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    prof = load_profile("dev" if a.tiny else a.profile)
    prof.apply_env()
    import torch
    from harness import sensitivity as Sn
    from harness.cache import Cache
    from harness.device import empty_cache, is_oom, resolve_device, set_threads
    from harness.model import ModelSpec, build_model

    git = provenance.git_state()
    if git["dirty"] and not (a.allow_dirty or a.tiny):
        raise SystemExit("working tree is dirty: commit, or pass --allow-dirty")
    g = json.loads((REPO_ROOT / "data" / a.generic).read_text())
    ids = g["ids"][:4] if a.smoke else g["ids"]
    corpus = {"ids": ids, "hash": g["hash"] + ("-smoke" if a.smoke else ""), "skip": int(g["skip_positions"])}   # as harness.run.Runner
    device = torch.device("cpu") if a.tiny else resolve_device(prof.device)
    set_threads(None)
    if a.tiny:
        vocab = 256_000 if a.model == "gemma-2-2b" else 262_208
        spec = ModelSpec(name="tiny", dtype="float32", tiny_vocab=vocab, tiny_layers=max(a.layers) + 2)
        cache = Cache(REPO_ROOT / "tests" / "_tiny_cache")
        corpus["ids"] = [[min(t, vocab - 1) for t in r] for r in ids[:8]]
    else:
        if not a.revision:
            raise SystemExit("--revision is required for a real model")
        spec = ModelSpec(name=a.model, revision=a.revision, dtype=prof.model_dtype)
        cache = prof.cache()
    todo = [L for L in a.layers if not Sn.entry(cache, spec, corpus["hash"], L, corpus["skip"]).complete]
    if not todo:
        log.info("sensitivity: layers %s already cached", a.layers)
        return 0
    model = build_model(spec, device, offload_embed=False if a.tiny else prof.offload_embed)
    batch = a.batch
    while True:
        try:
            Sn.fit_cached(model, spec, cache, corpus, todo, device, batch=batch)
            break
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and batch > 1:
                empty_cache()
                batch //= 2
                log.warning("out of memory: retrying with %d windows per pass", batch)
                continue
            raise
    for L in todo:
        info = json.loads((Sn.entry(cache, spec, corpus["hash"], L, corpus["skip"]).dir / "_COMPLETE").read_text())
        log.info("layer %d: trace(F) %.4g over %d rows", L, info["trace"], info["n_rows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

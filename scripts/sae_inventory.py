#!/usr/bin/env python
"""Enumerate Gemma Scope residual SAEs (widths x average_l0 variants per layer) from the HF repo tree.

    python scripts/sae_inventory.py --out configs/sae_inventory.json            # whole repo, current main
    python scripts/sae_inventory.py --revision <sha> --layers 8-18 --out ...
Writes {repo, revision, created, inventory: {layer: {width: [l0, ...]}}}. Metadata only; nothing is downloaded.
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness.sae import SAE_REPO, parse_tree


def layer_set(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=SAE_REPO)
    ap.add_argument("--revision", default=None, help="commit sha (default: resolve main and record it)")
    ap.add_argument("--layers", default=None, help="e.g. 8-16,18")
    ap.add_argument("--out", default="configs/sae_inventory.json")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    rev = a.revision or api.model_info(a.repo).sha
    want = layer_set(a.layers)
    paths = []
    for e in api.list_repo_tree(a.repo, revision=rev):
        name = e.path
        if not name.startswith("layer_") or (want is not None and int(name.split("_")[1]) not in want):
            continue
        for w in api.list_repo_tree(a.repo, path_in_repo=name, revision=rev):
            paths += [v.path for v in api.list_repo_tree(a.repo, path_in_repo=w.path, revision=rev)]
    inv = parse_tree(paths)
    out = {"repo": a.repo, "revision": rev, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "inventory": {str(k): v for k, v in inv.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{a.repo}@{rev[:10]}: {len(inv)} layers -> {a.out}")
    for L, ws in inv.items():
        print(f"  layer {L}: " + "; ".join(f"{w} {v}" for w, v in ws.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

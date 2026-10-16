#!/usr/bin/env python
"""Download gemma-2-2b and every Gemma Scope SAE the given configs need, BEFORE any compute.

Surfaces gated-repo / network / missing-id problems in minutes instead of hours into the run.
Unavailable (layer, width) pairs are reported in <out>/prefetch.json; they are skipped later, not fatal.
"""
import argparse
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.models.sae_loader import resolve_sae_id
from src.utils import load_config, save_json, setup_logging


def needed(cfg: dict) -> set:
    need = {(l, w) for l in cfg["layers"] for w in cfg["widths"]}
    need |= {(l, w) for l in cfg["geometry_layers"] for w in cfg["geometry_widths"]}
    return need


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="results/prefetch")
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    log = setup_logging(Path(a.out) / "prefetch.log")
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    report = {"ok": [], "missing": [], "model": None}
    cfg0 = load_config(a.configs[0])
    if not a.skip_model:
        snapshot_download("google/" + cfg0["model_name"], token=token,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"])
        report["model"] = cfg0["model_name"]
        log.info("model %s downloaded", cfg0["model_name"])
    if a.configs and not a.skip_model:
        try:
            import datasets  # noqa: F401
        except ImportError:
            log.warning("`datasets` not installed: generic_text_svd falls back to parquet via huggingface_hub")
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError:   # older SAELens layout
        from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

    directory = get_pretrained_saes_directory()
    need = set()
    for c in a.configs:
        need |= needed(load_config(c))
    for layer, width in sorted(need, key=lambda x: (x[0], x[1])):
        try:
            rel, sae_id = resolve_sae_id(layer, width, load_config(a.configs[0]).get("sae_release", "gemma-scope-2b-pt-res-canonical"))
            info = directory[rel]
            path = info.saes_map[sae_id]
            snapshot_download(info.repo_id, allow_patterns=[f"{path}/*"], token=token)
            report["ok"].append({"layer": layer, "width": width, "release": rel, "sae_id": sae_id, "path": path})
            log.info("SAE L%d %s -> %s/%s", layer, width, info.repo_id, path)
        except Exception as e:  # noqa: BLE001
            log.error("SAE L%d %s unavailable: %s: %s", layer, width, type(e).__name__, e)
            report["missing"].append({"layer": layer, "width": width, "error": str(e)})
    save_json(report, Path(a.out) / "prefetch.json")
    print(f"prefetch: {len(report['ok'])} SAEs ok, {len(report['missing'])} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())

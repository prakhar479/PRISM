#!/usr/bin/env python
"""Build configs/dict_registry.yaml: the non-Gemma-Scope dictionaries of the breadth pilot, at pinned revisions.

    python scripts/build_dict_registry.py               # needs network (Hugging Face API; small json files only)

Groups (read by scripts/make_pilot_configs.py):
  p2_L12 / p2_L19  SAEBench gemma-2-2b resid_post sweeps (Karvonen et al., ICML 2025): every architecture present at 4k (2^12),
                   16k (2^14) and 65k (2^16), the two trainers whose L0 is closest (log ratio) to 40 and to 80
  p9_L12           the Bhalla et al. temporal SAE (Matryoshka-BatchTopK format, gemma-2-2b L12, 16k)
  p8_g3            Gemma Scope 2 for gemma-3-1b-pt: resid_post, 16k, l0 'medium', at its four layers
Only config.json / eval_results.json are downloaded here; weights are fetched by the queue's prefetch job.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401

from harness.profile import REPO_ROOT

SAEBENCH = ["canrager/saebench_gemma-2-2b_width-2pow14_date-0107", "canrager/saebench_gemma-2-2b_width-2pow16_date-0107",
            "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108"]
ARCH = {"standard_new": "standard", "standard": "standard", "p_anneal": "p_anneal", "panneal": "p_anneal", "top_k": "topk",
        "topk": "topk", "batch_top_k": "batchtopk", "batchtopk": "batchtopk", "matryoshka_batch_top_k": "matryoshka",
        "matryoshkabatchtopk": "matryoshka", "jump_relu": "jumprelu", "jumprelu": "jumprelu", "gated": "gated", "gatedsae": "gated"}
TEMPORAL = ("alex-oesterling/temporal-saes", "trainer_0")
GS2 = ("google/gemma-scope-2-1b-pt", (7, 13, 17, 22))
TARGETS = (40, 80)
LAYERS = (12, 19)


def arch_of(path: str):
    head = path.split("/")[0].lower()
    m = re.match(r"gemma-2-2b_(.+?)_width-2pow\d+_date-\d+", head) or re.match(r"(.+?)_gemma-2-2b__\d+", head)
    return ARCH.get(m.group(1).replace("-", "_")) if m else None


def find_l0(cfg: dict, ev: dict):
    tr = cfg.get("trainer", cfg)
    for k in ("k", "target_l0"):
        if tr.get(k):
            return float(tr[k]), f"config.{k}"

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("l0", "L0") and isinstance(v, (int, float)):
                    return float(v)
                r = walk(v)
                if r is not None:
                    return r
        return None
    v = walk(ev or {})
    return (v, "eval_results.l0") if v is not None else (None, None)


def fmt_width(n: int) -> str:
    """Gemma Scope's names (65536 -> 65k), so widths read the same across sources."""
    named = {4096: "4k", 16384: "16k", 65536: "65k", 131072: "131k", 262144: "262k", 524288: "524k", 1048576: "1m"}
    return named.get(n, f"{n // 1024}k")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "configs" / "dict_registry.yaml"))
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    reg, notes = {}, []

    def jget(repo, f, rev):
        try:
            return json.loads(Path(hf_hub_download(repo, f, revision=rev)).read_text())
        except Exception as e:  # noqa: BLE001
            notes.append(f"{repo}/{f}: {type(e).__name__}")
            return {}

    for repo in SAEBENCH:
        rev = api.model_info(repo).sha
        files = api.list_repo_files(repo, revision=rev)
        dirs = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/ae.pt")})
        cands = {}
        for d in dirs:
            m = re.search(r"resid_post_layer_(\d+)/trainer_(\d+)$", d)
            if not m or int(m.group(1)) not in LAYERS:
                continue
            arch = arch_of(d)
            if arch is None:
                notes.append(f"{repo}/{d}: unknown architecture")
                continue
            cfg = jget(repo, f"{d}/config.json", rev)
            ev = jget(repo, f"{d}/eval_results.json", rev) if f"{d}/eval_results.json" in files else {}
            l0, src = find_l0(cfg, ev)
            tr = cfg.get("trainer") or {}
            n = int(tr.get("dict_size") or 0)
            pen = tr.get("l1_penalty", tr.get("sparsity_penalty"))
            if not n or (l0 is None and pen is None):
                notes.append(f"{repo}/{d}: no L0, no penalty or no dict_size")
                continue
            cands.setdefault((int(m.group(1)), arch, n), []).append({"dir": d, "l0": l0, "l0_source": src, "trainer": int(m.group(2)),
                                                                    "penalty": pen})
        for (L, arch, n), cs in sorted(cands.items()):
            picked = []
            if all(c["l0"] for c in cs):
                for t in TARGETS:
                    c = min(cs, key=lambda c: (abs(math.log(c["l0"] / t)), c["trainer"]))
                    if c not in picked:
                        picked.append(c)
            else:                       # no recorded L0 (L1-penalised sweeps without eval_results): two mid-sweep penalties;
                cs = sorted(cs, key=lambda c: (float(c["penalty"] or 0), c["trainer"]))     # the realised L0 is measured in P2
                for pos in (round(0.4 * (len(cs) - 1)), round(0.6 * (len(cs) - 1))):
                    c = {**cs[pos], "l0": 0, "l0_source": f"unknown: penalty rank {pos} of {len(cs)} (l1 {cs[pos]['penalty']})"}
                    if c["trainer"] not in [p["trainer"] for p in picked]:
                        picked.append(c)
            for c in picked:
                name = f"sb_{arch}_{fmt_width(n)}_L{L}_" + (f"l0{int(round(c['l0']))}" if c["l0"] else f"t{c['trainer']}")
                files_ = [f"{c['dir']}/ae.pt", f"{c['dir']}/config.json"]
                reg[name] = {"group": f"p2_L{L}", "layer": L, "width": fmt_width(n), "l0": int(round(c["l0"])), "l0_source": c["l0_source"],
                             "arch": f"dl_{arch}", "repo": repo, "revision": rev, "files": files_,
                             "source": "SAEBench (Karvonen et al., ICML 2025)", "trainer": c["trainer"]}
        time.sleep(0.2)

    repo, tdir = TEMPORAL
    try:
        rev = api.model_info(repo).sha
        cfg = jget(repo, f"{tdir}/config.json", rev)
        tr = cfg.get("trainer", {})
        L = int(tr.get("layer", 12))
        n = int(tr.get("dict_size") or 16384)
        reg[f"temporal_bhalla_{fmt_width(n)}_L{L}"] = {
            "group": f"p9_L{L}", "layer": L, "width": fmt_width(n), "l0": int(tr.get("k") or 20), "l0_source": "config.k",
            "arch": "dl_matryoshka", "repo": repo, "revision": rev, "files": [f"{tdir}/ae.pt", f"{tdir}/config.json"],
            "source": "Temporal SAE (Bhalla et al., arXiv 2511.05541), TemporalMatryoshkaBatchTopKSAE; encoded like Matryoshka-BatchTopK"}
    except Exception as e:  # noqa: BLE001
        notes.append(f"temporal SAE: {type(e).__name__}: {e}")

    repo, layers = GS2
    try:
        rev = api.model_info(repo).sha
        for L in layers:
            d = f"resid_post/layer_{L}_width_16k_l0_medium"
            cfg = jget(repo, f"{d}/config.json", rev)
            reg[f"gs2_1b_L{L}_16k_medium"] = {"group": "p8_g3", "layer": L, "width": "16k", "l0": int(cfg.get("l0") or 0),
                                             "l0_source": "config.l0", "arch": "gs2", "repo": repo, "revision": rev,
                                             "files": [f"{d}/params.safetensors", f"{d}/config.json"], "model": "gemma-3-1b-pt",
                                             "source": "Gemma Scope 2 (Dec 2025)", "hook": cfg.get("hf_hook_point_in")}
    except Exception as e:  # noqa: BLE001
        notes.append(f"Gemma Scope 2: {type(e).__name__}: {e}")

    out = Path(a.out)
    head = ("# GENERATED by scripts/build_dict_registry.py (" + time.strftime("%Y-%m-%d") + "): non-Gemma-Scope dictionaries of the\n"
            "# breadth pilot at pinned revisions. Formats: harness/dict_formats.py. Do not edit by hand; rerun the script.\n")
    out.write_text(head + yaml.safe_dump({"dictionaries": reg, "notes": notes}, sort_keys=False, width=140))
    groups = {}
    for v in reg.values():
        groups[v["group"]] = groups.get(v["group"], 0) + 1
    print(f"{len(reg)} dictionaries {groups} -> {out}; {len(notes)} notes")
    for n_ in notes[:20]:
        print("  note:", n_)
    return 0


if __name__ == "__main__":
    sys.exit(main())

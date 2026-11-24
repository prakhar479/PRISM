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


# ------------------------------------------------------------------ P12 / P13 (Gemma 3 1B / 1B-IT, Gemma Scope 2)
GS2_PT, GS2_IT = "google/gemma-scope-2-1b-pt", "google/gemma-scope-2-1b-it"
P12_RESID = {13: [(w, l) for w in ("16k", "65k", "262k", "1m") for l in ("small", "medium", "big")] + [("262k", "medium_seed_1")],
             17: [(w, "medium") for w in ("16k", "65k", "262k", "1m")]}
# composites (harness/composite.py): name -> (layer, recipe). Recipes: ("all", folder, variant) every layer 0..L of a *_all
# folder; ("single", folder, variant) layer L only; ("clt", variant) source layers 0..L; ("xc", variant) crosscoder blocks <= L;
# a second recipe (attention SAEs, all layers 0..L) is added with "+".
P12_COMP = {
    "tc16k_small_all": [("all", "transcoder_all", "width_16k_l0_small")],
    "tc16k_big_all": [("all", "transcoder_all", "width_16k_l0_big")],
    "tc16k_small_aff_all": [("all", "transcoder_all", "width_16k_l0_small_affine")],
    "tc16k_big_aff_all": [("all", "transcoder_all", "width_16k_l0_big_affine")],
    "tc262k_small_all": [("all", "transcoder_all", "width_262k_l0_small")],
    "mlpsae16k_small_all": [("all", "mlp_out_all", "width_16k_l0_small")],
    "mlpsae16k_big_all": [("all", "mlp_out_all", "width_16k_l0_big")],
    "clt262k_medium": [("clt", "width_262k_l0_medium")],
    "clt262k_medium_aff": [("clt", "width_262k_l0_medium_affine")],
    "clt524k_medium": [("clt", "width_524k_l0_medium")],
    "tc16k_medium_single": [("single", "transcoder", "width_16k_l0_medium")],
    "tc65k_medium_single": [("single", "transcoder", "width_65k_l0_medium")],
    "tc262k_medium_single": [("single", "transcoder", "width_262k_l0_medium")],
    "tc16k_medium_aff_single": [("single", "transcoder", "width_16k_l0_medium_affine")],
    "tc16k_small_all+attn16k_small": [("all", "transcoder_all", "width_16k_l0_small"), ("all", "attn_out_all", "width_16k_l0_small")],
    "clt262k_medium+attn16k_small": [("clt", "width_262k_l0_medium"), ("all", "attn_out_all", "width_16k_l0_small")],
    "xc65k_medium": [("xc", "width_65k_l0_medium")],
    "xc262k_medium": [("xc", "width_262k_l0_medium")],
    "xc524k_medium": [("xc", "width_524k_l0_medium")],
    "xc1m_medium": [("xc", "width_1m_l0_medium")],
    "xc65k_big": [("xc", "width_65k_l0_big")],
    "xc262k_big": [("xc", "width_262k_l0_big")],
}
P12_COMP_L17 = ("tc16k_small_all", "clt262k_medium", "xc65k_medium", "xc262k_medium")
P13_RESID = {L: [("16k", "medium"), ("262k", "medium")] for L in (7, 13, 17, 22)}
P13_COMP = ("tc16k_small_all", "clt262k_medium", "xc65k_medium")
XC_LAYERS = (7, 13, 17, 22)


def comp_files(L: int, recipe) -> list:
    files = []
    for r in recipe:
        if r[0] == "all":
            files += [f"{r[1]}/layer_{l}_{r[2]}/params.safetensors" for l in range(L + 1)]
        elif r[0] == "single":
            files.append(f"{r[1]}/layer_{L}_{r[2]}/params.safetensors")
        elif r[0] == "clt":
            files += [f"clt/{r[1]}/params_layer_{l}.safetensors" for l in range(L + 1)]
        elif r[0] == "xc":
            tag = "_".join(map(str, XC_LAYERS))
            files += [f"crosscoder/layer_{tag}_{r[1]}/params_layer_{i}.safetensors" for i, l in enumerate(XC_LAYERS) if l <= L]
    return files


def g3_entries(api, jget, notes, existing=()) -> dict:
    """P12 (gemma-3-1b-pt) and P13 (gemma-3-1b-it) dictionaries: residual SAEs and composites, pinned revisions; every
    file is checked to exist in the repo. Names already in the registry (P8's gs2_1b_L13/L17_16k_medium) are kept as they
    are (same dictionary, same cache keys); scripts/make_pilot_configs.py adds them to P12 by name."""
    reg = {}
    for repo, model, groups in ((GS2_PT, "gemma-3-1b-pt", ("p12_resid", "p12_comp")), (GS2_IT, "gemma-3-1b-it", ("p13_resid", "p13_comp"))):
        rev = api.model_info(repo).sha
        have = set(api.list_repo_files(repo, revision=rev))
        pre = "gs2_1b" if repo == GS2_PT else "gs2_1bit"
        resid = P12_RESID if repo == GS2_PT else P13_RESID
        for L, vs in resid.items():
            for w, l0 in vs:
                d = f"resid_post/layer_{L}_width_{w}_l0_{l0}"
                if f"{d}/params.safetensors" not in have:
                    notes.append(f"{repo}/{d}: missing")
                    continue
                cfg = jget(repo, f"{d}/config.json", rev)
                name = f"{pre}_L{L}_{w}_{l0}"
                if name in existing:
                    continue
                reg[name] = {"group": groups[0], "layer": L, "width": w, "l0": int(cfg.get("l0") or 0), "l0_source": "config.l0",
                             "arch": "gs2", "repo": repo, "revision": rev, "files": [f"{d}/params.safetensors", f"{d}/config.json"],
                             "model": model, "source": "Gemma Scope 2 (Dec 2025)", "hook": cfg.get("hf_hook_point_in")}
        comps = [(13, k) for k in P12_COMP] + [(17, k) for k in P12_COMP_L17] if repo == GS2_PT else [(13, k) for k in P13_COMP]
        for L, k in comps:
            recipe = P12_COMP[k]
            files = comp_files(L, recipe)
            miss = [f for f in files if f not in have]
            if miss:
                notes.append(f"{repo} composite {k} L{L}: {len(miss)} files missing, e.g. {miss[0]}")
                continue
            first = files[-1].rsplit("/", 1)[0]
            cfg = jget(repo, f"{first}/config.json", rev)
            width = re.search(r"width_(\d+[km])", files[0]).group(1)
            reg[f"{pre}c_L{L}_{k}"] = {"group": groups[1], "layer": L, "width": width, "l0": int(cfg.get("l0") or 0),
                                       "l0_source": "config.l0 (per sublayer dictionary)", "arch": "gs2c", "repo": repo,
                                       "revision": rev, "files": files, "model": model, "recipe": [list(r) for r in recipe],
                                       "source": "Gemma Scope 2 (Dec 2025), composite (harness/composite.py)"}
    return reg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "configs" / "dict_registry.yaml"))
    ap.add_argument("--g3-only", action="store_true",
                    help="keep the existing registry and (re)add only the P12/P13 Gemma Scope 2 groups (branch pilot-g3)")
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

    if a.g3_only:
        old = yaml.safe_load(Path(a.out).read_text()) or {}
        reg = {k: v for k, v in (old.get("dictionaries") or {}).items() if not str(v.get("group", "")).startswith(("p12_", "p13_"))}
        notes = [n for n in (old.get("notes") or []) if "gemma-scope-2" not in n]
        reg.update(g3_entries(api, jget, notes, set(reg)))
        out = Path(a.out)
        head = ("# GENERATED by scripts/build_dict_registry.py (" + time.strftime("%Y-%m-%d") + "): non-Gemma-Scope dictionaries of the\n"
                "# breadth pilot at pinned revisions. Formats: harness/dict_formats.py. Do not edit by hand; rerun the script.\n")
        out.write_text(head + yaml.safe_dump({"dictionaries": reg, "notes": notes}, sort_keys=False, width=140))
        groups = {}
        for v in reg.values():
            groups[v["group"]] = groups.get(v["group"], 0) + 1
        print(f"{len(reg)} dictionaries {groups} -> {out}; notes: {notes[-10:]}")
        return 0

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

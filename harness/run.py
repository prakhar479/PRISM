"""Experiment runner.

    python -m harness.run --config configs/E2.yaml --profile run [--stage 1|2|3|all] [--smoke] [--resume] [--dry-run]
    python -m harness.run --config configs/E2.yaml --profile dev --tiny --smoke     # plumbing test, never a result

Output: <runs_root>/<exp_id>/<run_hash>/{manifest.json, per_pair.parquet, summary.csv, summary.json, log.txt,
parts/, directions/}. run_hash = hash(config, git commit, smoke, tiny, model dtype), so rerunning the same release
resumes the same directory. Stages communicate only through content-addressed caches under <cache_root>.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")      # same numbering as nvidia-smi (see scripts/_bootstrap.py)

import numpy as np
import pandas as pd
import torch

from harness import conditions as C
from harness import directions as Dr
from harness import memory, provenance
from harness import parallel as Par
from harness import families as F
from harness import tasks as T
from harness.batching import Budget
from harness.cache import Cache, atomic_write_text
from harness.config import ExpConfig, das_rank, das_spec, parse_das  # noqa: F401
from harness.device import (cap_memory, device_memory, dtype_name, empty_cache, peak_bytes, probe_machine, reset_peak,
                            resolve_device, set_threads)
from harness.model import ModelSpec, build_model
from harness.model import stages as S
from harness.profile import REPO_ROOT, Profile, load_profile
from harness.sae import SAEId, load_params, params_path, write_random_params

log = logging.getLogger("harness")
UNIT_SIZE = 8                      # conditions per resumable stage-3 unit
TINY_D, TINY_SAE = 64, 512
RAM_MARGIN = 1.25                  # an SAE is loaded only if 1.25 x its weights fit in available host RAM


def _none_key(**kw) -> dict:
    return {"estimator": "", "source": "", "r": -1, "seed": -1, **kw}


@dataclass
class Bundle:
    """Everything fitted on the train pairs of one (task, layer, SAE)."""
    fitset: Dr.FitSet
    train_idx: List[int]
    spans: C.Spans
    mean_err: C.MeanError
    cov: C.CovNoise
    dirs: Dict[str, Dr.Direction] = field(default_factory=dict)
    greedy: Optional[C.GreedyLatents] = None            # E5: latents picked on train pairs
    mean_diff: Optional[C.MeanDiff] = None              # E10: train mean of x_A - x_B per row key (steering vector)


def subthresh_key(alpha: float, slot: str, k: int) -> dict:
    """Per-pair key of an E5 condition: estimator all|greedy, source 'alpha=<a>|slot=<B|meanPos>', r = greedy k."""
    return dict(condition="subthresh", estimator="greedy" if k else "all", source=f"alpha={alpha:.2f}|slot={slot}",
                r=int(k), seed=-1)


def pick_device(prof, args) -> torch.device:
    """CPU for tiny runs, dry runs and --finalize-only (a CPU-only queue job: no GPU slot, CUDA_VISIBLE_DEVICES is empty);
    otherwise the profile's device (an unavailable accelerator is an error)."""
    if getattr(args, "tiny", False) or getattr(args, "dry_run", False) or getattr(args, "finalize_only", False):
        return torch.device("cpu")
    return resolve_device(prof.device)


MIN_TRAIN_PAIRS = 20        # fewer train pairs than this: the cell is skipped (recorded), never fitted on a handful of pairs


def select_cells(cells, selectors):
    """Cells matching ANY selector (smoke / test subsets; `--only-sae`). A selector is a dictionary-name prefix, or key=value:
    layer=13, task=entity_track, width=65k, addon=sens8 (prefix of the add-on name; addon=NONE: cells without one), dict=<prefix>."""
    def match(c, sel):
        k, _, v = sel.partition("=")
        if not _:
            return (c.sae.dict or "").startswith(sel)
        if k == "layer":
            return c.layer == int(v)
        if k == "task":
            return c.task == v
        if k == "width":
            return c.sae.width == v
        if k == "dict":
            return (c.sae.dict or "").startswith(v)
        if k == "addon":
            return not c.sae.addon if v == "NONE" else (c.sae.addon or "").startswith(v)
        raise ValueError(f"unknown selector {sel!r} (prefix | layer= | task= | width= | dict= | addon=)")
    return [c for c in cells if any(match(c, sel) for sel in selectors)]


class CellSkip(RuntimeError):
    """A cell that cannot be computed for a data reason (too few train pairs passing the fit filter, e.g. a language the
    model cannot do). Stage 3 and finalize record it in the cell's checks and move on; every other error still fails."""


class Runner:
    def __init__(self, cfg: ExpConfig, prof: Profile, args) -> None:
        self.cfg, self.prof, self.args = cfg, prof, args
        self.tiny, self.smoke = bool(args.tiny), bool(args.smoke)
        self.finalizing = False
        self.worker = getattr(args, "worker", None) or Par.worker_id()      # multi-GPU stage-3 worker id
        self.resources: dict = {}                                            # measured machine + resolved autos
        # a dry run plans for the profile's machine without touching any device
        self.device = pick_device(prof, args)
        self.cache = prof.cache() if not self.tiny else Cache(REPO_ROOT / "tests" / "_tiny_cache")
        inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())
        self.inventory, self.sae_revision = inv["inventory"], inv["revision"]
        from harness import dicts as Dct
        self.registry = Dct.load_registry()
        deepest = max([c.layer for c in cfg.cells] + [0])
        tiny_vocab = 256_000 if cfg.model == "gemma-2-2b" else 262_208          # Gemma 3 token ids reach 262k
        self.spec = (ModelSpec(name="tiny", dtype="float32", tiny_vocab=tiny_vocab, tiny_layers=deepest + 2) if self.tiny
                     else ModelSpec(name=cfg.model, revision=cfg.model_revision, dtype=prof.model_dtype))
        self.rows: Dict[str, List[dict]] = {}
        self.set_meta: Dict[str, dict] = {}
        self._check_composites()
        for task, rel in cfg.prompt_sets.items():
            rows, meta = T.load_set(prof.root("data_root") / rel)
            if self.smoke:
                k = 3 * cfg.smoke_pairs
                rows = [r for r in rows if r["split"] == "train" and r["rank"] < k] + \
                       [r for r in rows if r["split"] == "test" and r["rank"] < k]
            self.rows[task], self.set_meta[task] = rows, {**meta, "hash": T.set_hash(rows), "file": rel}
            if any(not F.is_pilot(r.get("family", "pilot")) for r in rows):
                for c in cfg.cells:
                    if c.task == task and not c.eval_families:
                        raise ValueError(f"{rel} contains fresh-family rows: cell {task} L{c.layer} must set "
                                         "fit_families and eval_families")
        self.generic = None
        if cfg.generic:
            g = json.loads((prof.root("data_root") / cfg.generic["set"]).read_text())
            ids = g["ids"][: 4] if self.smoke else g["ids"]
            self.generic = {"ids": ids, "hash": g["hash"] + ("-smoke" if self.smoke else ""), "skip": g["skip_positions"],
                            "file": cfg.generic["set"]}
        self.n_test = cfg.smoke_pairs if self.smoke else cfg.n_test
        self.n_train = cfg.smoke_pairs if self.smoke else cfg.n_train
        self.budget = Budget(max_tokens=prof.max_batch_tokens, max_rows=prof.max_batch_size)
        self.model = None
        self.bundles: Dict[tuple, Bundle] = {}
        self.code_set_log: List[dict] = []
        self._cur_sid, self._cur_te = None, ()
        self.timings: Dict[str, float] = {}
        self.checks: List[dict] = []
        gs = provenance.git_state()
        ident = {"config": cfg.to_dict(), "commit": gs["commit"], "smoke": self.smoke, "tiny": self.tiny,
                 "model_dtype": self.spec.dtype}
        self.run_hash = provenance.canonical_hash(ident)
        root = REPO_ROOT / "tests" / "_tiny_runs" if self.tiny else prof.root("runs_root")
        self.run_dir = root / cfg.exp_id / self.run_hash
        self.git = gs

    # ------------------------------------------------------------------ identifiers
    def sae_id(self, layer: int, spec) -> SAEId:
        """Gemma Scope cells resolve exactly as before (same key, same to_dict); registry / trained / add-on dictionaries
        resolve through harness.dicts (DictId)."""
        from harness import dicts as Dct
        did = Dct.resolve(layer, spec, self.inventory, self.sae_revision, self.registry)
        if did.source == "trained" and not self.tiny:
            # revision = digest of the trained weights, so every cache keyed by sae_key follows a retrained dictionary
            # (None until the training job has run: stage 1, plans and prefetch never read SAE caches)
            import dataclasses
            did = dataclasses.replace(did, revision=Dct.trained_digest(self.prof.root("cache_root"), did.name))
        return did

    @staticmethod
    def is_composite(sid) -> bool:
        return getattr(sid, "arch", "") == "gs2c"

    def comp_parts(self, sid):
        from harness import composite as Cm
        return Cm.parse_parts(sid.files)

    def _check_composites(self) -> None:
        """Composite dictionaries (P12) give x_hat / eps at the cell's site only: no latent codes, no generic text."""
        from harness import composite as Cm
        from harness.config import CODE_CONDITIONS
        comp = [c for c in self.cfg.cells if self.is_composite(self.sae_id(c.layer, c.sae))]
        if not comp:
            return
        d = self.cfg.directions
        bad = []
        if self.cfg.subthresh:
            bad.append("subthresh")
        if self.cfg.generic_l0:
            bad.append("generic_l0")
        if d and any(n in CODE_CONDITIONS for n in d.conditions):
            bad.append("code-level conditions")
        if d and any(s == "generic" for s in d.sources):
            bad.append("generic direction sources")
        if any(c.sae.addon for c in comp):
            bad.append("add-ons")
        if bad:
            raise ValueError(f"composite dictionaries do not support: {', '.join(bad)}")
        for c in comp:
            Cm.validate(self.comp_parts(self.sae_id(c.layer, c.sae)), c.layer)

    def stage1_sites(self) -> Dict[str, list]:
        """Per task: the cell layers (resid_post) plus every (layer, site) a composite cell reads."""
        from harness import composite as Cm
        out = {t: list(Ls) for t, Ls in self.cfg.layers_by_task().items()}
        for c in self.cfg.cells:
            sid = self.sae_id(c.layer, c.sae)
            if self.is_composite(sid):
                extra = sorted(Cm.needed_sites(self.comp_parts(sid), c.layer) - {(L, "resid_post") for L in out[c.task]})
                out[c.task] += [k for k in extra if k not in out[c.task]]
        return out

    def attn_layers(self) -> List[int]:
        from harness import composite as Cm
        L = set()
        for c in self.cfg.cells:
            sid = self.sae_id(c.layer, c.sae)
            if self.is_composite(sid):
                L |= set(Cm.attn_layers(self.comp_parts(sid)))
        return sorted(L)

    def sae_key(self, sid: SAEId) -> dict:
        k = {**sid.to_dict(), "tiny": self.tiny}
        if getattr(sid, "addon", ""):
            from harness.dicts import ADDON_LAMBDA, parse_addon
            k["addon_fit"] = {"corpus": self.generic["hash"] if self.generic else None, "lam": ADDON_LAMBDA}
            p = parse_addon(sid.addon)
            if p["needs_F"] and self.generic:              # the add-on follows the Fisher it was chosen with
                k["addon_fit"]["sens"] = self.sensitivity_entry(sid.layer).key
        return k

    def sensitivity_entry(self, layer: int):
        """Generic-text Fisher of the LM loss at `layer` (harness.sensitivity; filled by scripts/fit_sensitivity.py)."""
        from harness import sensitivity as Sn
        return Sn.entry(self.cache, self.spec, self.generic["hash"], layer, self.generic["skip"])

    def need_codes(self) -> bool:
        """Active codes of A and B are cached in stage 2 when a code-level condition is evaluated (P0)."""
        from harness.config import CODE_CONDITIONS
        d = self.cfg.directions
        return bool(d and any(n in CODE_CONDITIONS for n in d.conditions))

    def stage2_jobs(self) -> List[tuple]:
        """(source kind, task or 'generic', layer, SAEId) needed by the cells and their direction sources."""
        jobs = []
        d = self.cfg.directions
        # several SAEs per (task, layer) are allowed (E4 widths); `layer:<L>` sources use the first cell at layer L
        cell_sae = {}
        for c in self.cfg.cells:
            cell_sae.setdefault((c.task, c.layer), self.sae_id(c.layer, c.sae))
        for c in self.cfg.cells:
            sid = self.sae_id(c.layer, c.sae)
            jobs.append(("task", c.task, c.layer, sid))
            if self.cfg.generic_l0:
                jobs.append(("generic", "generic", c.layer, sid))
            if d and "other_task" in d.sources:
                jobs += [("task", t, c.layer, sid) for t in self.cfg.prompt_sets if t != c.task]
            if d and "generic" in d.sources:
                jobs.append(("generic", "generic", c.layer, sid))
            if d:
                for s in d.sources:
                    if s.startswith("layer:"):
                        L2 = int(s.split(":")[1])
                        if (c.task, L2) not in cell_sae:
                            raise ValueError(f"source {s} needs a {c.task} cell at layer {L2}")
                        jobs.append(("task", c.task, L2, cell_sae[(c.task, L2)]))
        seen, out = set(), []
        for j in jobs:
            k = (j[0], j[1], j[2], j[3].key)
            if k not in seen:
                seen.add(k)
                out.append(j)
        return out

    # ------------------------------------------------------------------ dry run / plan
    def plan(self) -> dict:
        cap = int(self.prof.gpu_mem_cap_gb * 2 ** 30) if self.prof.gpu_mem_cap_gb else None
        s1, s3 = memory.stage1(self.prof), memory.stage3(self.prof, max(r["T"] for rs in self.rows.values() for r in rs))
        jobs = [{"stage": 1, "what": f"{t}: {len(self.rows[t])} pairs x 2 prompts, layers {L}", "peak_gb": round(s1.peak_gb, 2),
                 "fits": memory.fits(s1, cap)} for t, L in self.cfg.layers_by_task().items()]
        if self.generic:
            layers = sorted({c.layer for c in self.cfg.cells})
            jobs.append({"stage": 1, "what": f"generic: {len(self.generic['ids'])} docs, layers {layers}",
                         "peak_gb": round(s1.peak_gb, 2), "fits": memory.fits(s1, cap)})
        for kind, t, L, sid in self.stage2_jobs():
            e2 = memory.stage2(self.prof, sid.width, cap)
            jobs.append({"stage": 2, "what": f"{t} L{L} {sid.key}", "placement": e2.placement,
                         "block_latents": e2.block_latents, "peak_gb": round(e2.peak_gb, 2), "fits": memory.fits(e2, cap),
                         "sae_dtype": memory.sae_dtype_for(sid.width, self.prof),
                         "host_ram_gb": round(memory.sae_bytes(sid.width, memory.sae_dtype_for(sid.width, self.prof)) / 2 ** 30, 2)})
        for c in self.cfg.cells:
            n_cond = len(self.condition_specs_count(c))
            jobs.append({"stage": 3, "what": f"{c.task} L{c.layer} {self.sae_id(c.layer, c.sae).key}{c.tag()}: {n_cond} conditions x <= "
                         f"{self.n_test * len(self.cfg.filters)} test pairs", "peak_gb": round(s3.peak_gb, 2),
                         "fits": memory.fits(s3, cap), "n_conditions": n_cond})
        return {"exp_id": self.cfg.exp_id, "run_hash": self.run_hash, "cap_gb": self.prof.gpu_mem_cap_gb, "jobs": jobs,
                "all_fit": all(j["fits"] for j in jobs)}

    def condition_specs_count(self, cell) -> list:
        """Condition keys for a cell without fitting anything (dry-run sizing)."""
        keys = [("clean_B",), ("full_A",), ("full_B",)]
        noise = {"feat_B_noise", "feat_A_noise", "feat_A_covnoise"}
        for n in self.cfg.conditions:
            if n in ("clean_B", "full_A", "full_B"):
                continue
            keys += [(n, s) for s in self.cfg.noise_seeds] if n in noise else [(n,)]
        d = self.cfg.directions
        if d:
            n_src = sum(len(self.cfg.prompt_sets) - 1 if s == "other_task" else 1 for s in d.sources)
            for n in d.conditions:
                cols = {"ridge_value": 1, "fourier_T2": 1, "fourier_T5": 2, "fourier_T10": 2, "diffmean": 1}   # as cell_specs skips r > columns
                keys += [(n, e, s, r) for e in d.estimators for s in range(n_src) for r in d.ranks
                         if not (e.startswith("das") and r != das_rank(e)) and not (n.startswith("steer_") and r == 0)
                         and r <= cols.get(e, r)]
                keys += [(n, "random", s, r) for s in range(d.random_seeds) for r in d.ranks if r > 0]
        st_ = self.cfg.subthresh
        if st_:
            keys += [("subthresh", k, a, s) for k in [0] + list(st_.greedy_ks) for a in st_.alphas for s in st_.slot_errors]
        return keys

    # ------------------------------------------------------------------ stage 1
    def load_model(self):
        if self.model is None:
            t0 = time.time()
            tok = None
            need = 0 if self.tiny else 2 * memory.model_bytes(self.spec.dtype)
            with Par.reserve_ram(self.prof.root("cache_root"), need, "model load"):
                self.model = build_model(self.spec, self.device, offload_embed=self.prof.offload_embed, tokenizer=tok)
            self.timings["model_load_s"] = round(time.time() - t0, 1)
        return self.model

    def free_model(self):
        self.model = None
        empty_cache()

    def run_stage1(self):
        model = self.load_model()
        for task, layers in self.stage1_sites().items():
            t0 = time.time()
            S.stage1(model, self.spec, self.rows[task], self.set_meta[task]["hash"], layers, self.cache, self.budget, self.device)
            self.timings[f"stage1_{task}_s"] = round(time.time() - t0, 1)
        if self.attn_layers():
            S.write_sublayer_params(model, self.spec, self.cache, self.attn_layers())
        if self.generic:
            layers = self.generic_layers()
            if layers:
                S.stage1_generic(model, self.spec, self.generic["ids"], self.generic["hash"], layers, self.generic["skip"],
                                 self.cache, self.budget, self.device)

    def generic_layers(self) -> List[int]:
        """Layers whose generic-text residuals stage 1 caches: generic stage-2 jobs, plus every add-on cell's layer (the
        add-on is fitted on those residuals; no generic stage-2 output is written for it)."""
        L = {j[2] for j in self.stage2_jobs() if j[0] == "generic"}
        L |= {c.layer for c in self.cfg.cells if c.sae.addon}
        return sorted(L)

    # ------------------------------------------------------------------ stage 2
    def get_sae(self, sid: SAEId):
        from harness import dicts as Dct
        if getattr(sid, "addon", ""):
            base = self.get_sae(sid.base())
            fit = self.addon_map(sid, base)
            return Dct.AddonDict(base, sid.addon, fit["W"], fit.get("c"), fit.get("info"), fit.get("extra"))
        source = getattr(sid, "source", "gemma_scope")
        if self.is_composite(sid):
            from harness import composite as Cm
            if self.tiny:
                return Cm.Composite(self.comp_parts(sid), sid.layer, sid.width, sid.key, self._tiny_comp_fetch(sid))
            from huggingface_hub import hf_hub_download
            return Cm.Composite(self.comp_parts(sid), sid.layer, sid.width, sid.key,
                                lambda f, sid=sid: Path(hf_hub_download(sid.repo, f, revision=sid.revision)))
        if self.tiny:
            if source == "gemma_scope":
                p = write_random_params(self.cache.root / "tiny_saes" / f"{sid.key}.npz", TINY_D, TINY_SAE, seed=sid.layer, thr=0.05)
                return load_params(p, sid.layer, sid.width)
            kind = {"dl_topk": "topk", "dl_gated": "gated"}.get(getattr(sid, "arch", ""), "jumprelu")
            return Dct.random_dict(TINY_D, TINY_SAE, kind, seed=sid.layer + 7, k=8, layer=sid.layer, width=sid.width)
        from harness.memory import sae_dtype_for
        dt = sae_dtype_for(sid.width, self.prof)
        if source == "registry":
            from harness.device import dtype_of
            return Dct.load_registry_dict(sid, dtype_of(dt))
        if source == "trained":
            from harness.device import dtype_of
            return Dct.load_trained(self.prof.root("cache_root"), sid, dtype_of(dt))
        return load_params(params_path(sid), sid.layer, sid.width, dt, sid.key)

    def _tiny_comp_fetch(self, sid):
        """Random Gemma-Scope-2-shaped weight files for a composite on the tiny model (tests only)."""
        from harness import composite as Cm
        root = self.cache.root / "tiny_comp"
        d, dz, nL = TINY_D, 4 * 16, self.spec.tiny_layers          # harness.model.tiny_gemma2: 4 heads x d_head 16

        def fetch(f):
            path = root / f
            if not path.exists():
                from safetensors.torch import save_file
                part = [p for p in Cm.parse_parts([f])][0]
                import hashlib
                g = torch.Generator().manual_seed(int(hashlib.sha256(f.encode()).hexdigest()[:8], 16))
                d_in = dz if part.kind == "attn" else d
                n = 96
                t = {"w_enc": torch.randn(d_in, n, generator=g) / d_in ** 0.5, "b_enc": torch.zeros(n),
                     "threshold": torch.full((n,), 0.05), "b_dec": torch.randn(d_in, generator=g) * 0.01}
                if part.kind == "clt":
                    W = torch.randn(n, nL, d, generator=g) / n ** 0.5
                    W[:, :part.layer] = 0
                    t["w_dec"] = W
                elif part.kind == "xc":
                    W = torch.randn(n, len(part.xc_layers), d, generator=g) / n ** 0.5
                    W[:, :part.block] = 0
                    t["w_dec"] = W
                else:
                    t["w_dec"] = torch.randn(n, d_in, generator=g) / n ** 0.5
                if part.affine:
                    t["affine_skip_connection"] = torch.randn(d, d, generator=g) * 0.01
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(t, str(path))
            return path
        return fetch

    @staticmethod
    def _capped_shards(entry, max_rows: int):
        """The first shards of a generic residual entry, up to max_rows rows in total (in shard order)."""
        n = 0
        for si in entry.shards():
            if n >= max_rows:
                return
            x = entry.read_shard(si)["x"]
            x = x[: max_rows - n]
            n += len(x)
            yield x

    def addon_entry(self, sid):
        r, _ = self.generic_entries(sid.layer, sid.base())
        from harness.dicts import ADDON_LAMBDA, ADDON_MAX_ROWS, parse_addon
        parts = dict(resid=r.key, base=self.sae_key(sid.base()), mode=sid.addon, lam=ADDON_LAMBDA, max_rows=ADDON_MAX_ROWS)
        p = parse_addon(sid.addon)
        if p["kind"] not in ("lin", "pcx"):                 # batch-2 modes: keyed by their inputs and fit version
            parts["fit_version"] = 2
            if p["needs_F"]:
                parts["sens"] = self.sensitivity_entry(sid.layer).key
        return self.cache.entry("addon", **parts)

    def addon_map(self, sid, base) -> dict:
        """The add-on's fitted map (cached; fitted once from the generic set's cached residuals at the cell's layer)."""
        from harness import dicts as Dct
        if not self.generic:
            raise ValueError(f"{sid.key}: add-ons need a `generic:` block")
        ent = self.addon_entry(sid)
        if not ent.complete:
            with ent.writer_lock():
                ent.open()
                if not ent.complete:
                    r, _ = self.generic_entries(sid.layer, sid.base())
                    r.require_complete()
                    dev = self.device if not self.tiny else torch.device("cpu")
                    chunks = self._capped_shards(r, Dct.ADDON_MAX_ROWS)
                    t0 = time.time()
                    F = None
                    if Dct.parse_addon(sid.addon)["needs_F"]:
                        from harness import sensitivity as Sn
                        se = self.sensitivity_entry(sid.layer)
                        if not se.complete:
                            raise FileNotFoundError(f"{sid.key}: no generic-text Fisher for layer {sid.layer}: run "
                                                    f"scripts/fit_sensitivity.py (queue job S_*) before stage 2")
                        F = Sn.load(se)
                    import zlib
                    seed = zlib.crc32(sid.addon.encode()) % 100_000 + int(sid.layer)
                    fit = Dct.fit_addon(base, chunks, sid.addon, device=dev, F=F, seed=seed)
                    ent.write_shard(0, {"W": fit["W"], **({"c": fit["c"]} if fit["c"] is not None else {}),
                                        **{f"x_{k}": v for k, v in (fit.get("extra") or {}).items()}})
                    ent.write_json("info.json", fit["info"])
                    ent.mark_complete(fit["info"])
                    log.info("add-on %s fitted on %d generic rows (%.0fs)", sid.key, fit["info"]["n_rows"], time.time() - t0)
        sh = ent.read_shard(0)
        return {"W": sh["W"], "c": sh.get("c"), "info": ent.read_json("info.json"),
                "extra": {k[2:]: v for k, v in sh.items() if k.startswith("x_")}}

    def spans_entry(self, task, layer):
        return self.cache.entry("spans", model=self.spec.key(), set_hash=self.set_meta[task]["hash"], layer=int(layer),
                                site=self.spec.site)

    def sae_entry(self, task, layer, sid):
        return self.cache.entry("sae_out", spans=self.spans_entry(task, layer).key, sae=self.sae_key(sid))

    def codes_entry(self, task, layer, sid):
        return self.cache.entry("sae_codes", spans=self.spans_entry(task, layer).key, sae=self.sae_key(sid),
                                alpha_min=float(self.cfg.subthresh.alpha_min))

    def dec_entry(self, task, layer, sid):
        return self.cache.entry("sae_codes_dec", codes=self.codes_entry(task, layer, sid).key)

    def generic_entries(self, layer, sid):
        r = self.cache.entry("generic_resid", model=self.spec.key(), corpus=self.generic["hash"], layer=int(layer),
                             skip=self.generic["skip"], site=self.spec.site)
        return r, self.cache.entry("generic_sae", resid=r.key, sae=self.sae_key(sid))

    def run_stage2(self):
        cap = int(self.prof.gpu_mem_cap_gb * 2 ** 30) if self.prof.gpu_mem_cap_gb else None
        by_sae: Dict[str, list] = {}
        for j in self.stage2_jobs():
            by_sae.setdefault(j[3].key, []).append(j)
        if self.model is not None:                          # stage 2 never shares the GPU with the model
            self.free_model()
        skipped = self.skipped_saes()
        groups = list(by_sae.items())
        sl = getattr(self.args, "stage2_slice", None)
        if sl:                                             # disjoint SAE groups per process (multi-GPU warm-up)
            i, n = (int(x) for x in sl.split("/"))
            groups = [g for k, g in enumerate(sorted(groups, key=lambda kv: kv[0])) if k % n == i]
        for _, jobs in groups:                             # one SAE resident at a time
            sid = jobs[0][3]
            est = memory.stage2(self.prof, sid.width, cap)
            if all(self.stage2_done(j) for j in jobs):
                continue
            need = memory.sae_bytes(sid.width, memory.sae_dtype_for(sid.width, self.prof))
            from src.models.runtime import available_ram
            # in parallel mode a short dip in free RAM is waited out (reserve_ram), never recorded as a skip; only an
            # SAE that cannot fit the machine's whole budget is skipped
            avail = Par.ram_budget() if Par.parallel_mode() else available_ram()
            if not self.tiny and need * RAM_MARGIN > avail:
                skipped[sid.key] = (f"needs {need / 2 ** 30:.1f} GiB host RAM (x{RAM_MARGIN} margin), "
                                    f"{avail / 2 ** 30:.1f} GiB available")
                log.warning("skipping SAE %s: %s", sid.key, skipped[sid.key])
                self.write_skipped(skipped)
                continue
            with Par.reserve_ram(self.prof.root("cache_root"), 0 if self.tiny else int(need * RAM_MARGIN), sid.key):
                self._stage2_group(sid, jobs, est)

    def _stage2_group(self, sid, jobs, est):
        """Run every stage-2 job of one SAE (loaded once, freed after)."""
        sae = self.get_sae(sid)
        if getattr(sae, "composite", False):         # streams one weight file at a time: always on the run's device
            dev = self.device if not self.tiny else torch.device("cpu")
            sae.to_device(dev)
            sub = S.load_sublayer_params(self.cache, self.spec, self.attn_layers()) if self.attn_layers() else None
            t0 = time.time()
            for kind, t, L, _ in jobs:
                if kind != "task":
                    raise ValueError(f"{sid.key}: composite dictionaries have no generic-text stage 2")
                h = self.set_meta[t]["hash"]
                ents = {k: S.site_entry(self.cache, self.spec, h, k[0], k[1]) for k in sae.sites()}
                S.stage2_composite(self.cache, self.spans_entry(t, L), ents, sae, self.sae_key(sid), sub)
            self.timings[f"stage2_{sid.key}_s"] = round(time.time() - t0, 1)
            del sae
            empty_cache()
            return
        dev = self.device if est.placement in ("gpu", "blockwise-gpu") else torch.device("cpu")
        if est.placement == "gpu" and not self.tiny:
            sae.to_device(dev)
        t0 = time.time()
        for kind, t, L, _ in jobs:
            if kind == "task":
                S.stage2(self.cache, self.spans_entry(t, L), sae, self.sae_key(sid), dev, est.block_latents)
                if self.need_codes():
                    S.stage2_active(self.cache, self.spans_entry(t, L), sae, self.sae_key(sid), dev, est.block_latents)
                if self.cfg.subthresh:
                    S.stage2_codes(self.cache, self.spans_entry(t, L), sae, self.sae_key(sid), dev,
                                   self.cfg.subthresh.alpha_min, est.block_latents)
            else:
                r, _ = self.generic_entries(L, sid)
                S.stage2_generic(self.cache, r, sae, self.sae_key(sid), dev, est.block_latents)
        self.timings[f"stage2_{sid.key}_s"] = round(time.time() - t0, 1)
        del sae
        empty_cache()

    def stage2_done(self, job) -> bool:
        kind, t, L, sid = job
        e = self.sae_entry(t, L, sid) if kind == "task" else self.generic_entries(L, sid)[1]
        if kind == "task" and self.need_codes():
            ae = S.active_entry(self.cache, self.spans_entry(t, L), self.sae_key(sid))
            if not (ae.complete and self.cache.entry("sae_active_dec", codes=ae.key).complete):
                return False
        return e.complete and (kind != "task" or not self.cfg.subthresh or self.codes_entry(t, L, sid).complete)

    def skipped_saes(self) -> Dict[str, str]:
        p = self.run_dir / "skipped_saes.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def write_skipped(self, skipped: Dict[str, str]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.run_dir / "skipped_saes.json", json.dumps(skipped, indent=1))

    # ------------------------------------------------------------------ selection / fitting
    def flags(self, task):
        rows = self.rows[task]
        ent = S.entries_stage1(self.cache, self.spec, self.set_meta[task]["hash"], [])[0]
        cl = S.load_clean(ent, len(rows))
        DA = cl["A"]["logit_A"] - cl["A"]["logit_B"]
        DB = cl["B"]["logit_A"] - cl["B"]["logit_B"]
        mode = self.set_meta[task]["settings"]["filter_mode"]
        fl = T.filter_flags(rows, cl["A"]["top1_id"], cl["B"]["top1_id"], DA, DB, mode,
                            self.set_meta[task]["settings"].get("min_logit_diff") or 2.0)
        if self.tiny:                 # a random model fails every behavioural filter; tiny runs test plumbing only
            fl = {k: np.ones(len(rows), bool) for k in fl}
        return cl, fl

    def bundle(self, task, layer, sid, fit_families=None) -> Bundle:
        """Fitted objects for (task, layer, SAE) from the train pairs of `fit_families` (None = every row)."""
        groups = F.expand(task, fit_families) if fit_families else None
        key = (task, layer, sid.key, tuple(groups) if groups else None)
        if key in self.bundles:
            return self.bundles[key]
        rows = self.rows[task]
        _, fl = self.flags(task)
        tr = T.select(rows, fl, "train", self.cfg.fit_filter, self.n_train, groups=groups)
        need = max(MIN_TRAIN_PAIRS, (max(self.cfg.directions.ranks) + 2) // 2 + 1 if self.cfg.directions else 2)
        if len(tr) < (2 if (self.tiny or self.smoke) else need):
            raise CellSkip(f"{task} L{layer}: only {len(tr)} train pairs pass {self.cfg.fit_filter}"
                           + (f" in families {groups}" if groups else "") + f" (need {need})")
        sp = S.load_spans(self.spans_entry(task, layer), rows, tr, self.sae_entry(task, layer, sid))
        fs = Dr.FitSet(tuple(sp.pair_ids))
        b = Bundle(fs, tr, sp, C.MeanError.fit(fs, sp), C.CovNoise.fit(fs, torch.cat([sp.e_A, sp.e_B]), sp.pair_ids))
        d = self.cfg.directions
        if d:
            last = S.last_rows(sp)
            E_last = torch.cat([sp.e_A[last], sp.e_B[last]])
            labels = np.concatenate([sp.meta["state_A"], sp.meta["state_B"]])
            keys = sp.keys()
            for est in d.estimators:
                if est == "pca_pooled":
                    b.dirs[est] = Dr.pca_pooled(fs, sp.pair_ids, E_last, d.r_max)
                elif est == "answer_sep":
                    b.dirs[est] = Dr.answer_sep(fs, sp.pair_ids, E_last, labels, d.r_max)
                elif est == "pca_pos_centered":
                    b.dirs[est] = Dr.pca_pos_centered(fs, sp.pair_ids, torch.cat([sp.e_A, sp.e_B]), keys * 2, d.r_max)
                elif est == "pca_slot_centered_pairdiff":
                    b.dirs[est] = Dr.pca_slot_centered_pairdiff(fs, sp.pair_ids, sp.e_A - sp.e_B, keys, d.r_max)
                elif est in ("ridge_value", "fourier_T2", "fourier_T5", "fourier_T10"):
                    X_last = torch.cat([sp.x_A[last], sp.x_B[last]])
                    b.dirs[est] = Dr.value_direction(fs, sp.pair_ids, X_last, labels, est)
                elif est == "diffmean":
                    b.dirs[est] = Dr.diff_mean(fs, sp.pair_ids, sp.x_A[last], sp.x_B[last])
                elif est in ("grad_pooled", "grad_err"):
                    b.dirs[est] = None                          # filled below (needs the model; cached like DAS)
                elif est.startswith("das"):
                    b.dirs[est] = None                          # filled below, after the cheap estimators
            das = [e for e in d.estimators if e.startswith("das")]
            # pass 1: train every rank whose lock is free (other workers take the other ranks: DAS spreads over the GPUs);
            # pass 2: wait for the ranks another worker is training, then read them from the cache
            for blocking in (False, True):
                for est in das:
                    if b.dirs.get(est) is None:
                        b.dirs[est] = self.das_direction(task, layer, rows, tr, sp, fs, est, blocking=blocking)
            grads = [e for e in d.estimators if e in ("grad_pooled", "grad_err")]
            if grads:
                got = self.grad_directions(task, layer, sid, rows, tr, sp, fs, d.r_max)
                for e in grads:
                    b.dirs[e] = got[e]
            if any(c.startswith("steer_") for c in d.conditions):
                b.mean_diff = C.MeanDiff.fit(fs, sp)
            self.save_dirs(task, layer, sid, b, "+".join(fit_families) if fit_families else "")
        st_ = self.cfg.subthresh
        if st_ and st_.greedy_ks:
            codes = S.load_codes(self.codes_entry(task, layer, sid), self.spans_entry(task, layer), tr)
            dec = S.load_decoder(self.dec_entry(task, layer, sid))
            b.greedy = C.greedy_subthreshold(fs, sp.pair_ids, codes, dec, sp.e_A - sp.e_B, st_.g_alpha, max(st_.greedy_ks))
            out = self.run_dir / "subthresh"
            out.mkdir(parents=True, exist_ok=True)
            stem = f"{task}_L{layer}_{sid.key}" + (f"__fit-{'+'.join(fit_families)}" if fit_families else "")
            (out / f"{stem}.json").write_text(json.dumps(
                {"latents": b.greedy.latents, "objective": b.greedy.objective, "alpha": b.greedy.alpha,
                 "n_candidates": b.greedy.n_candidates, "fitset": {"digest": fs.digest, "n_pairs": len(tr),
                 "filter": self.cfg.fit_filter}, "target": "eps_A - eps_B on train span rows",
                 "objective_def": "sum of squared residuals of the target after each pick"}, indent=1))
        self.bundles[key] = b
        return b

    DAS_VERSION = 1                                     # bump when fit_das changes: old cached rotations are not reused

    def das_direction(self, task, layer, rows, tr, sp, fs, est: str, blocking: bool = True) -> Optional[Dr.Direction]:
        """Rank-k DAS rotation on the train pairs (E9), trained once and kept in the content-addressed cache, so every
        worker of a run (and every later run with the same train set) uses the identical rotation. blocking=False returns
        None when another worker is training it right now. --finalize-only never trains: a missing rotation is an error."""
        from harness.cache import fs_type, NETWORK_FS
        from harness.model.das import fit_das, fit_das_v2, init_seed_v2
        spec = das_spec(est)
        k, lr = spec["k"], spec["lr"]
        seed = int(self.cfg.seed if spec["seed"] is None else spec["seed"])
        epochs = 1 if self.tiny or self.smoke else spec["epochs"]
        if spec["version"] == 1:                    # v1 keys are unchanged: E9/E17 rotations stay reusable
            ent = self.cache.entry("das", spans=self.spans_entry(task, layer).key, fitset=fs.digest, k=int(k), epochs=epochs,
                                   lr=lr, seed=seed, model=self.spec.key(), version=self.DAS_VERSION)
        else:
            init = init_seed_v2(seed, task, layer, fs.digest)
            ent = self.cache.entry("das", spans=self.spans_entry(task, layer).key, fitset=fs.digest, k=int(k), epochs=epochs,
                                   lr=lr, seed=seed, model=self.spec.key(), version=2, objective=spec["objective"],
                                   holdout=spec["holdout"], step_tokens=spec["step_tokens"], init_seed=init)

        def read():
            U = ent.read_shard(0)["U"].numpy()
            return Dr.Direction(U, est, "same_task", fs.digest, len(tr), None, ent.read_json("info.json"))

        if ent.complete:
            return read()
        if self.finalizing:
            raise RuntimeError(f"finalize: DAS rotation das{k} for {task} L{layer} is not in the cache ({ent.dir}); "
                               "it is trained by the stage-3 workers, never during finalize")
        if fs_type(ent.dir).startswith(NETWORK_FS):
            raise RuntimeError(f"DAS cache {ent.dir} is on {fs_type(ent.dir)}: no reliable lock, so workers could train "
                               "different rotations; put cache_root on a local disk (profiles/local.yaml)")
        lock = ent.writer_lock() if blocking else ent.try_writer_lock()
        with lock as held:
            if held is False:
                return None
            ent.open()
            if not ent.complete:
                log.info("DAS: training %s (k=%d, %d epochs, lr %g, seed %d) for %s L%d on %d train pairs", est, k, epochs, lr, seed, task, layer, len(tr))
                model = self.load_model()
                empty_cache()
                if spec["version"] == 1:
                    U, info = fit_das(model, layer, rows, tr, sp, k, epochs=epochs, lr=lr, seed=seed, device=self.device)
                else:
                    U, info = fit_das_v2(model, layer, rows, tr, sp, k, objective=spec["objective"], epochs=epochs, lr=lr,
                                         seed=seed, init_seed=init, holdout=spec["holdout"],
                                         step_tokens=spec["step_tokens"], device=self.device, log=log.info)
                if not np.isfinite(U).all():
                    raise RuntimeError(f"DAS das{k} for {task} L{layer} diverged (non-finite rotation); not cached")
                ent.write_shard(0, {"U": torch.from_numpy(U)})
                ent.write_json("info.json", info)
                ent.mark_complete(info)
        return read()

    GRAD_VERSION = 1

    def grad_directions(self, task, layer, sid, rows, tr, sp, fs, r_max: int) -> Dict[str, Dr.Direction]:
        """P7: task-gradient directions on the train pairs, linearised at the feat_A_errB patch (cached; one writer)."""
        from harness.model.grads import fit_grad_dirs
        ent = self.cache.entry("grad", spans=self.spans_entry(task, layer).key, sae=self.sae_key(sid), fitset=fs.digest,
                               model=self.spec.key(), r_max=int(r_max), version=self.GRAD_VERSION)

        def read():
            sh = ent.read_shard(0)
            info = ent.read_json("info.json")
            return {e: Dr.Direction(sh[e].numpy(), e, "same_task", fs.digest, len(tr), None, info) for e in ("grad_pooled", "grad_err")}

        if ent.complete:
            return read()
        if self.finalizing:
            raise RuntimeError(f"finalize: gradient directions for {task} L{layer} are not in the cache ({ent.dir})")
        with ent.writer_lock():
            ent.open()
            if not ent.complete:
                log.info("grad directions: %s L%d %s on %d train pairs", task, layer, sid.key, len(tr))
                model = self.load_model()
                empty_cache()
                U, info = fit_grad_dirs(model, layer, rows, tr, sp, r_max, device=self.device)
                ent.write_shard(0, {k: torch.from_numpy(v) for k, v in U.items()})
                ent.write_json("info.json", info)
                ent.mark_complete(info)
        return read()

    def code_material(self, cell, sid, b: Bundle, test: C.Spans, te_idx) -> dict:
        """Active codes (train and test) and the decoder table for the code-level write-back (P0)."""
        if getattr(test, "_codes", None) is None:
            ae = S.active_entry(self.cache, self.spans_entry(cell.task, cell.layer), self.sae_key(sid))
            test._codes = S.load_active(ae, self.spans_entry(cell.task, cell.layer), te_idx)
            if getattr(b, "_codes", None) is None:
                b._codes = S.load_active(ae, self.spans_entry(cell.task, cell.layer), b.train_idx)
                b._dec = S.load_decoder(self.cache.entry("sae_active_dec", codes=ae.key))
                b._sets = {}
        return {"train": b._codes, "test": test._codes, "dec": b._dec, "sets": b._sets}

    def save_dirs(self, task, layer, sid, b: Bundle, fit_tag: str = ""):
        out = self.run_dir / "directions"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{task}_L{layer}_{sid.key}" + (f"__fit-{fit_tag}" if fit_tag else "")
        np.savez_compressed(out / f"{stem}.npz", **{k: v.U.astype(np.float32) for k, v in b.dirs.items()})
        info = {k: {"estimator": v.estimator, "fit_digest": v.fit_digest, "n_fit_rows": v.n_fit_rows,
                    "evr": None if v.evr is None else [float(x) for x in v.evr[:8]], "meta": v.meta} for k, v in b.dirs.items()}
        ests = list(b.dirs)
        own = lambda e: das_rank(e) if e.startswith("das") else None          # DAS columns are unordered: rank k only
        info["_cosines"] = {f"{a}|{c}|r{r}": [float(x) for x in Dr.principal_cosines(b.dirs[a].U[:, :r], b.dirs[c].U[:, :r])]
                            for i, a in enumerate(ests) for c in ests[i + 1:] for r in (1, 2, 4, 8)
                            if r <= min(b.dirs[a].U.shape[1], b.dirs[c].U.shape[1])
                            and own(a) in (None, r) and own(c) in (None, r)}
        info["_fitset"] = {"digest": b.fitset.digest, "n_pairs": len(b.train_idx), "filter": self.cfg.fit_filter}
        (out / f"{stem}.json").write_text(json.dumps(info, indent=1))

    def generic_dir(self, layer, sid) -> Optional[Dr.Direction]:
        _, ge = self.generic_entries(layer, sid)
        E = ge.require_complete().read_all(["eps"])["eps"]
        return Dr.generic(Dr.FitSet.generic(self.generic["file"]), E, self.cfg.directions.r_max)

    # ------------------------------------------------------------------ stage 3
    def cell_specs(self, cell, sid, test: C.Spans, bun: Bundle, te_idx: Sequence[int] = ()) -> List[dict]:
        """Condition specs (keys + how to build the tensor) for one cell, in a fixed order."""
        specs = [dict(_none_key(condition=n), kind="plain") for n in ("clean_B", "full_A", "full_B")]
        noise = {"feat_B_noise", "feat_A_noise", "feat_A_covnoise"}
        for n in self.cfg.conditions:
            if n in ("clean_B", "full_A", "full_B"):
                continue
            if n in noise:
                specs += [dict(_none_key(condition=n, seed=s), kind="noise") for s in self.cfg.noise_seeds]
            else:
                specs.append(dict(_none_key(condition=n), kind="plain"))
        d = self.cfg.directions
        if d:
            sources = []
            for s in d.sources:
                if s == "same_task":
                    sources += [(e, "same_task", bun.dirs[e]) for e in d.estimators]
                elif s == "other_task":
                    for t in self.cfg.prompt_sets:
                        if t != cell.task:
                            ob = self.bundle(t, cell.layer, sid)
                            sources += [(e, f"other_task:{t}", ob.dirs[e]) for e in d.estimators]
                elif s == "generic":
                    if "pca_pooled" in d.estimators:
                        sources.append(("pca_pooled", "generic", self.generic_dir(cell.layer, sid)))
                elif s.startswith("layer:"):
                    L2 = int(s.split(":")[1])
                    if L2 != cell.layer:
                        c2 = next(c for c in self.cfg.cells if c.task == cell.task and c.layer == L2)
                        ob = self.bundle(cell.task, L2, self.sae_id(L2, c2.sae), cell.fit_families)
                        sources += [(e, s, ob.dirs[e]) for e in d.estimators]
            for n in d.conditions:
                for e, src, D in sources:
                    for r in d.ranks:
                        if e.startswith("das") and r != das_rank(e):
                            continue                    # a DAS rotation is fitted for its own rank only (columns are not ordered)
                        if D is not None and r > D.U.shape[1]:
                            continue                    # value / Fourier estimators have 1-2 columns
                        if n.startswith("steer_") and r == 0:
                            continue                    # rank 0 adds nothing: it is clean_B
                        specs.append(dict(condition=n, estimator=e, source=src, r=r, seed=-1, kind="dir", D=D))
                for s in range(d.random_seeds):
                    D = Dr.random(test.x_A.shape[1], max(d.ranks), self.cfg.seed + 101 * s + cell.layer)
                    for r in d.ranks:
                        if r > 0:
                            specs.append(dict(condition=n, estimator="random", source="random", r=r, seed=s, kind="dir", D=D))
        st_ = self.cfg.subthresh
        if st_:
            codes = S.load_codes(self.codes_entry(cell.task, cell.layer, sid), self.spans_entry(cell.task, cell.layer),
                                 te_idx)
            dec = S.load_decoder(self.dec_entry(cell.task, cell.layer, sid))
            ks = [0] + ([k for k in st_.greedy_ks if bun.greedy and k <= len(bun.greedy.latents)])
            for k in ks:
                lat = bun.greedy.first(k) if k else None
                for a in st_.alphas:
                    for slot in st_.slot_errors:
                        specs.append(dict(subthresh_key(a, slot, k), kind="sub", alpha=a, slot=slot, latents=lat,
                                          codes=codes, dec=dec))
        return specs

    def materialize(self, spec, test: C.Spans, bun: Bundle, cell, donors) -> Optional[torch.Tensor]:
        n = spec["condition"]
        if spec["kind"] == "sub":
            mean_pos = bun.mean_err.per_row(test) if spec["slot"] == "meanPos" else None
            return C.build("subthresh", test, codes_A=spec["codes"], sae=spec["dec"], alpha=spec["alpha"],
                           slot_error="B" if spec["slot"] == "B" else "meanPos", eps_mean_pos=mean_pos,
                           latents=spec["latents"])
        if spec["kind"] == "dir" and n.startswith("steer_"):
            return C.steer(test, spec["D"].r(spec["r"]), bun.mean_diff.per_row(test), C.steer_scale(n))
        if spec["kind"] == "dir":
            U = spec["D"].r(spec["r"])
            if n in ("resampU_sameAns", "resampU_diffAns"):
                ds = donors["same" if n.endswith("sameAns") else "diff"]
                eC = C.donor_errors(test, ds)
                bad = torch.isnan(eC).any(1)
                eC[bad] = test.e_B[bad]                          # no donor: the U-part of B's own error (no change)
                return C.build(n, test, U=U, eps_C=eC)
            if n == "feat_codes_along_U":
                m = self.code_material(cell, self._cur_sid, bun, test, self._cur_te)
                key = (spec["estimator"], spec["source"], int(spec["r"]), int(spec["seed"]))
                if key not in m["sets"]:
                    cA, cB = m["train"]
                    lat, info = C.select_latents_along_U(cA, cB, m["dec"], U)
                    m["sets"][key] = lat
                    self.code_set_log.append({"cell": f"{cell.task}_L{cell.layer}_{self._cur_sid.key}", "estimator": key[0],
                                              "source": key[1], "r": key[2], "seed": key[3], **info})
                tA, tBc = m["test"]
                return C.build(n, test, eps_C=C.codes_writeback(tA, tBc, m["dec"], m["sets"][key]))
            mean_pos = bun.mean_err.per_row(test) if n == "repair_mean" else None
            return C.build(n, test, U=U, eps_mean_pos=mean_pos)
        if spec["kind"] == "noise":
            g = torch.Generator().manual_seed(self.cfg.seed + 1000 * spec["seed"] + cell.layer + (500 if n != "feat_B_noise" else 0))
            if n == "feat_B_noise":
                eta = C.gaussian_noise(test.e_A, g)            # pilot: norm matched to eps_A
            elif n == "feat_A_noise":
                eta = C.gaussian_noise(test.e_B, g)            # the error it replaces
            else:
                eta = C.rescale_rows(bun.cov.sample(test.m, g), test.e_B)
            return C.build(n, test, eta=eta)
        if n == "feat_A_meanTask":
            return C.build(n, test, eps_mean_task=bun.mean_err.task)
        if n == "feat_A_meanPos":
            return C.build(n, test, eps_mean_pos=bun.mean_err.per_row(test))
        if n.startswith("feat_A_resampC"):
            ds = donors["same" if n.endswith("sameAns") else "diff"]
            eC = C.donor_errors(test, ds)
            bad = torch.isnan(eC).any(1)
            eC[bad] = test.e_B[bad]                              # placeholder; rows flagged donor_status = none
            return C.build(n, test, eps_C=eC)
        return C.build(n, test)

    def run_stage3(self, compute: bool = True):
        """compute=False (--finalize-only): walk every cell on CPU to rebuild its checks, and require every unit done.
        With a worker id (multi-GPU), units are claimed (O_EXCL) so concurrent workers never compute the same one."""
        self.finalizing = not compute
        model = None
        parts = self.run_dir / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        skipped = self.skipped_saes()
        missing = []
        for cell in self.cfg.cells:
            sid = self.sae_id(cell.layer, cell.sae)
            if sid.key in skipped:
                log.warning("cell %s %s skipped: %s", cell.task, sid.key, skipped[sid.key])
                ck = {"cell": f"{cell.task}_{sid.key}{cell.tag()}", "role": cell.role, "skipped": skipped[sid.key]}
                self.checks.append(ck)
                atomic_write_text(parts / f"{ck['cell']}.checks.json", json.dumps(ck, indent=1, default=str))
                continue
            rows = self.rows[cell.task]
            cl, fl = self.flags(cell.task)
            groups = F.expand(cell.task, cell.eval_families) if cell.eval_families else None
            sel = {f: T.select(rows, fl, "test", f, self.n_test, groups=groups, per_group=True) for f in self.cfg.filters}
            te = sorted(set().union(*sel.values()), key=lambda i: (rows[i].get("family", "pilot"), rows[i]["rank"]))
            cell_key = f"{cell.task}_{sid.key}{cell.tag()}"
            try:
                if len(te) < 2:
                    raise CellSkip(f"{cell_key}: {len(te)} test pairs")
                bun = self.bundle(cell.task, cell.layer, sid, cell.fit_families)
            except CellSkip as e:
                log.warning("cell %s skipped: %s", cell_key, e)
                ck = {"cell": cell_key, "role": cell.role, "skipped": str(e)}
                self.checks.append(ck)
                atomic_write_text(parts / f"{cell_key}.checks.json", json.dumps(ck, indent=1, default=str))
                continue
            bun.fitset.assert_disjoint([rows[i]["pair_id"] for i in te])
            test = S.load_spans(self.spans_entry(cell.task, cell.layer), rows, te, self.sae_entry(cell.task, cell.layer, sid))
            self._cur_sid, self._cur_te = sid, te                     # code-level write-back material (materialize)
            donors = {"same": C.select_donors(test.meta, True, self.cfg.seed), "diff": C.select_donors(test.meta, False, self.cfg.seed)}
            try:
                specs = self.cell_specs(cell, sid, test, bun, te)        # other_task sources may hit a CellSkip too
            except CellSkip as e:
                log.warning("cell %s skipped: %s", cell_key, e)
                ck = {"cell": cell_key, "role": cell.role, "skipped": str(e)}
                self.checks.append(ck)
                atomic_write_text(parts / f"{cell_key}.checks.json", json.dumps(ck, indent=1, default=str))
                continue
            self.record_cell(cell, sid, rows, fl, sel, te, bun, test, parts / f"{cell_key}.checks.json")
            base = self.base_rows(cell, sid, rows, fl, te, cl, test)
            pq = parts / f"{cell_key}__clean.parquet"
            if not (parts / f"{cell_key}__clean.done").exists():
                tmp = pq.with_name(f"{pq.name}.tmp.{os.getpid()}")
                pd.DataFrame(base).to_parquet(tmp)
                os.replace(tmp, pq)                          # identical content whichever worker writes it
                atomic_write_text(parts / f"{cell_key}__clean.done", "")
            for u in range(0, len(specs), UNIT_SIZE):
                stem = f"{cell_key}__{u // UNIT_SIZE:04d}"
                done = parts / f"{stem}.done"
                if done.exists():
                    continue
                if not compute:
                    missing.append(stem)
                    continue
                cl_path = parts / f"{stem}.claim"
                if self.worker and not Par.claim(cl_path, self.worker):
                    continue                                 # another live worker owns this unit
                if done.exists():                            # finished between our check and the claim
                    Par.release(cl_path)
                    continue
                if model is None:
                    model = self.load_model()
                chunk = specs[u:u + UNIT_SIZE]
                tens = {i: self.materialize(sp, test, bun, cell, donors) for i, sp in enumerate(chunk)}
                t0 = time.time()
                out = S.stage3(model, rows, te, cell.layer, test.row_ptr, {str(i): v for i, v in tens.items()},
                               self.budget, self.device)
                recs = []
                for i, sp in enumerate(chunk):
                    o = out[str(i)]
                    for k, pi in enumerate(te):
                        ds = "n/a"
                        if sp["condition"].startswith(("feat_A_resampC", "resampU_")):
                            ds = donors["same" if sp["condition"].endswith("sameAns") else "diff"][k]["status"]
                        recs.append(self.pair_record(cell, sid, rows[pi], fl, pi, sp, o, k, ds))
                tmp = parts / f"{stem}.parquet.tmp.{os.getpid()}"
                pd.DataFrame(recs).to_parquet(tmp)
                os.replace(tmp, parts / f"{stem}.parquet")
                atomic_write_text(done, json.dumps({"seconds": round(time.time() - t0, 1), "n_conditions": len(chunk),
                                                    "worker": self.worker, "max_tokens": self.budget.max_tokens,
                                                    "backoffs": self.budget.backoffs}))
                if self.worker:
                    Par.release(cl_path)
                log.info("%s unit %d/%d (%d conditions, %.1fs)", cell_key, u // UNIT_SIZE + 1,
                         (len(specs) + UNIT_SIZE - 1) // UNIT_SIZE, len(chunk), time.time() - t0)
            if self.code_set_log:                                    # latent sets of the code-level write-back (P0)
                p = parts / f"{cell_key}.codesets.json"
                old = json.loads(p.read_text()) if p.exists() else []
                seen = {json.dumps(x, sort_keys=True) for x in old}
                new = old + [x for x in self.code_set_log if json.dumps(x, sort_keys=True) not in seen]
                atomic_write_text(p, json.dumps(new, indent=1))
                self.code_set_log = []
        if missing:
            raise RuntimeError(f"finalize: {len(missing)} stage-3 units are not done (e.g. {missing[:3]}); "
                               "run the workers to completion first")

    def pair_record(self, cell, sid, r, fl, pi, sp, o, k, donor_status) -> dict:
        rec = {"pair_id": r["pair_id"], "split": r["split"], "family": r.get("family", "pilot"), "task": cell.task,
               "layer": cell.layer, "sae": sid.key, "condition": sp["condition"], "estimator": sp["estimator"],
               "source": sp["source"], "r": int(sp["r"]), "seed": int(sp["seed"]), "ans_A_id": r["ans_A"],
               "ans_B_id": r["ans_B"], "donor_status": donor_status, "fit_families": cell.fit_tag()}
        rec.update({f: bool(fl[f][pi]) for f in T.FILTERS})
        for f in S.OUT_FIELDS:
            v = o[f][k]
            rec[f] = v.tolist() if f == "cand_logits" else (int(v) if f == "top1_id" else float(v))
        return rec

    def base_rows(self, cell, sid, rows, fl, te, cl, test) -> List[dict]:
        sp = dict(_none_key(condition="clean_A"))
        out = []
        for k, pi in enumerate(te):
            o = {f: cl["A"][f][pi:pi + 1] for f in S.OUT_FIELDS}
            out.append(self.pair_record(cell, sid, rows[pi], fl, pi, sp, o, 0, "n/a"))
        return out

    def record_cell(self, cell, sid, rows, fl, sel, te, bun, test, checks_path: Optional[Path] = None):
        keep = {}
        for split in ("train", "test"):
            idx = [i for i, r in enumerate(rows) if r["split"] == split]
            keep[split] = {f: round(float(fl[f][idx].mean()), 4) for f in T.FILTERS}
            fams = sorted({F.group_of(rows[i].get("family", "pilot")) for i in idx})
            if len(fams) > 1:
                for g in fams:
                    gi = [i for i in idx if F.group_of(rows[i].get("family", "pilot")) == g]
                    keep[f"{split}:{g}"] = {f: round(float(fl[f][gi].mean()), 4) for f in T.FILTERS}
        self.checks.append({"cell": f"{cell.task}_{sid.key}{cell.tag()}", "role": cell.role, "sae": sid.to_dict(),
                            "fit_families": cell.fit_families, "eval_families": cell.eval_families,
                            "n_test_by_filter": {f: len(v) for f, v in sel.items()}, "n_test_union": len(te),
                            "n_train": len(bun.train_idx), "fitset_digest": bun.fitset.digest, "keep_rates": keep,
                            "mean_l0_test": [float(test.l0_A.float().mean()), float(test.l0_B.float().mean())],
                            "rel_err_norm_test": float((test.e_A.norm(dim=-1) / test.x_A.norm(dim=-1)).mean()),
                            "mean_pos_coverage": (bun.mean_err.per_row(test), bun.mean_err.last_coverage)[1],
                            **({"generic": self.generic_stats(cell.layer, sid)} if self.cfg.generic_l0 else {})})
        if checks_path is not None:                      # finalize merges these (workers each see only some cells)
            atomic_write_text(checks_path, json.dumps(self.checks[-1], indent=1, default=str))

    def generic_stats(self, layer, sid) -> dict:
        """Realized L0 and error energy sum||eps||^2 / sum||x||^2 of the cell's SAE on the generic set (E4)."""
        _, ge = self.generic_entries(layer, sid)
        t = ge.require_complete().read_all(["eps", "l0", "x_norm"])
        return {"rows": int(len(t["l0"])), "mean_l0": float(t["l0"].float().mean()),
                "err_energy_ratio": float((t["eps"].float() ** 2).sum() / (t["x_norm"].float() ** 2).sum()),
                "corpus": self.generic["file"]}

    # ------------------------------------------------------------------ finalize
    def finalize(self) -> None:
        from harness.analysis import add_gaps, summarize

        pdir = self.run_dir / "parts"
        parts = sorted(p for p in pdir.glob("*.parquet") if (pdir / f"{p.stem}.done").exists())
        if not parts:
            return
        order = [f"{c.task}_{self.sae_id(c.layer, c.sae).key}{c.tag()}" for c in self.cfg.cells]
        disk = {p.name[:-len(".checks.json")]: json.loads(p.read_text()) for p in pdir.glob("*.checks.json")}
        if disk:
            self.checks = [disk[k] for k in order if k in disk]
        pp = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        pp.to_parquet(self.run_dir / "per_pair.parquet")
        summ = add_gaps(summarize(pp, self.cfg.filters, n_boot=self.cfg.bootstrap, seed=self.cfg.seed))
        summ.to_csv(self.run_dir / "summary.csv", index=False)
        hooks = []
        from harness.analysis import with_defaults
        for cell, g in with_defaults(pp).groupby(["task", "layer", "sae", "fit_families"]):
            a = g[g.condition == "clean_A"].set_index("pair_id")
            for name in ("full_B", "full_A"):
                p = g[g.condition == name].set_index("pair_id")
                ref = g[g.condition == ("clean_B" if name == "full_B" else "clean_A")].set_index("pair_id")
                common = p.index.intersection(ref.index)
                dD = ((p.loc[common, "logit_A"] - p.loc[common, "logit_B"]) - (ref.loc[common, "logit_A"] - ref.loc[common, "logit_B"])).abs()
                hooks.append({"cell": "_".join(map(str, cell)), "check": f"{name} vs {'clean_B' if name == 'full_B' else 'clean_A'}",
                              "max_abs_dD": float(dD.max()) if len(dD) else None})
        (self.run_dir / "summary.json").write_text(json.dumps({"exp_id": self.cfg.exp_id, "run_hash": self.run_hash,
                                                               "n_rows": len(pp), "hook_checks": hooks,
                                                               "cells": self.checks}, indent=1, default=str))

    def manifest(self, status: str, t_start: float, stages) -> dict:
        roots = [str(self.prof.root(k)) for k in ("cache_root", "data_root", "runs_root")] + [str(p) for p in self.prof.spill_roots()] + [str(REPO_ROOT)]
        m = {"exp_id": self.cfg.exp_id, "run_hash": self.run_hash, "status": status, "stages": stages,
             "smoke": self.smoke, "tiny": self.tiny, "created": provenance.now(),
             "wall_time_s": round(time.time() - t_start, 1), "config": self.cfg.to_dict(),
             "config_file": str(Path(self.args.config)), "profile": {**self.prof.to_dict()},
             "git": self.git, "packages": provenance.packages(), "machine": provenance.machine(self.device),
             "model": self.spec.key(), "sae_revision": self.sae_revision,
             "prompt_sets": {t: {k: m[k] for k in ("hash", "file", "n") if k in m} for t, m in self.set_meta.items()},
             "generic": None if not self.generic else {k: self.generic[k] for k in ("hash", "file", "skip")},
             "seeds": {"seed": self.cfg.seed, "noise_seeds": self.cfg.noise_seeds,
                       "random_seeds": self.cfg.directions.random_seeds if self.cfg.directions else 0},
             "timings": self.timings, "budget": {"max_tokens": self.budget.max_tokens, "backoffs": self.budget.backoffs},
             "peak_gpu_gb": round(peak_bytes(self.device) / 2 ** 30, 2), "cells": self.checks,
             "allow_dirty": bool(self.args.allow_dirty), "worker": self.worker,
             "resources": {k: v for k, v in self.resources.items() if k != "probe"},
             "machine_probe": self.resources.get("probe"),
             "filters_overridden": "tiny: all filter flags set to True" if self.tiny else None}
        m["config_file"] = str(Path(self.args.config).resolve().relative_to(REPO_ROOT)) if Path(self.args.config).resolve().is_relative_to(REPO_ROOT) else Path(self.args.config).name
        for k in ("cache_root", "data_root", "runs_root", "hf_home"):
            if m["profile"].get(k) and Path(str(m["profile"][k])).is_absolute():
                m["profile"][k] = f"<{k}>"                    # machine-specific; keep manifests portable
        if m["profile"].get("cache_spill"):
            m["profile"]["cache_spill"] = ["<cache_spill>"] * len(m["profile"]["cache_spill"])
        provenance.assert_relative({k: v for k, v in m.items() if k not in ("machine", "git")}, roots)
        return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="accepted for clarity; runs always resume finished units")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="tiny random model + random SAEs (tests only)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--worker", default=None, help="stage-3 worker id: claim units so several workers share a run")
    ap.add_argument("--finalize-only", action="store_true",
                    help="no model: require every stage-3 unit done, merge parts and checks, write the run's manifest")
    ap.add_argument("--stage2-slice", default=None, help="i/n: only the i-th of n disjoint SAE groups (stage 2)")
    ap.add_argument("--max-batch-tokens", type=int, default=None, help="pin the token budget (overrides auto)")
    ap.add_argument("--only-sae", action="append", default=None,
                    help="smoke only: keep the cells matching this selector (repeatable; a dictionary-name prefix or layer= / task= / width= / "
                         "dict= / addon= key=value, see select_cells); the run hash "
                         "follows the reduced cell list, so several smoke processes of one config never share a run")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    cfg = ExpConfig.load(a.config)
    if a.only_sae:
        if not (a.smoke or a.tiny):
            raise SystemExit("--only-sae is for smoke runs (a real run keeps its config's cells)")
        import dataclasses
        keep = select_cells(cfg.cells, a.only_sae)
        if not keep:
            raise SystemExit(f"--only-sae {a.only_sae}: no cell matches")
        cfg = dataclasses.replace(cfg, cells=keep)
    prof = load_profile(a.profile)
    prof.check_experiment(cfg.exp_id)
    prof.apply_env()
    # measure the machine, then replace every `auto` profile value by a measured one (plan step 2-4)
    probe = probe_machine(prof.root("cache_root"))
    dev_mem = None
    if not (a.tiny or a.dry_run or a.finalize_only):
        dev_mem = device_memory(resolve_device(prof.device))
    prof, resources = memory.resolve_resources(prof, probe, dev_mem)
    if a.max_batch_tokens:
        import dataclasses
        prof = dataclasses.replace(prof, max_batch_tokens=a.max_batch_tokens)
        resources.setdefault("rules", {})["max_batch_tokens"] = "pinned by --max-batch-tokens"
    resources["cpu_threads"] = set_threads(prof.cpu_threads if isinstance(prof.cpu_threads, int) else None)
    if not a.tiny and not cfg.model_revision:
        raise SystemExit(f"{a.config}: model_revision must be pinned")
    R = Runner(cfg, prof, a)
    R.resources = resources
    if a.dry_run:
        p = R.plan()
        print(json.dumps(p, indent=1))
        return 0 if p["all_fit"] else 2
    if not a.tiny and not prof.allow_real_model and not a.smoke:
        raise SystemExit(f"profile {prof.name} does not allow full real-model runs")
    if R.git["dirty"] and not (a.allow_dirty or a.smoke or a.tiny):
        raise SystemExit("working tree is dirty: commit, or pass --allow-dirty (recorded in the manifest)\n  "
                         + "\n  ".join(R.git.get("dirty_files", [])[:10]))
    R.run_dir.mkdir(parents=True, exist_ok=True)
    # one log / manifest per partial process; only a full run or --finalize-only writes the run's manifest.json
    tag = (f".w{R.worker}" if R.worker else f".s2-{a.stage2_slice.replace('/', 'of')}" if a.stage2_slice
           else f".stage{a.stage}" if a.stage != "all" else "")
    fh = logging.FileHandler(R.run_dir / f"log{tag}.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    cap_memory(R.device, prof.gpu_mem_cap_gb)
    reset_peak(R.device)
    t0 = time.time()
    stages = ["1", "2", "3"] if a.stage == "all" else [a.stage]
    status = "running"
    if a.finalize_only:
        try:
            R.run_stage3(compute=False)
            R.finalize()
            status = "complete"
            return 0
        except Exception:
            status = "failed"
            log.exception("finalize failed")
            return 1
        finally:
            atomic_write_text(R.run_dir / "manifest.json", json.dumps(R.manifest(status, t0, ["finalize"]), indent=1, default=str))
            log.info("%s finalize -> %s (%s)", cfg.exp_id, R.run_dir, status)
    try:
        for s in stages:
            ts = time.time()
            {"1": R.run_stage1, "2": R.run_stage2, "3": R.run_stage3}[s]()
            R.timings[f"stage{s}_total_s"] = round(time.time() - ts, 1)
            if s == "1":
                R.free_model()
        if "3" in stages and not R.worker:
            R.finalize()
        status = ("complete" if "3" in stages and not R.worker else
                  "worker_done" if R.worker else f"stage{stages[-1]}_complete")
        return 0
    except Exception:
        status = "failed"
        log.exception("run failed")
        return 1
    finally:
        atomic_write_text(R.run_dir / f"manifest{tag}.json", json.dumps(R.manifest(status, t0, stages), indent=1, default=str))
        log.info("%s -> %s (%s)", cfg.exp_id, R.run_dir, status)


if __name__ == "__main__":
    sys.exit(main())

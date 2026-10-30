#!/usr/bin/env python
"""HF top-1 agreement on the pilot task prompts (RUN machine). Investigates the "95% top-1 agreement with Hugging Face"
figure, which results/check_forward measured on 2026-10-20 on the legacy (near-chance) tasks, not the pilot tasks.

    python scripts/hf_agreement.py --pairs-dir golden/run1 --out hf_agreement/run1      # after export_golden.py
    python scripts/hf_agreement.py --pairs-dir /tmp/golden --out /tmp/hfa --mock         # CPU plumbing test

For every scored candidate pair in <pairs-dir>/pairs_<task>.jsonl.gz (capped by --max-per-task: kept test pairs first, then
rejected candidates, then train pairs)
it computes last-position logits for prompt A and prompt B with
    ours : the pilot path (TransformerLens no_processing, bf16 on CUDA, CPU-offloaded embedding, last-position head)
    hf   : transformers Gemma2ForCausalLM, same dtype, eager attention, logits_to_keep=1
    hf32 : (optional, --fp32-subset N) HF in float32 on CPU for N prompts, as a tie-breaking reference
and reports: top-1 agreement, top-1 margins of disagreements (near-tie?), |D_ours - D_hf| with
D = logit(ans_A) - logit(ans_B), and filter decisions (the pilot's strict / clean_only rule) that flip.
Models are loaded one at a time. Output: <out>/{report.md, per_prompt.csv, manifest.json}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.data.synthetic_tasks import batch_by_length, get_tokenizer, valid_mask
from src.models import runtime
from src.utils import free_memory, get_device, get_dtype, load_config, save_json, setup_logging

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["entity_track", "add_chain", "var_binding", "ioi_control"]
NEAR_TIE = 0.25          # logits: a disagreement whose margin is below this is reported as a near-tie
log = None


class Item:
    """Minimal stand-in for TokenizedPair so pilot batching (batch_by_length) can be reused."""

    def __init__(self, row: dict, task: str) -> None:
        self.row, self.task = row, task
        self.clean_ids, self.corrupt_ids = row["clean_ids"], row["corrupt_ids"]


def load_items(pairs_dir: Path, task: str, cap: int):
    f = pairs_dir / f"pairs_{task}.jsonl.gz"
    if not f.exists():
        return []
    with gzip.open(f, "rt") as fh:
        rows = [json.loads(l) for l in fh]
    rows = [r for r in rows if r.get("clean_ids")]
    rows.sort(key=lambda r: (r["split"] != "test", r["split"] is not None, r["pair_id"]))   # test, rejected, train
    return [Item(r, task) for r in rows[:cap]]


def summarise_logits(lg: torch.Tensor, ca: int, ka: int) -> dict:
    v, i = lg.float().topk(2)
    return {"top1": int(i[0]), "top2": int(i[1]), "margin": float(v[0] - v[1]), "D": float(lg[ca] - lg[ka]),
            "logit_ca": float(lg[ca]), "logit_ka": float(lg[ka])}


@torch.inference_mode()
def run_ours(items, cfg, mock: bool, tok) -> dict:
    from src.models.sae_loader import load_model

    device = get_device("cpu" if mock else None)
    model = load_model(mock=mock, device=device, dtype=get_dtype(device), mock_cfg=cfg.get("mock"), tokenizer=tok,
                       model_name=cfg["model_name"], runtime_cfg=cfg.get("runtime"))
    runtime.setup_runtime(cfg, model, device)
    out = {}
    for kind in ("clean", "corrupt"):
        for b in batch_by_length(items, cfg["batch_size"]):
            t = torch.tensor([getattr(it, f"{kind}_ids") for it in b], device=device)
            lg = runtime.last_logits(model, t).cpu()
            for it, l in zip(b, lg):
                out[(it.row["pair_id"], kind)] = (l, summarise_logits(l, it.row["clean_ans"], it.row["corrupt_ans"]))
    del model
    free_memory()
    return out


@torch.inference_mode()
def run_hf(items, name: str, device, dtype, keys=None) -> dict:
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
    except TypeError:                                    # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="eager")
    model = model.to(device).eval()
    log.info("HF %s loaded on %s (%s) in %.0fs", name, device, dtype, time.time() - t0)
    out = {}
    for kind in ("clean", "corrupt"):
        for it in items:
            if keys is not None and (it.row["pair_id"], kind) not in keys:
                continue
            t = torch.tensor([getattr(it, f"{kind}_ids")], device=device)
            try:
                lg = model(t, logits_to_keep=1).logits[0, -1].float().cpu()
            except TypeError:
                lg = model(t).logits[0, -1].float().cpu()
            out[(it.row["pair_id"], kind)] = (lg, summarise_logits(lg, it.row["clean_ans"], it.row["corrupt_ans"]))
    del model
    free_memory()
    return out


def fake_hf(ours: dict, seed: int = 0) -> dict:
    """--mock: 'HF' = ours + small noise, to exercise the comparison and report."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, (lg, _) in ours.items():
        l2 = lg + 0.3 * torch.randn(lg.shape, generator=g)
        it_ca, it_ka = _ANS[k[0]]
        out[k] = (l2, summarise_logits(l2, it_ca, it_ka))
    return out


_ANS = {}


def filter_flags(rows_ours: dict, rows_hf: dict, items, cfg, task: str) -> dict:
    """Pilot filter decision per pair under ours vs hf scores (task's filter_mode, min_logit_diff)."""
    c = {**cfg, **(cfg.get("task_overrides") or {}).get(task, {})}
    mode = c.get("filter_mode", "strict")
    if mode == "none":
        return {}
    pids = [it.row["pair_id"] for it in items if (it.row["pair_id"], "clean") in rows_hf and (it.row["pair_id"], "corrupt") in rows_hf]
    res = {}
    for name, R in (("ours", rows_ours), ("hf", rows_hf)):
        sc = {"dl_clean": np.array([R[(p, "clean")][1]["D"] for p in pids]),
              "dl_corrupt": np.array([R[(p, "corrupt")][1]["D"] for p in pids]),
              "clean_top1": np.array([R[(p, "clean")][1]["top1"] == _ANS[p][0] for p in pids]),
              "corrupt_top1": np.array([R[(p, "corrupt")][1]["top1"] == _ANS[p][1] for p in pids])}
        res[name] = valid_mask(sc, mode, c.get("min_logit_diff", 2.0))
    return {p: (bool(a), bool(b)) for p, a, b in zip(pids, res["ours"], res["hf"])} | {"__mode__": mode}


def main() -> int:
    global log
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", required=True, help="export_golden.py output dir (pairs_<task>.jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/paper_residue.yaml")
    ap.add_argument("--pilot-run", default="results/paper_20261026_001341")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max-per-task", type=int, default=600)
    ap.add_argument("--fp32-subset", type=int, default=24, help="prompts per run for the float32 CPU HF reference (0 = off)")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "hf_agreement.log")
    cfg = load_config(str(ROOT / a.config))
    pre = ROOT / a.pilot_run / "preflight" / "preflight.json"
    if pre.exists():
        from src.experiments.common import merge_task_overrides
        cfg["task_overrides"] = merge_task_overrides(cfg.get("task_overrides"), json.loads(pre.read_text()).get("task_overrides"))
    tok = get_tokenizer(a.mock, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
    if not a.mock and type(tok).__name__ == "FallbackTokenizer":   # pilot get_tokenizer falls back silently
        raise SystemExit("HF_TOKEN is not set: the pilot code would use the offline FallbackTokenizer with the "
                         "real model. export HF_TOKEN=hf_... and rerun.")
    items = []
    for t in a.tasks.split(","):
        items += load_items(Path(a.pairs_dir), t, a.max_per_task)
    for it in items:
        _ANS[it.row["pair_id"]] = (it.row["clean_ans"], it.row["corrupt_ans"])
    log.info("%d pairs (%d prompts)", len(items), 2 * len(items))
    t0 = time.time()
    ours = run_ours(items, cfg, a.mock, tok)
    device = get_device("cpu" if a.mock else None)
    hf = fake_hf(ours) if a.mock else run_hf(items, cfg.get("hf_tokenizer", "google/gemma-2-2b"), device, get_dtype(device))
    hf32 = {}
    if a.fp32_subset and not a.mock:
        avail = runtime.available_ram() / 2 ** 30
        if avail < 14:
            log.warning("fp32 reference skipped: %.1f GiB RAM available (< 14)", avail)
        else:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(items), size=min(a.fp32_subset, len(items)), replace=False)
            keys = {(items[i].row["pair_id"], k) for i in pick for k in ("clean", "corrupt")}
            hf32 = run_hf([items[i] for i in pick], cfg.get("hf_tokenizer", "google/gemma-2-2b"), torch.device("cpu"),
                          torch.float32, keys)
    rows = []
    flips = {}
    for t in a.tasks.split(","):
        flips[t] = filter_flags(ours, hf, [it for it in items if it.task == t], cfg, t)
    for it in items:
        for kind in ("clean", "corrupt"):
            k = (it.row["pair_id"], kind)
            o, h = ours[k][1], hf[k][1]
            r = {"task": it.task, "pair_id": it.row["pair_id"], "prompt": "A" if kind == "clean" else "B",
                 "split": it.row["split"], "T": len(it.clean_ids), "top1_agree": o["top1"] == h["top1"],
                 "ours_top1": o["top1"], "hf_top1": h["top1"], "ours_margin": o["margin"], "hf_margin": h["margin"],
                 "near_tie": (o["top1"] != h["top1"]) and min(o["margin"], h["margin"]) < NEAR_TIE,
                 "swap_top2": o["top1"] == h["top2"] and o["top2"] == h["top1"],
                 "D_ours": o["D"], "D_hf": h["D"], "abs_dD": abs(o["D"] - h["D"]),
                 "max_abs_logit_diff": float((ours[k][0] - hf[k][0]).abs().max())}
            f = flips[it.task].get(it.row["pair_id"])
            if f:
                r.update(filter_ours=f[0], filter_hf=f[1])
            if k in hf32:
                g = hf32[k][1]
                r.update(fp32_top1=g["top1"], D_fp32=g["D"], ours_top1_eq_fp32=o["top1"] == g["top1"],
                         hf_top1_eq_fp32=h["top1"] == g["top1"], abs_dD_ours_fp32=abs(o["D"] - g["D"]),
                         abs_dD_hf_fp32=abs(h["D"] - g["D"]))
            rows.append(r)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_prompt.csv", index=False)
    g = df.groupby(["task", "prompt"])
    summ = g.agg(n=("top1_agree", "size"), top1_agree=("top1_agree", "mean"), disagree=("top1_agree", lambda s: int((~s).sum())),
                 near_tie_disagree=("near_tie", "sum"), median_abs_dD=("abs_dD", "median"), max_abs_dD=("abs_dD", "max"),
                 median_max_logit_diff=("max_abs_logit_diff", "median")).reset_index()
    L = [f"# HF agreement on pilot task prompts {'(MOCK: plumbing only)' if a.mock else ''}", "",
         f"{len(items)} pairs, {len(df)} prompts, overall top-1 agreement {df.top1_agree.mean():.4f}; "
         f"near-tie threshold {NEAR_TIE} logits; {time.time() - t0:.0f} s", "",
         "| task | prompt | n | top-1 agree | disagreements | of which near-ties | median abs dD | max abs dD | median max abs logit diff |",
         "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.task} | {r.prompt} | {r.n} | {r.top1_agree:.4f} | {r.disagree} | {int(r.near_tie_disagree)} | "
          f"{r.median_abs_dD:.3f} | {r.max_abs_dD:.3f} | {r.median_max_logit_diff:.3f} |" for r in summ.itertuples()]
    if "filter_ours" in df:
        L += ["", "## Pilot filter decisions, ours vs HF scores", "", "| task | mode | pairs | kept (ours) | kept (hf) | flips |",
              "|---|---|---|---|---|---|"]
        for t, fl in flips.items():
            v = [x for k, x in fl.items() if k != "__mode__"]
            if v:
                L.append(f"| {t} | {fl['__mode__']} | {len(v)} | {sum(x[0] for x in v)} | {sum(x[1] for x in v)} | "
                         f"{sum(x[0] != x[1] for x in v)} |")
    if hf32:
        s = df.dropna(subset=["fp32_top1"])
        L += ["", f"## float32 CPU HF reference ({len(s)} prompts)", "",
              f"- ours top-1 == fp32: {s.ours_top1_eq_fp32.mean():.4f}; hf(bf16) top-1 == fp32: {s.hf_top1_eq_fp32.mean():.4f}",
              f"- median |D - D_fp32|: ours {s.abs_dD_ours_fp32.median():.3f}, hf {s.abs_dD_hf_fp32.median():.3f}"]
    dis = df[~df.top1_agree].sort_values("abs_dD", ascending=False).head(20)
    if len(dis):
        L += ["", "## Largest disagreements", "", "| task | pair | prompt | ours top-1 (margin) | hf top-1 (margin) | D ours | D hf |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {r.task} | {r.pair_id} | {r.prompt} | {tok.decode([r.ours_top1])!r} ({r.ours_margin:.2f}) | "
              f"{tok.decode([r.hf_top1])!r} ({r.hf_margin:.2f}) | {r.D_ours:.2f} | {r.D_hf:.2f} |" for r in dis.itertuples()]
    (out / "report.md").write_text("\n".join(L) + "\n")
    from importlib import metadata as md

    vers = {}
    for p in ("torch", "transformers", "transformer-lens", "numpy"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    save_json({"args": vars(a), "versions": vers, "n_pairs": len(items), "near_tie": NEAR_TIE,
               "summary": summ.to_dict("records"), "seconds": round(time.time() - t0, 1)}, out / "manifest.json")
    print("\n".join(L[:8 + len(summ)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

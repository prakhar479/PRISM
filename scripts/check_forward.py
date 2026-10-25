#!/usr/bin/env python
"""Ground-truth check of the real-model forward pass and the task data (run on the GPU box, ~5-10 min).

The overnight run of 2026-10-19 kept 0 valid pairs: IOI top-1 was 0%, the algorithmic tasks were at chance
and mean logit gaps exceeded the +-60 that Gemma-2's softcap allows. This script finds out which layer is at
fault by comparing, prompt by prompt, the next-token logits of

    hf          Hugging Face transformers' own Gemma2ForCausalLM (the reference)
    tl_plain    TransformerLens' normal forward, model(tokens)   (embedding on GPU)
    ours        src.models.runtime.last_logits as the experiments use it (embedding on CPU when offloaded)
    ours_gpu    the same with the embedding on GPU (isolates the CPU-embedding path)

plus the resume-from-layer path, the chunked NLL, batched vs one-at-a-time scoring, the tokenizer (BOS),
task accuracy per prompt setting, and whether each task's prompt space is big enough for the configured
splits and few-shot prefix. Everything is written to <out>/report.md and <out>/report.json.

    python scripts/check_forward.py                              # full check
    python scripts/check_forward.py --skip-hf                    # no HF reference (saves ~1 min / 5 GB)
    python scripts/check_forward.py --config configs/tier1_core.yaml --overrides results/<run>/preflight/preflight.json
"""
import argparse
import contextlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.data import synthetic_tasks as st
from src.experiments.common import merge_task_overrides, task_cfg
from src.models import runtime
from src.models.sae_loader import load_model
from src.utils import free_memory, get_device, get_dtype, load_config, load_json, model_device, save_json, setup_logging

ROOT = Path(__file__).resolve().parents[1]
SOFTCAP = 30.0
# natural-language prompts gemma-2-2b gets right; if it fails these, nothing downstream can work
SANITY = [
    ("The capital of France is", " Paris"),
    ("When Mary and John went to the store, John gave a drink to", " Mary"),
    ("When Tom and Anna went to the park, Anna gave a book to", " Tom"),
    ("Monday, Tuesday, Wednesday,", " Thursday"),
    ("The opposite of hot is", " cold"),
    ("1, 2, 3, 4,", " "),          # Gemma splits digits: ' 5' is '▁' + '5'
    ("1, 2, 3, 4, ", "5"),
]
TASK_FORMAT = {"add_chain": "space_prompt", "entity_track": "space_target", "var_binding": "space_prompt",
               "ioi_control": "space_target", "mod_sum": "space_prompt", "dyck_depth": "space_prompt",
               "graph_reach": "space_prompt"}
SUITE = st.STATE_TASKS + ["ioi_control"]      # the configured task suite (legacy tasks: see analyze_tasks.py)
GEN_KW = {}                                   # task -> generator difficulty, filled from the config in main()

log = logging.getLogger("prism")
CHECKS = []


def check(name: str, ok, detail: str = "", warn_only: bool = False):
    """ok: True / False / None (skipped)."""
    status = "SKIP" if ok is None else "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"check": name, "status": status, "detail": detail})
    log.log(logging.INFO if status in ("PASS", "SKIP") else logging.WARNING, "[%s] %s %s", status, name, detail)
    return ok


def sh(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def version(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def show(tok, i):
    return repr(tok.decode([int(i)]))


def topk(tok, logits, k=5):
    p = torch.softmax(logits.float(), -1)
    v, i = p.topk(k)
    return [(show(tok, j), round(float(x), 3)) for x, j in zip(v, i)]


# --------------------------------------------------------------------------- 1. environment
def env_section(out: Path) -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": version("transformers"), "transformer_lens": version("transformer_lens"),
           "sae_lens": version("sae_lens"), "numpy": np.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "vram_gb": round(runtime.total_vram() / 2 ** 30, 2), "ram_available_gb": round(runtime.available_ram() / 2 ** 30, 1),
           "torch_threads": torch.get_num_threads(), "git_head": sh(["git", "rev-parse", "--short", "HEAD"]),
           "git_branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])}
    status = sh(["git", "status", "--porcelain", "--untracked-files=normal"])
    code_changes = [l for l in status.splitlines() if l and "results/" not in l]
    env["git_dirty_files"] = code_changes
    (out / "git_diff.patch").write_text(sh(["git", "diff", "HEAD", "--", ".", ":(exclude)results"]) + "\n")
    log.info("environment: %s", env)
    check("code matches a commit (no local edits outside results/)", not code_changes,
          f"modified: {code_changes} (full diff in git_diff.patch)" if code_changes else f"HEAD {env['git_head']}",
          warn_only=True)
    tv = env["transformers"] or "0"
    check("transformers < 5 (TransformerLens 2.x is not tested with 5.x)", int(tv.split(".")[0]) < 5, tv)
    return env


# --------------------------------------------------------------------------- 2. tokenizer
def tokenizer_section(tok) -> dict:
    ids = tok.encode("When Mary")
    bos = getattr(tok, "bos_token_id", None)
    check("tokenizer prepends BOS (Gemma is garbage without it)", bos is not None and ids[0] == bos,
          f"encode('When Mary') = {ids[:6]}, bos_token_id = {bos}")
    info = {"bos_token_id": bos, "pad_token_id": getattr(tok, "pad_token_id", None),
            "padding_side": getattr(tok, "padding_side", None), "examples": {}}
    for task in SUITE:
        p = st._with_format(st.generate_pairs(task, 1, 4242, split="check", gen_kwargs=GEN_KW.get(task))[0],
                            TASK_FORMAT[task])
        c, k = tok.encode(p.clean_prompt), tok.encode(p.corrupt_prompt)
        ca = st.answer_token_id(tok, p.clean_prompt, p.clean_target_str)
        info["examples"][task] = {"prompt": p.clean_prompt, "target": p.clean_target_str,
                                  "tail_tokens": [tok.decode([i]) for i in c[-6:]],
                                  "answer_token": None if ca is None else tok.decode([ca]), "aligned": len(c) == len(k)}
        log.info("[%s/%s] %r -> %r | tail %s | answer token %s", task, TASK_FORMAT[task], p.clean_prompt[-40:],
                 p.clean_target_str, info["examples"][task]["tail_tokens"], info["examples"][task]["answer_token"])
        check(f"{task}: answer is one token and prompts are token-aligned", ca is not None and len(c) == len(k),
              f"answer token {info['examples'][task]['answer_token']!r}")
    return info


# --------------------------------------------------------------------------- 3. prompt space (no model)
def space_section(cfg: dict) -> dict:
    """Re-plays build_task_splits' generation: the shared few-shot prefix first, then train and test."""
    out = {}
    level = log.level
    log.setLevel(logging.ERROR)       # generate_pairs warns on every short split; the checks below report it
    try:
        for task in [t for t in st.TASKS if t in cfg["tasks"] or t == "ioi_control"]:
            ov = task_cfg(cfg, task)
            mult = int(ov.get("candidate_multiplier", 3))
            shots = int(ov.get("n_shots", 0))
            gk = ov.get("gen_kwargs")
            t0 = time.time()
            shot_pairs = st.draw_shots(task, shots, cfg["seed"] + 77, gen_kwargs=gk)
            n_shot, exclude, got = len(shot_pairs), st.shot_prompts(shot_pairs), {}
            for split, n, s in (("train", cfg["n_train"], cfg["seed"]), ("test", cfg["n_test"], cfg["seed"] + 10_000)):
                pairs = st.generate_pairs(task, n * mult, s, exclude=exclude, split=split, gen_kwargs=gk)
                got[split] = (len(pairs), n * mult)
                exclude |= st.shot_prompts(pairs)
            out[task] = {"candidate_multiplier": mult, "n_shots_wanted": shots, "n_shots_got": n_shot,
                         "train_got/wanted": got["train"], "test_got/wanted": got["test"],
                         "seconds": round(time.time() - t0, 1)}
            check(f"{task}: prompt space fills train+test candidates",
                  all(g >= w for g, w in got.values()), f"train {got['train'][0]}/{got['train'][1]}, "
                  f"test {got['test'][0]}/{got['test'][1]} (n_train={cfg['n_train']}, n_test={cfg['n_test']}, x{mult})")
            check(f"{task}: few-shot prefix actually gets its {shots} shots", n_shot == shots,
                  f"got {n_shot}/{shots}")
    finally:
        log.setLevel(level)
    return out


# --------------------------------------------------------------------------- probe set
def build_probes(tok, n_task: int) -> list:
    """[(name, ids, target_id or None, meta)] - sanity prompts plus clean prompts from each task (0 and 2 shots)."""
    probes = []
    for prompt, target in SANITY:
        tid = st.answer_token_id(tok, prompt, target)
        probes.append({"name": f"sanity:{prompt[-25:]}", "group": "sanity", "ids": tok.encode(prompt), "target": tid,
                       "target_str": target})
    for task in SUITE:
        for shots in (0, 2):
            pairs = st.generate_with_shots(task, n_task, shots, 9000 + shots, 77, gen_kwargs=GEN_KW.get(task))
            for it in st.tokenize_pairs(tok, pairs, TASK_FORMAT[task]):
                probes.append({"name": f"{task}/{shots}shot:{it.pair.pair_id}", "group": f"{task}/{shots}shot",
                               "ids": it.clean_ids, "target": it.clean_ans, "foil": it.corrupt_ans,
                               "target_str": it.pair.clean_target_str})
    return probes


# --------------------------------------------------------------------------- 4. HF reference
@torch.inference_mode()
def hf_logits(name: str, probes: list, device: torch.device) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM

    hcfg = AutoConfig.from_pretrained(name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    log.info("HF %s loaded on %s (%s) in %.0fs", name, device, dtype, time.time() - t0)
    out = {"final_logit_softcapping": getattr(hcfg, "final_logit_softcapping", None),
           "hidden_size": hcfg.hidden_size, "vocab_size": hcfg.vocab_size, "logits": []}
    for p in probes:
        ids = torch.tensor([p["ids"]], device=device)
        out["logits"].append(model(input_ids=ids).logits[0, -1].float().cpu())
    del model
    free_memory()
    return out


# --------------------------------------------------------------------------- 5. TL paths
@contextlib.contextmanager
def embed_on_gpu(model, device):
    """Temporarily put an offloaded embedding back on the GPU (the plain TL forward needs it there)."""
    if not getattr(model, "_prism_embed_on_cpu", False):
        yield
        return
    model.embed.to(device)
    model.hook_embed.to(device)
    model._prism_embed_on_cpu = False
    try:
        yield
    finally:
        model.embed.to("cpu")
        model.hook_embed.to("cpu")
        model._prism_embed_on_cpu = True
        free_memory()


@torch.inference_mode()
def tl_paths(model, probes: list, device: torch.device) -> dict:
    res = {"ours": [], "tl_plain": [], "ours_gpu": [], "nll": [], "resume": {}}
    for p in probes:
        t = torch.tensor([p["ids"]], device=device)
        res["ours"].append(runtime.last_logits(model, t)[0].cpu())
    try:
        with embed_on_gpu(model, device):
            for p in probes:
                t = torch.tensor([p["ids"]], device=device)
                full = model(t, return_type="logits")[0].float()
                res["tl_plain"].append(full[-1].cpu())
                res["ours_gpu"].append(runtime.last_logits(model, t)[0].cpu())
                lp = torch.log_softmax(full[:-1], -1)
                ref = -float(lp.gather(-1, t[0, 1:, None]).sum())
                res["nll"].append((ref, runtime.token_nll_sum(model, t)))
    except torch.cuda.OutOfMemoryError:
        log.warning("not enough VRAM for the embedding on GPU: tl_plain / ours_gpu skipped")
        res["tl_plain"] = res["ours_gpu"] = res["nll"] = []
        free_memory()
    for layer in (2, model.cfg.n_layers // 2, model.cfg.n_layers - 3):
        d = []
        for p, ref in zip(probes, res["ours"]):
            t = torch.tensor([p["ids"]], device=device)
            r = runtime.resid_at(model, t, layer)
            d.append(float((runtime.last_logits_from(model, r, layer, t)[0].cpu() - ref).abs().max()))
        res["resume"][layer] = max(d)
    return res


def compare(a: list, b: list) -> dict:
    if not a or not b:
        return {}
    dmax = [float((x - y).abs().max()) for x, y in zip(a, b)]
    agree = [int(x.argmax()) == int(y.argmax()) for x, y in zip(a, b)]
    return {"max_abs_diff": max(dmax), "median_max_abs_diff": float(np.median(dmax)), "top1_agree": float(np.mean(agree))}


@torch.inference_mode()
def batching_section(model, tok) -> None:
    """score_pairs with real batches vs one pair at a time (catches row-order / batching bugs)."""
    pairs = st.generate_pairs("ioi_control", 48, 31337, split="batch")
    items = st.tokenize_pairs(tok, pairs, "space_target")
    big = st.score_pairs(model, items, batch_size=128)
    one = st.score_pairs(model, items, batch_size=1)
    d = float(np.abs(big["dl_clean"] - one["dl_clean"]).max())
    check("batched scoring == one-at-a-time scoring", d < 0.25, f"max dl diff = {d:.3f} over {len(items)} pairs (bf16)")


# --------------------------------------------------------------------------- 6. task behaviour
@torch.inference_mode()
def task_section(model, tok, n: int) -> dict:
    out = {}
    for task in SUITE:
        chance = 1.0 / len(st.TASK_STATES[task])
        for fmt in ("auto",):
            for shots in (0, 2, 8):
                pairs = st.generate_with_shots(task, n, shots, 5150 + shots, 77, split="behav", gen_kwargs=GEN_KW.get(task))
                items = st.tokenize_pairs(tok, pairs, fmt)
                key = f"{task}/{fmt}/{shots}shot"
                if not items:
                    out[key] = {"tokenized": 0.0}
                    continue
                sc = st.score_pairs(model, items, 128)
                preds = Counter()
                ex = []
                for b in st.batch_by_length(items, 64):
                    t = st.make_tensors(b, model_device(model))
                    lg = runtime.last_logits(model, t["clean"])
                    preds.update(tok.decode([int(i)]) for i in lg.argmax(-1))
                    if len(ex) < 2:
                        ex.append({"prompt_tail": b[0].pair.clean_prompt[-60:], "target": b[0].pair.clean_target_str,
                                   "top5": topk(tok, lg[0])})
                out[key] = {"tokenized": len(items) / len(pairs), "n": len(items), "chance": round(chance, 3),
                            "clean_top1": float(sc["clean_top1"].mean()), "corrupt_top1": float(sc["corrupt_top1"].mean()),
                            "dl_clean_mean": float(sc["dl_clean"].mean()), "dl_corrupt_mean": float(sc["dl_corrupt"].mean()),
                            "dl_abs_max": float(np.abs(np.r_[sc["dl_clean"], sc["dl_corrupt"]]).max()),
                            "clean_only_kept": float(st.valid_mask(sc, "clean_only").mean()),
                            "strict_kept": float(st.valid_mask(sc, "strict").mean()),
                            "top_predictions": preds.most_common(5), "examples": ex}
                log.info("%s: %s", key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items()
                                          if k != "examples"})
    worst = max(((v["dl_abs_max"], k) for k, v in out.items() if "dl_abs_max" in v), default=(0.0, "-"))
    check(f"task logit gaps within +-{2 * SOFTCAP:.0f} (softcap bound)", worst[0] <= 2 * SOFTCAP + 0.5,
          f"largest gap {worst[0]:.1f} at {worst[1]}")
    ioi = max((v.get("clean_top1", 0), k) for k, v in out.items() if k.startswith("ioi_control"))
    check("IOI clean top-1 >= 0.7 at its best shot count (gemma-2-2b: ~1.0 with 2 shots)", ioi[0] >= 0.7,
          f"clean top-1 {ioi[0]:.2f} at {ioi[1]}")
    for task in st.STATE_TASKS:
        best = max((v.get("clean_top1", 0), k) for k, v in out.items() if k.startswith(task))
        check(f"{task}: best setting beats chance by 1.5x", best[0] >= 1.5 / len(st.TASK_STATES[task]),
              f"best clean top-1 {best[0]:.2f} at {best[1]} (chance {1 / len(st.TASK_STATES[task]):.2f})", warn_only=True)
    return out


# --------------------------------------------------------------------------- report
DIAGNOSIS = """
## How to read this
| hf sanity | tl_plain vs hf | ours vs tl_plain | ours vs ours_gpu | tasks | Meaning |
|---|---|---|---|---|---|
| fails | - | - | - | - | Weights / tokenizer / environment problem: nothing to do with the harness |
| ok | disagree | - | - | - | TransformerLens (or its transformers version) converts Gemma-2 wrongly: pin TL / transformers versions |
| ok | agree | disagree | disagree | - | Bug in src/models/runtime.py head / forward path |
| ok | agree | disagree | agree | - | Bug in the CPU-embedding path only: set `runtime.offload_embed: false` and report |
| ok | agree | agree | agree | at chance | Model is fine; the task prompts are too hard: change prompts / few-shot, not code |
"""


def write_report(out: Path, report: dict) -> int:
    save_json(report, out / "report.json")
    n_fail = sum(c["status"] == "FAIL" for c in CHECKS)
    n_warn = sum(c["status"] == "WARN" for c in CHECKS)
    lines = [f"# Forward / data check ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"**{n_fail} FAIL, {n_warn} WARN, {sum(c['status'] == 'PASS' for c in CHECKS)} PASS**", "",
             "| status | check | detail |", "|---|---|---|"]
    lines += [f"| {c['status']} | {c['check']} | {c['detail']} |" for c in CHECKS]
    if report.get("compare"):
        lines += ["", "## Logit agreement (last position, all probes)", "", "| pair | max abs diff | median | top-1 agree |",
                  "|---|---|---|---|"]
        lines += [f"| {k} | {v['max_abs_diff']:.3f} | {v['median_max_abs_diff']:.3f} | {v['top1_agree']:.2f} |"
                  for k, v in report["compare"].items() if v]
    if report.get("sanity"):
        lines += ["", "## Sanity prompts (top-5 next tokens)", "", "| prompt | target | path | top-5 |", "|---|---|---|---|"]
        for s in report["sanity"]:
            for path, t5 in s["top5"].items():
                lines.append(f"| {s['prompt']!r} | {s['target']!r} | {path} | {t5} |")
    if report.get("tasks"):
        lines += ["", "## Task behaviour (ours)", "", "| setting | n | chance | clean top-1 | corrupt top-1 | dl clean | "
                  "dl corrupt | clean_only kept | top predictions |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in report["tasks"].items():
            if "n" in v:
                lines.append(f"| {k} | {v['n']} | {v['chance']} | {v['clean_top1']:.2f} | {v['corrupt_top1']:.2f} | "
                             f"{v['dl_clean_mean']:.2f} | {v['dl_corrupt_mean']:.2f} | {v['clean_only_kept']:.2f} | "
                             f"{v['top_predictions']} |")
            else:
                lines.append(f"| {k} | 0 (not single-token) | | | | | | | |")
    lines.append(DIAGNOSIS)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    log.info("wrote %s and %s: %d FAIL, %d WARN", out / "report.md", out / "report.json", n_fail, n_warn)
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tier1_core.yaml"))
    ap.add_argument("--overrides", default=None, help="preflight.json with task_overrides (default: config only)")
    ap.add_argument("--out", default="results/check_forward")
    ap.add_argument("--skip-hf", action="store_true", help="skip the Hugging Face reference model")
    ap.add_argument("--hf-device", default=None, help="device for the HF reference (default: same as TL)")
    ap.add_argument("--n-probe", type=int, default=12, help="clean prompts per task/shot setting compared against HF")
    ap.add_argument("--n-task", type=int, default=64, help="pairs per task setting in the behaviour section")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "check.log")
    cfg = load_config(a.config)
    ovp = a.overrides
    if ovp and Path(ovp).exists():
        cfg["task_overrides"] = merge_task_overrides(cfg.get("task_overrides"), load_json(Path(ovp)).get("task_overrides"))
        log.info("task_overrides from %s", ovp)
    GEN_KW.update({t: task_cfg(cfg, t).get("gen_kwargs") for t in st.TASKS})
    device = get_device()
    report = {"config": a.config, "overrides": str(ovp) if ovp else None}

    try:
        log.info("== 1/6 environment")
        report["env"] = env_section(out)
        log.info("== 2/6 tokenizer")
        tok = st.get_tokenizer(False, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
        report["tokenizer"] = tokenizer_section(tok)
        log.info("== 3/6 prompt space (replays build_task_splits with the run's sizes)")
        report["prompt_space"] = space_section(cfg)

        probes = build_probes(tok, a.n_probe)
        log.info("%d probe prompts", len(probes))
        hf = None
        if a.skip_hf:
            check("HF reference", None, "skipped (--skip-hf)")
        else:
            log.info("== 4/6 Hugging Face reference")
            try:
                hf = hf_logits(cfg.get("hf_tokenizer", "google/gemma-2-2b"), probes, torch.device(a.hf_device) if a.hf_device else device)
                report["hf"] = {k: v for k, v in hf.items() if k != "logits"}
                check("HF config final_logit_softcapping == 30", hf["final_logit_softcapping"] == SOFTCAP,
                      str(hf["final_logit_softcapping"]))
            except Exception as e:  # noqa: BLE001
                check("HF reference loads", False, f"{type(e).__name__}: {e}"[:300])
                free_memory()

        log.info("== 5/6 TransformerLens paths")
        t0 = time.time()
        model = load_model(device=device, dtype=get_dtype(device), model_name=cfg["model_name"], tokenizer=tok,
                           runtime_cfg=cfg.get("runtime"))
        runtime.setup_runtime(cfg, model, device)
        mc = model.cfg
        report["tl_cfg"] = {k: str(getattr(mc, k, None)) for k in (
            "n_layers", "d_model", "d_vocab", "normalization_type", "output_logits_soft_cap", "attn_scores_soft_cap",
            "positional_embedding_type", "default_prepend_bos", "use_normalization_before_and_after", "dtype", "device")}
        report["tl_cfg"]["embed_on_cpu"] = bool(getattr(model, "_prism_embed_on_cpu", False))
        log.info("TL loaded in %.0fs: %s", time.time() - t0, report["tl_cfg"])
        check("TL output_logits_soft_cap == 30", float(getattr(mc, "output_logits_soft_cap", 0) or 0) == SOFTCAP,
              report["tl_cfg"]["output_logits_soft_cap"])
        w_scale = float(model.embed.W_E[:4096].float().norm(dim=-1).median().cpu()
                        / model.unembed.W_U[:, :4096].float().norm(dim=0).median().cpu())
        check("W_E carries Gemma's sqrt(d_model) embedding scale", abs(w_scale / mc.d_model ** 0.5 - 1) < 0.2,
              f"median W_E row norm / W_U column norm = {w_scale:.1f} (sqrt(d_model) = {mc.d_model ** 0.5:.1f})", warn_only=True)

        tl = tl_paths(model, probes, device)
        cmp = {"ours vs tl_plain": compare(tl["ours"], tl["tl_plain"]), "ours vs ours_gpu": compare(tl["ours"], tl["ours_gpu"])}
        if hf:
            cmp["tl_plain vs hf"] = compare(tl["tl_plain"], hf["logits"])
            cmp["ours vs hf"] = compare(tl["ours"], hf["logits"])
        report["compare"] = cmp
        for k, v in cmp.items():
            if not v:
                check(f"logits {k}", None, "not computed")
                continue
            tol = (0.1, 1.0) if "hf" not in k else (3.0, 0.9)    # same TL graph: ~exact; TL vs HF in bf16: small drift
            check(f"logits {k}", v["median_max_abs_diff"] <= tol[0] and v["top1_agree"] >= tol[1],
                  f"max {v['max_abs_diff']:.3f}, median {v['median_max_abs_diff']:.3f}, top-1 agree {v['top1_agree']:.2f}")
        allmax = max(float(x.abs().max()) for x in tl["ours"])
        check(f"all logits within the softcap (+-{SOFTCAP:.0f})", allmax <= SOFTCAP + 0.01, f"largest abs logit {allmax:.2f}")
        for layer, d in tl["resume"].items():
            check(f"resume from layer {layer} == full forward", d < 0.1, f"max dlogit {d:.4f}")
        if tl["nll"]:
            d = max(abs(r - o) / max(abs(r), 1) for r, o in tl["nll"])
            check("chunked token_nll_sum == full-logit NLL", d < 0.01, f"max relative diff {d:.4f}")

        sanity = []
        for i, p in enumerate(probes):
            if p["group"] != "sanity":
                continue
            row = {"prompt": SANITY[i][0], "target": SANITY[i][1], "top5": {"ours": topk(tok, tl["ours"][i])}}
            if hf:
                row["top5"]["hf"] = topk(tok, hf["logits"][i])
            row["ours_top1_ok"] = p["target"] is not None and int(tl["ours"][i].argmax()) == p["target"]
            row["hf_top1_ok"] = bool(hf) and p["target"] is not None and int(hf["logits"][i].argmax()) == p["target"]
            sanity.append(row)
        report["sanity"] = sanity
        n_ok = sum(r["ours_top1_ok"] for r in sanity)
        check("sanity prompts: ours top-1 correct on >= 5/7", n_ok >= 5, f"{n_ok}/{len(sanity)} correct")
        if hf:
            n_hf = sum(r["hf_top1_ok"] for r in sanity)
            check("sanity prompts: HF top-1 correct on >= 5/7", n_hf >= 5, f"{n_hf}/{len(sanity)} correct")
            groups = sorted({p["group"] for p in probes if p["group"] != "sanity"})
            acc = {}
            for g in groups:
                idx = [i for i, p in enumerate(probes) if p["group"] == g]
                acc[g] = {path: float(np.mean([int(src[i].argmax()) == probes[i]["target"] for i in idx]))
                          for path, src in (("hf", hf["logits"]), ("ours", tl["ours"]))}
            report["task_top1_hf_vs_ours"] = acc
            log.info("task top-1 on probe prompts, HF vs ours: %s", acc)

        log.info("== 6/6 batching + task behaviour")
        batching_section(model, tok)
        report["tasks"] = task_section(model, tok, a.n_task)
    except Exception as e:  # noqa: BLE001 - keep everything gathered so far
        log.exception("check_forward crashed")
        check("script ran to the end", False, f"{type(e).__name__}: {e}"[:300])
    return write_report(out, report)


if __name__ == "__main__":
    sys.exit(main())

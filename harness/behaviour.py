"""E13: what the base model does on the language families, before any SAE is involved.

Per (task, family) over the test candidates (rank order, unfiltered): clean accuracy (top-1 on A is A's answer), corrupt
accuracy, the share kept by each filter (f_A_correct, f_both_correct, the pilot's rule, and clean_only), the median clean logit
difference, and the length-matching success recorded when the set was built. Two gates, kept separate:

  Gate A (handoff, per language and task; scripts/analyze_tasks.py VIABLE): clean accuracy >= 0.6 AND clean_only keep >= 0.3.
      Decides whether the language is runnable for that task. A failure is a result, not an error.
  Gate B (PREREGISTRATION, "task accuracy within 10 points of English"): |acc_lang - acc_en| <= 0.10 on the same task.
      Governs gap comparisons (E14); a language can pass A and fail B.

Intervals are Wilson 95% for proportions. Pure functions here; scripts/run_behaviour.py drives the model.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

VIABLE_ACC, VIABLE_KEEP, GATE_B_POINTS = 0.6, 0.3, 0.10
PILOT_MODE = {"entity_track": "clean_only", "add_chain": "strict", "people_cities": "strict"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


@torch.inference_mode()
def score_rows(model, rows: Sequence[dict], budget, device, log=print) -> Dict[str, np.ndarray]:
    """Last-position logits of the clean (A) and corrupt (B) run of every row -> top-1 ids and answer logits."""
    from harness.batching import group_by_length
    from harness.device import empty_cache, is_oom
    from src.models import runtime
    n = len(rows)
    out = {k: np.zeros(n, dtype=np.int64) for k in ("top1_A", "top1_B")}
    out.update({k: np.zeros(n) for k in ("lA_ansA", "lA_ansB", "lB_ansA", "lB_ansB")})
    for key, col in (("clean_ids", "A"), ("corrupt_ids", "B")):
        groups = group_by_length([len(r[key]) for r in rows])
        for T, idx in groups.items():
            a = 0
            while a < len(idx):
                b = min(a + budget.rows_for(T), len(idx))
                ii = idx[a:b]
                t = torch.tensor([rows[i][key] for i in ii], dtype=torch.long, device=device)
                try:
                    lg = runtime.last_logits(model, t)
                except Exception as e:  # noqa: BLE001
                    if is_oom(e) and budget.shrink():
                        empty_cache()
                        continue
                    raise
                ansA = torch.tensor([rows[i]["ans_A"] for i in ii], device=lg.device)
                ansB = torch.tensor([rows[i]["ans_B"] for i in ii], device=lg.device)
                out[f"top1_{col}"][ii] = lg.argmax(-1).cpu().numpy()
                out[f"l{col}_ansA"][ii] = lg.gather(1, ansA[:, None])[:, 0].float().cpu().numpy()
                out[f"l{col}_ansB"][ii] = lg.gather(1, ansB[:, None])[:, 0].float().cpu().numpy()
                a = b
    return out


def summarize(task: str, rows: Sequence[dict], s: Dict[str, np.ndarray], min_logit_diff: float = 2.0) -> dict:
    from harness.tasks import filter_flags
    from src.data import synthetic_tasks as st
    n = len(rows)
    D_A, D_B = s["lA_ansA"] - s["lA_ansB"], s["lB_ansA"] - s["lB_ansB"]        # logit(ans_A) - logit(ans_B) on each run
    flags = filter_flags(rows, s["top1_A"], s["top1_B"], D_A, D_B, PILOT_MODE[task], min_logit_diff)
    sc = {"clean_top1": flags["f_A_correct"], "corrupt_top1": s["top1_B"] == np.array([r["ans_B"] for r in rows]),
          "dl_clean": D_A, "dl_corrupt": D_B}
    clean_only = st.valid_mask(sc, "clean_only", min_logit_diff)
    cnt = lambda m: int(np.sum(m))
    row = {"n": n, "acc_clean": cnt(flags["f_A_correct"]) / n, "acc_corrupt": cnt(sc["corrupt_top1"]) / n,
           "keep_f_A_correct": cnt(flags["f_A_correct"]) / n, "keep_f_both_correct": cnt(flags["f_both_correct"]) / n,
           "keep_pilot_rule": cnt(flags["f_pilot"]) / n, "pilot_rule": PILOT_MODE[task], "keep_clean_only": cnt(clean_only) / n,
           "median_D_clean": float(np.median(D_A)), "median_D_corrupt": float(np.median(D_B)),
           "chance_acc": 1.0 / len(rows[0]["cand_ids"])}
    for k, m in (("acc_clean", flags["f_A_correct"]), ("keep_clean_only", clean_only), ("keep_pilot_rule", flags["f_pilot"])):
        row[k + "_lo"], row[k + "_hi"] = wilson(cnt(m), n)
    row["gate_A_viable"] = bool(row["acc_clean"] >= VIABLE_ACC and row["keep_clean_only"] >= VIABLE_KEEP)
    return row


def add_gate_b(table: List[dict]) -> None:
    """Gate B per (task, family): |acc - acc of the English family of the same task| <= 0.10 (English rows: n/a)."""
    en = {(r["task"], r.get("split", "test")): r["acc_clean"] for r in table if r["lang"] == "en"}
    for r in table:
        key = (r["task"], r.get("split", "test"))                 # compare within the same split
        if r["lang"] == "en" or key not in en:
            r["acc_gap_vs_en"], r["gate_B_within_10pts"] = (0.0, "n/a (English)") if r["lang"] == "en" else (float("nan"), "no English row")
        else:
            r["acc_gap_vs_en"] = r["acc_clean"] - en[key]
            r["gate_B_within_10pts"] = bool(abs(r["acc_gap_vs_en"]) <= GATE_B_POINTS)

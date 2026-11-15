"""E13: gates, intervals, summary logic (no model), then the tiny-model script on the real prompt sets."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import behaviour as B
from harness import langfamilies as LF

ROOT = Path(__file__).resolve().parents[2]


def test_wilson_interval_brackets_and_edges():
    lo, hi = B.wilson(30, 100)
    assert lo < 0.30 < hi and 0.21 < lo and hi < 0.40
    assert B.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12) and B.wilson(10, 10)[1] == pytest.approx(1.0)
    assert np.isnan(B.wilson(0, 0)[0])


def _rows(n):
    return [{"ans_A": 5, "ans_B": 6, "cand_ids": [5, 6, 7, 8]} for _ in range(n)]


def test_summarize_counts_and_gate_a():
    n = 10
    s = {"top1_A": np.array([5] * 7 + [9] * 3), "top1_B": np.array([6] * 10),
         "lA_ansA": np.array([6.0] * 7 + [0.0] * 3), "lA_ansB": np.zeros(n), "lB_ansA": np.zeros(n), "lB_ansB": np.array([6.0] * n)}
    r = B.summarize("add_chain", _rows(n), s)
    assert r["acc_clean"] == 0.7 and r["acc_corrupt"] == 1.0 and r["chance_acc"] == 0.25
    assert r["keep_f_A_correct"] == 0.7 and r["keep_pilot_rule"] == 0.7           # strict: both correct and D_A > 2
    assert r["keep_clean_only"] == 0.7 and r["gate_A_viable"] is True
    s["top1_A"][:] = 9
    r2 = B.summarize("add_chain", _rows(n), s)
    assert r2["acc_clean"] == 0.0 and r2["gate_A_viable"] is False


def test_gate_b_is_relative_to_english_of_the_same_task():
    t = [{"task": "entity_track", "lang": "en", "acc_clean": 0.80}, {"task": "entity_track", "lang": "hi", "acc_clean": 0.72},
         {"task": "entity_track", "lang": "fr", "acc_clean": 0.60}, {"task": "add_chain", "lang": "hi", "acc_clean": 0.9}]
    B.add_gate_b(t)
    assert t[0]["gate_B_within_10pts"] == "n/a (English)"
    assert t[1]["gate_B_within_10pts"] is True and t[2]["gate_B_within_10pts"] is False
    assert t[3]["gate_B_within_10pts"] == "no English row"


def test_every_pack_is_unvalidated_until_a_human_signs():
    assert set(LF.VALIDATED) == set(LF.LANG_OF) and not any(LF.VALIDATED.values())


def test_script_tiny_end_to_end_and_resume():
    cmd = [sys.executable, str(ROOT / "scripts" / "run_behaviour.py"), "--tiny", "--tasks", "add_chain,people_cities", "--langs", "en,hi",
           "--limit", "12", "--profile", "dev"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    dirs = sorted((ROOT / "tests" / "_tiny_runs" / "E13").glob("*"), key=lambda p: p.stat().st_mtime)
    out = dirs[-1]
    try:
        rows = list(csv.DictReader((out / "behaviour.csv").open()))
        assert [(x["task"], x["family"]) for x in rows] == [("add_chain", "rs_en"), ("add_chain", "rs_hi"), ("people_cities", "pc_en"), ("people_cities", "pc_hi")]
        assert all(x["validated"] == "False" and float(x["length_matching_success"]) == 1.0 and int(x["n"]) == 12 for x in rows)
        from harness.schema import validate_manifest
        m = json.loads((out / "manifest.json").read_text())
        assert m["status"] == "complete" and m["table"] == "behaviour.csv"
        assert validate_manifest(m) == ["tiny-model run (test plumbing only; never ingested)"]
        r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0 and len(list(csv.DictReader((out / "behaviour.csv").open()))) == 4      # nothing redone
    finally:
        shutil.rmtree(out, ignore_errors=True)

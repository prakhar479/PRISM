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


def test_validated_reflects_only_the_signoff_file(tmp_path, monkeypatch):
    import harness.langfamilies as lf
    monkeypatch.setattr(lf, "SIGNOFF_FILE", tmp_path / "none.json")
    assert set(lf.VALIDATED) == set(lf.LANG_OF) and not any(lf.VALIDATED.values()) and not lf.signoffs()


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
        assert all(float(x["length_matching_success"]) == 1.0 and int(x["n"]) == 12 for x in rows)
        from harness.schema import validate_manifest
        m = json.loads((out / "manifest.json").read_text())
        assert m["status"] == "complete" and m["table"] == "behaviour.csv"
        assert validate_manifest(m) == ["tiny-model run (test plumbing only; never ingested)"]
        r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0 and len(list(csv.DictReader((out / "behaviour.csv").open()))) == 4      # nothing redone
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_signoff_is_bound_to_the_pack_content(tmp_path, monkeypatch):
    import harness.langfamilies as lf
    monkeypatch.setattr(lf, "SIGNOFF_FILE", tmp_path / "signoff.json")
    assert not lf.is_validated("ot_hi")
    h = lf.pack_hash("ot_hi")
    (tmp_path / "signoff.json").write_text(json.dumps({"ot_hi": {"by": "x", "date": "d", "pack_hash": h}}))
    assert lf.is_validated("ot_hi") and lf.VALIDATED["ot_hi"] and not lf.VALIDATED["rs_hi"]
    (tmp_path / "signoff.json").write_text(json.dumps({"ot_hi": {"by": "x", "date": "d", "pack_hash": "stale"}}))
    assert not lf.is_validated("ot_hi")                      # the pack changed after the sign-off: void
    assert lf.pack_hash("ot_hi") != lf.pack_hash("ot_hi_latn")


def test_sign_script_requires_a_name_and_writes_hashes(tmp_path):
    import os, subprocess
    f = tmp_path / "signoff.json"
    env = {**os.environ, "PRISM_SIGNOFF_FILE": str(f)}
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts" / "sign_templates.py"), *a], capture_output=True, text=True, cwd=ROOT, env=env)
    r = run("--langs", "hi")
    assert r.returncode != 0 and not f.exists()          # no --by, no sign-off
    r = run("--langs", "hi", "--by", "Test Reviewer")
    assert r.returncode == 0, r.stderr
    d = json.loads(f.read_text())
    assert set(d) == {"ot_hi", "rs_hi", "pc_hi"} and d["ot_hi"]["by"] == "Test Reviewer"
    assert "VALID" in run("--list").stdout

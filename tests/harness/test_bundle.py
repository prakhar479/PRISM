"""Bundle -> ingest round trip with schema and checksum checks (handoff test 18)."""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness import provenance
from harness.schema import PER_PAIR, validate_per_pair

ROOT = Path(__file__).resolve().parents[2]


def root_commit() -> str:
    """Never tagged (HEAD is tagged right after a release, so HEAD cannot stand in for 'untagged')."""
    return provenance.git("rev-list", "--max-parents=0", "HEAD").split()[0]


def fake_run(runs_root: Path, exp="E0", h="abc123def456", tiny=False, commit=None):
    d = runs_root / exp / h
    d.mkdir(parents=True)
    rows = []
    for i in range(4):
        for cond in ("clean_A", "clean_B", "feat_A_errB"):
            rows.append({"pair_id": f"t-test-{i:05d}", "split": "test", "family": "pilot", "task": "t", "layer": 12,
                         "sae": "L12_16k_l082", "condition": cond, "estimator": "", "source": "", "r": -1, "seed": -1,
                         "ans_A_id": 1, "ans_B_id": 2, "donor_status": "n/a", "f_none": True, "f_A_correct": True,
                         "f_both_correct": True, "f_pilot": True, "logit_A": 1.0 + i, "logit_B": 0.5, "lse": 3.0,
                         "top1_id": 1, "cand_logits": [1.0 + i, 0.5]})
    df = pd.DataFrame(rows)
    assert validate_per_pair(df) == []
    df.to_parquet(d / "per_pair.parquet")
    m = {"exp_id": exp, "run_hash": h, "status": "complete", "config": {"name": "fake"}, "profile": {}, "machine": {},
         "git": {"commit": commit or provenance.git_state()["commit"]}, "packages": {}, "model": {}, "sae_revision": "x",
         "prompt_sets": {}, "seeds": {}, "timings": {}, "smoke": False, "tiny": tiny, "wall_time_s": 1.0}
    (d / "manifest.json").write_text(json.dumps(m))
    return d


def run(script, args, env):
    return subprocess.run([sys.executable, f"scripts/{script}", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def test_bundle_ingest_round_trip(tmp_path):
    src, dst = tmp_path / "run_side", tmp_path / "dev_side"
    fake_run(src)
    env = {**os.environ, "HL_RUNS_ROOT": str(src)}
    r = run("bundle_results.py", ["--tag", "t1", "--out-dir", str(tmp_path / "b")], env)
    assert r.returncode == 0, r.stderr
    tar = next((tmp_path / "b").glob("*.tar.gz"))
    env2 = {**os.environ, "HL_RUNS_ROOT": str(dst)}
    r = run("ingest_results.py", [str(tar), "--allow-untagged"], env2)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dst / "E0" / "abc123def456" / "per_pair.parquet").exists() and (dst / "RESULTS_INDEX.md").exists()
    r = run("ingest_results.py", [str(tar), "--allow-untagged"], env2)          # same bundle again: no-op
    assert r.returncode == 0 and "already present 1" in r.stdout
    old = tmp_path / "run_side_old"
    fake_run(old, h="0dd0dd0dd0dd", commit=root_commit())
    r = run("bundle_results.py", ["--tag", "t2", "--out-dir", str(tmp_path / "b2")], {**os.environ, "HL_RUNS_ROOT": str(old)})
    assert r.returncode == 0 and "0 runs" in r.stdout                           # default: only runs made at HEAD
    r = run("bundle_results.py", ["--tag", "t2", "--out-dir", str(tmp_path / "b3"), "--all-commits"],
            {**os.environ, "HL_RUNS_ROOT": str(old)})
    assert r.returncode == 0, r.stderr
    tar2 = next((tmp_path / "b3").glob("*.tar.gz"))
    r = run("ingest_results.py", [str(tar2)], {**env2, "HL_RUNS_ROOT": str(tmp_path / "dev3")})
    assert r.returncode == 1 and "no run-* tag" in r.stdout                     # untagged commits are refused


def test_ingest_refuses_tampering_and_tiny(tmp_path):
    src = tmp_path / "run_side"
    fake_run(src, h="tiny000000aa", tiny=True)
    env = {**os.environ, "HL_RUNS_ROOT": str(src)}
    assert run("bundle_results.py", ["--tag", "t1", "--out-dir", str(tmp_path / "b")], env).returncode == 0
    tar = next((tmp_path / "b").glob("*.tar.gz"))
    r = run("ingest_results.py", [str(tar), "--allow-untagged"], {**os.environ, "HL_RUNS_ROOT": str(tmp_path / "d")})
    assert r.returncode == 0 and "merged 0" in r.stdout                        # bundler already leaves tiny runs out
    shutil_tar = tmp_path / "b" / tar.name
    shutil_tar.write_bytes(shutil_tar.read_bytes()[:-10] + b"0123456789")       # corrupt the archive
    r = run("ingest_results.py", [str(tar), "--allow-untagged"], {**os.environ, "HL_RUNS_ROOT": str(tmp_path / "d")})
    assert r.returncode == 1 and "checksum" in r.stdout


def test_schema_catches_missing_and_duplicates():
    df = pd.DataFrame([{k: (1 if v == "int" else 1.0 if v == "float" else True if v == "bool" else [1.0] if v == "list" else "x")
                        for k, v in PER_PAIR.items()}] * 2)
    assert any("duplicate" in e for e in validate_per_pair(df))
    assert any("missing column lse" in e for e in validate_per_pair(df.drop(columns=["lse"])))


def test_bundle_dir_follows_runs_root(tmp_path):
    """A runs_root on a scratch disk puts the bundles on that disk too; the default stays <repo>/bundles."""
    import importlib.util
    from harness.profile import REPO_ROOT, Profile
    spec = importlib.util.spec_from_file_location("bundle_results", ROOT / "scripts" / "bundle_results.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(m)
    assert m.default_out_dir(Profile(name="x", runs_root="runs")) == REPO_ROOT / "bundles"
    assert m.default_out_dir(Profile(name="x", runs_root=str(tmp_path / "w" / "runs"))) == tmp_path / "w" / "bundles"

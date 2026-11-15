"""E12 fidelity: pure maths, then the tiny model with (a) a perfectly reconstructing SAE and (b) the random-SAE script run."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from harness import fidelity as F
from harness.batching import Budget
from harness.model import tiny_gemma2
from harness.sae import load_params

ROOT = Path(__file__).resolve().parents[2]


def test_recovered_formula_and_nan():
    assert F.recovered(3.0, 3.0, 6.0) == 1.0 and F.recovered(3.0, 6.0, 6.0) == 0.0
    assert F.recovered(3.0, 4.5, 6.0) == pytest.approx(0.5) and np.isnan(F.recovered(3.0, 3.0, 3.0))


def test_summarize_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    n = 60
    base = rng.uniform(2, 3, n)
    per = {"pred": np.full(n, 100.0), "pos": np.full(n, 124.0), "nll_clean": base * 100, "nll_splice": (base + 0.2) * 100,
           "nll_mean": (base + 2) * 100, "nll_zero": (base + 5) * 100, "fvu_num": rng.uniform(0.1, 0.2, n), "fvu_den": np.ones(n),
           "eps_ratio_sum": np.full(n, 12.4), "l0_sum": np.full(n, 124 * 50.0)}
    a, b = F.summarize(per, n * 100, 300), F.summarize(per, n * 100, 300)
    assert a == b
    assert a["recovered_mean"] == pytest.approx(0.9, abs=1e-6) and a["recovered_zero"] == pytest.approx(1 - 0.2 / 5, abs=1e-6)
    assert a["recovered_mean_lo"] <= a["recovered_mean"] <= a["recovered_mean_hi"]
    assert a["l0"] == pytest.approx(50) and a["eps_over_x"] == pytest.approx(0.1)


def _identity_sae(tmp_path, d):
    """JumpReLU SAE with relu(x) and relu(-x) latents: reconstructs every input exactly."""
    eye = np.eye(d, dtype=np.float32)
    np.savez(tmp_path / "id.npz", W_enc=np.concatenate([eye, -eye], 1), W_dec=np.concatenate([eye, -eye], 0),
             b_enc=np.zeros(2 * d, np.float32), b_dec=np.zeros(d, np.float32), threshold=np.zeros(2 * d, np.float32))
    return load_params(tmp_path / "id.npz", 1, "128")


def test_perfect_sae_recovers_everything_on_the_tiny_model(tmp_path):
    model = tiny_gemma2(512, n_layers=3, d_model=64, seed=1)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 512, (6, 32), generator=g).tolist()
    sae = _identity_sae(tmp_path, 64)
    per = F.measure(model, sae, 1, ids, 4, Budget(4096, 3), torch.device("cpu"))
    s = F.summarize(per, 6 * (32 - 4 - 1), 50)
    assert s["ce_delta"] == pytest.approx(0.0, abs=1e-4)           # splice == clean
    assert s["fvu"] < 1e-8 and s["eps_over_x"] < 1e-5
    assert s["recovered_mean"] == pytest.approx(1.0, abs=1e-3)
    assert 0 < s["l0"] <= 64


def test_batch_size_does_not_change_results(tmp_path):
    model = tiny_gemma2(512, n_layers=3, d_model=64, seed=2)
    ids = torch.randint(3, 512, (7, 24), generator=torch.Generator().manual_seed(1)).tolist()
    sae = _identity_sae(tmp_path, 64)
    a = F.measure(model, sae, 1, ids, 4, Budget(4096, 7), torch.device("cpu"))
    b = F.measure(model, sae, 1, ids, 4, Budget(4096, 2), torch.device("cpu"))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-4, atol=1e-6, err_msg=k)


def test_script_tiny_end_to_end_and_resume(tmp_path):
    lang = tmp_path / "lang"; lang.mkdir()
    ids = torch.randint(3, 500, (10, 32), generator=torch.Generator().manual_seed(3)).tolist()
    (lang / "xx.json").write_text(json.dumps({"ids": ids, "skip_positions": 4, "lang": "xx", "script": "Latn", "genre": "t",
                                              "licence": "CC0", "tokens_per_word": 1.5}))
    (lang / "MANIFEST.json").write_text(json.dumps({"xx": {"file": "xx.json", "hash": "h", "sha256": "s", "n_docs": 10, "tokens_scored": 270}}))
    cmd = [sys.executable, str(ROOT / "scripts" / "run_fidelity.py"), "--tiny", "--sets", "xx", "--layers", "1", "--lang-dir", str(lang),
           "--n-boot", "20", "--profile", "dev"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    dirs = list((ROOT / "tests" / "_tiny_runs" / "E12").glob("*"))
    assert len(dirs) >= 1
    out = max(dirs, key=lambda p: p.stat().st_mtime)
    rows = (out / "fidelity.csv").read_text().strip().splitlines()
    assert len(rows) == 2 and rows[1].startswith("xx,xx,Latn")
    from harness.schema import validate_manifest
    assert validate_manifest(json.loads((out / "manifest.json").read_text())) == ["tiny-model run (test plumbing only; never ingested)"]
    assert (out / "per_window_xx_L1.npz").exists() and json.loads((out / "manifest.json").read_text())["exp_id"] == "E12"
    r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0 and len((out / "fidelity.csv").read_text().strip().splitlines()) == 2    # nothing redone
    import shutil; shutil.rmtree(out, ignore_errors=True)

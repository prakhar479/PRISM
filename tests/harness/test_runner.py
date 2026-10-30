"""Runner end to end on the tiny model (handoff tests 16-resume, 17-paths). Outputs go to tests/_tiny_runs only."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from harness import run as R

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    d = yaml.safe_load((ROOT / "configs" / "experiments" / "E2.yaml").read_text())
    d.update(exp_id="E2", name="tiny_test", cells=d["cells"][:1], bootstrap=20, smoke_pairs=6,
             conditions=["feat_A_errB", "feat_A_meanPos", "feat_A_resampC_sameAns", "feat_A_covnoise"], noise_seeds=[0])
    d["directions"].update(conditions=["repair"], ranks=[0, 1], random_seeds=2)
    p = tmp_path_factory.mktemp("cfg") / "E2_tiny.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.mark.slow
def test_tiny_smoke_runs_resumes_and_writes_portable_manifest(cfg_path):
    args = ["--config", str(cfg_path), "--profile", "dev", "--tiny", "--smoke"]
    assert R.main(args) == 0
    runs = sorted((ROOT / "tests" / "_tiny_runs" / "E2").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
    run_dir = runs[-1].parent
    m = json.loads(runs[-1].read_text())
    assert m["status"] == "complete" and m["tiny"] and m["smoke"]
    for k in ("cache_root", "data_root", "runs_root"):
        assert not str(m["profile"][k]).startswith("/")
    pp = pd.read_parquet(run_dir / "per_pair.parquet")
    assert {"clean_A", "clean_B", "full_A", "full_B", "repair"} <= set(pp.condition)
    assert set(pp.r[pp.condition == "repair"]) == {0, 1}
    done = sorted((run_dir / "parts").glob("*.done"))
    mt = {p: p.stat().st_mtime for p in done}
    assert R.main(args) == 0                                     # resume: every unit is skipped
    assert all(p.stat().st_mtime == mt[p] for p in done)
    s = json.loads((run_dir / "summary.json").read_text())
    assert all(c["max_abs_dD"] < 1e-3 for c in s["hook_checks"])


def test_dry_run_all_release_configs_fit_run_profile(capsys):
    for f in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        assert R.main(["--config", str(f), "--profile", "run", "--dry-run"]) == 0, f
        plan = json.loads(capsys.readouterr().out)
        assert plan["all_fit"] and plan["jobs"]

"""Batch-2 configs and queues: generated files in sync, every config dry-runs, every queue's jobs / scripts / download targets resolve."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
QUEUES = sorted((ROOT / "queue").glob("pilot-x*-server.yaml"))


def test_batch2_configs_are_generated_valid_and_dry_run(capsys):
    from harness import run as R
    from harness.config import ExpConfig
    rc = subprocess.run([sys.executable, "scripts/make_batch2_configs.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    files = sorted((ROOT / "configs" / "batch2").glob("P*.yaml"))
    assert {f.stem for f in files} == {"P20", "P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28"}
    for p in files:
        c = ExpConfig.load(p)
        assert all(cell.role == "exploratory" for cell in c.cells)
        rc = R.main(["--config", str(p), "--profile", "server", "--dry-run"])
        plan = json.loads(capsys.readouterr().out)
        assert rc in (0, 2) and plan["jobs"], p


def test_batch2_protocol_in_every_config():
    """Catalogue S10: DAS v2 with the kl objective (or the difference of means), >= 2 independent seeds, random nulls."""
    from harness.config import ExpConfig, parse_das2
    for p in sorted((ROOT / "configs" / "batch2").glob("P*.yaml")):
        d = ExpConfig.load(p).directions
        ests = [e for e in d.estimators if e != "diffmean"]
        assert d.random_seeds >= 5 and all(parse_das2(e)["objective"] == "kl" for e in ests), p
        seeds = {parse_das2(e)["seed"] for e in ests}
        assert len(seeds) >= 2 or p.stem == "P22", (p, seeds)


@pytest.mark.skipif(not QUEUES, reason="the queue files live on the branches batch2-atlas / batch2-repair")
@pytest.mark.parametrize("qpath", QUEUES, ids=lambda p: p.stem)
def test_batch2_queue_resolves(qpath):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_prefetch as rp
    from harness.queue import Queue
    q = Queue.load(qpath)
    for c in q.configs():
        assert (ROOT / c).exists(), c
    for j in q.jobs:
        if j.kind == "script":
            assert (ROOT / j.script).exists(), j.script
        for dep in j.depends_on or []:
            assert dep in {x.id for x in q.jobs}, (j.id, dep)
    t = rp.targets(q)
    repos = {m[0] for m in t["models"]}
    assert "google/gemma-2-2b" in repos
    assert len(t["saes"]) >= 5
    ids = {j.id for j in q.jobs}
    assert "bundle" in ids and "publish" in ids and all(f"{c.split('/')[-1][:-5]}_fin" in ids for c in q.configs())
    # every experiment of the queue is allowed by every profile
    for prof in ("dev", "run", "server"):
        allowed = yaml.safe_load((ROOT / "profiles" / f"{prof}.yaml").read_text())["allowed_experiments"]
        assert all(Path(c).stem in allowed for c in q.configs()), prof

"""A cell without enough train pairs (e.g. a language the model cannot do) is skipped and recorded; the other cells run and
the run still finalizes (code review 2026-11-17, blocker on E14)."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DRIVER = '''
import sys
sys.path.insert(0, %r)
import harness.tasks as T
real = T.select
def select(rows, flags, split, flt, n, groups=None, per_group=False):
    idx = real(rows, flags, split, flt, n, groups=groups, per_group=per_group)
    if split == "train" and groups == ["ot_hi"]:
        return []                                      # simulate: no ot_hi train pair passes the fit filter
    return idx
T.select = select
import harness.run as R
sys.exit(R.main(sys.argv[1:]))
'''


def test_cell_without_train_pairs_is_skipped_and_run_finalizes(tmp_path):
    d = yaml.safe_load((ROOT / "configs" / "experiments" / "E14.yaml").read_text())
    d["cells"] = [c for c in d["cells"] if c["eval_families"][0] in ("ot_hi", "rs_en")]   # ot_hi native + ot_en->ot_hi + rs_en
    d["directions"]["random_seeds"] = 1
    d["bootstrap"], d["smoke_pairs"] = 10, 4
    d["exp_id"] = "E14"
    cfg = tmp_path / "E14skip.yaml"
    cfg.write_text(yaml.safe_dump(d))
    drv = tmp_path / "drive.py"
    drv.write_text(DRIVER % str(ROOT))
    base = [sys.executable, str(drv), "--config", str(cfg), "--profile", "dev", "--tiny", "--smoke"]
    r = subprocess.run(base, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    runs = sorted((ROOT / "tests" / "_tiny_runs" / "E14").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
    m = json.loads(runs[-1].read_text())
    assert m["status"] == "complete"
    skipped = [c for c in m["cells"] if c.get("skipped")]
    assert len(skipped) == 1 and "ot_hi" in skipped[0]["cell"] and "fit-ot_hi" in skipped[0]["cell"]
    assert len(m["cells"]) == 3
    r2 = subprocess.run(base + ["--stage", "3", "--finalize-only"], capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0, r2.stderr[-2000:]

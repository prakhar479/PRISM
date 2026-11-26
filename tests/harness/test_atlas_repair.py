"""scripts/make_atlas.py and scripts/make_repair.py on tiny runs (plumbing and bookkeeping; the numbers of a random model mean nothing)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def _atlas_cfg(tmp_path, layers=(1, 2, 3, 4, 5)):
    d = yaml.safe_load((ROOT / "configs" / "batch2" / "P20.yaml").read_text())
    cells = []
    for L in layers:
        for task in ("people_cities", "ioi_control"):
            for l0 in ("small", "big"):
                c = [x for x in d["cells"] if x["task"] == task][0]
                cells.append({**c, "layer": L, "sae": {"dict": f"gs2_1b_all_L{L}_16k_{l0}"}})
    d["cells"] = cells
    d["prompt_sets"] = {k: v for k, v in d["prompt_sets"].items() if k in ("people_cities", "ioi_control")}
    d.update(bootstrap=20, smoke_pairs=5)
    d["directions"].update(estimators=["das8v2_kl_s0", "das8v2_kl_s1"], random_seeds=2)
    p = tmp_path / "P20_tiny.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.mark.slow
def test_make_atlas_on_tiny_runs(tmp_path):
    from harness import run as R
    import make_atlas as ma
    cfg = _atlas_cfg(tmp_path)
    assert R.main(["--config", str(cfg), "--profile", "dev", "--tiny", "--smoke", "--allow-dirty"]) == 0
    rc = subprocess.run([sys.executable, "scripts/run_dict_fidelity.py", "--config", str(cfg), "--tiny", "--n-windows", "4", "--n-boot", "20",
                         "--extras", "--allow-dirty"], cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr[-800:]
    out = tmp_path / "atlas"
    sys.argv = ["make_atlas.py", "--runs", str(ROOT / "tests" / "_tiny_runs"), "--out", str(out), "--boot", "20"]
    assert ma.main() == 0
    cells = pd.read_csv(out / "atlas_cells.csv")
    p20 = cells[cells.exp == "P20"]
    assert len(p20) == 5 * 2 * 2 and set(p20.n_seeds) == {2} and (p20.capture_sd >= 0).all()
    assert (out / "atlas_curves_P20.csv").exists() and (out / "atlas_h3_P20.csv").exists()
    text = (out / "ATLAS.md").read_text()
    assert "H3 (P20)" in text and "## H4" in text
    met = pd.read_csv(out / "atlas_metrics.csv")
    assert {"fvu", "ce_delta"} <= set(met.metric)


def test_h3_and_h4_logic_on_planted_curves():
    import make_atlas as ma
    layers = list(range(26))
    dip = pd.DataFrame({"task": "t", "layer": layers, "capture": [0.9 - 0.3 * np.exp(-((L - 12) / 3.0) ** 2) for L in layers],
                        "capture_sd_seed": 0.03})
    h = ma.h3_rows(dip).iloc[0]
    assert h.min_layer == 12 and h.in_mid_third and h.holds and h.dip > 0.15
    mono = dip.assign(capture=[0.9 - 0.01 * L for L in layers])
    assert not ma.h3_rows(mono).iloc[0].in_mid_third
    flat = dip.assign(capture=[0.8 + 0.001 * (L % 3) for L in layers], capture_sd_seed=0.05)
    assert not ma.h3_rows(flat).iloc[0].holds
    mt = pd.DataFrame([{"set": "x", "task": "ALL", "metric": "fvu", "spearman_pooled": -0.3, "spearman_within_layer": -0.2},
                       {"set": "x", "task": "ALL", "metric": "splice_over_noise", "spearman_pooled": -0.8, "spearman_within_layer": -0.75}])
    assert ma.h4_verdict(mt)[0].startswith("GO (splice_over_noise")
    mt2 = mt.assign(spearman_pooled=[-0.8, -0.5], spearman_within_layer=[-0.8, -0.4])
    assert ma.h4_verdict(mt2)[0].startswith("NO-GO")
    assert ma.h4_verdict(pd.DataFrame())[0].startswith("NOT RUN")


def test_within_layer_rank_correlation_removes_the_depth_trend():
    import make_atlas as ma
    rows = []
    rng = np.random.default_rng(0)
    for L in range(6):
        for j in range(5):
            m = rng.normal()
            rows.append({"task": "t", "layer": L, "capture": 0.1 * L + 0.05 * m, "metric": 10 * L + m})   # metric tracks depth AND the within-layer noise
    d = pd.DataFrame(rows)
    assert ma.rank_demeaned(d, "capture", "metric", ["task", "layer"]) > 0.9
    shuffled = d.assign(metric=d.metric.sample(frac=1, random_state=1).to_numpy())
    assert abs(ma.rank_demeaned(shuffled.assign(metric=shuffled.groupby("layer").metric.transform(lambda s: s.sample(frac=1, random_state=2).to_numpy())),
                                "capture", "metric", ["task", "layer"])) < 0.6


def _cells(rows):
    base = {"exp": "P23", "n_pairs": 150, "capture": 0.7, "capture_sd_seed": 0.02, "oracle": False, "k": 0, "kind": "base", "base": "B", "addon": ""}
    out = []
    for r in rows:
        out.append({**base, **r})
    return pd.DataFrame(out)


def _rule_frame(sens, var, rnd, k=8, gap=0.3, hi=0.05, dce=-0.01):
    rows = [{"task": "T", "layer": 1, "sae": "B", "gap": gap}]
    for kind, c, h in (("sens", sens, sens + 0.05), ("var", var, var + 0.05), ("rnd", rnd, hi)):
        rows.append({"task": "T", "layer": 1, "sae": f"B__{kind}{k}", "addon": f"{kind}{k}", "kind": kind, "k": k, "oracle": False, "gap": gap * (1 - c),
                     "closure_ref": c, "closure_ref_lo": c - 0.05, "closure_ref_hi": h, "d_ce_delta": dce})
    return _cells(rows)


def test_h1_rule_on_planted_closures():
    import make_repair as mr
    good = pd.concat([_rule_frame(0.6, 0.1, 0.02, k=k).assign(task=t) for t in ("A", "B") for k in (2, 8, 32)])
    rows, verdict = mr.h1_rule(good, (2, 8, 32), 2)
    assert verdict.startswith("GO") and rows.holds.all() and (rows.passes == 3).all()
    tie = pd.concat([_rule_frame(0.3, 0.28, 0.02, k=k).assign(task=t) for t in ("A", "B") for k in (2, 8, 32)])
    assert mr.h1_rule(tie, (2, 8, 32), 2)[1].startswith("NO-GO")                    # sens not 2x var
    noise = pd.concat([_rule_frame(0.6, 0.1, 0.0, k=k, hi=0.7).assign(task=t) for t in ("A", "B") for k in (2, 8, 32)])
    assert mr.h1_rule(noise, (2, 8, 32), 2)[1].startswith("NO-GO")                  # inside the random-subspace CI
    costly = pd.concat([_rule_frame(0.6, 0.1, 0.02, k=k, dce=0.05).assign(task=t) for t in ("A", "B") for k in (2, 8, 32)])
    assert mr.h1_rule(costly, (2, 8, 32), 2)[1].startswith("NO-GO")                 # raises the CE splice cost
    small = pd.concat([_rule_frame(0.6, 0.1, 0.02, k=k, gap=0.02).assign(task=t) for t in ("A", "B") for k in (2, 8, 32)])
    assert mr.h1_rule(small, (2, 8, 32), 2)[1].startswith("UNCLEAR")                # gap below GAP_MIN: nothing counted


def test_h2_and_h6_verdicts_on_planted_closures():
    import make_repair as mr

    def p27(clin, mlp, cmlp, kl_gap, sens, klsens):
        rows = [{"exp": "P27", "task": "T", "layer": 1, "sae": "B", "gap": 0.4, "addon": "", "closure_ref": np.nan},
                {"exp": "P27", "task": "T", "layer": 1, "sae": "L1_klft", "gap": kl_gap, "addon": "", "closure_ref": np.nan}]
        for a, c in (("clin", clin), ("mlp1024", mlp), ("cmlp1024", cmlp), ("lin", 0.1), ("sens32", sens)):
            rows.append({"exp": "P27", "task": "T", "layer": 1, "sae": f"B__{a}", "gap": 0.4 * (1 - c), "addon": a, "closure_ref": c})
        rows.append({"exp": "P27", "task": "T", "layer": 1, "sae": "L1_klft__sens32", "gap": 0.4 * (1 - klsens), "addon": "sens32", "closure_ref": klsens})
        return _cells(rows)
    assert "decoder problem" in mr.h2_verdict(p27(0.5, 0.6, 0.55, 0.3, 0.5, 0.7))[0][2]
    assert "information problem" in mr.h2_verdict(p27(0.05, 0.6, 0.1, 0.3, 0.5, 0.7))[0][2]
    assert "mixed" in mr.h2_verdict(p27(0.02, 0.05, 0.03, 0.3, 0.5, 0.7))[0][2]
    st = mr.h6_verdict(p27(0.5, 0.6, 0.55, 0.28, 0.5, 0.7))[0]               # klft closes 0.30, sens32 0.5, together 0.7
    assert st[-1] is True and abs(st[1] - 0.3) < 1e-9
    assert mr.h6_verdict(p27(0.5, 0.6, 0.55, 0.28, 0.5, 0.52))[0][-1] is False


@pytest.mark.slow
def test_make_repair_on_tiny_runs(tmp_path):
    import yaml
    import make_repair as mr
    import fit_sensitivity as fs
    from harness import run as R
    for smoke in (["--smoke"], []):
        assert fs.main(["--tiny", *smoke, "--layers", "1", "--generic", "generic/pile10k_6000x128_skip100.json"]) == 0
    d = yaml.safe_load((ROOT / "configs" / "batch2" / "P23.yaml").read_text())
    base = [c for c in d["cells"] if c["task"] == "people_cities" and not c["sae"].get("addon")][0]
    modes = ["lin", "var4", "sens4", "rnd4", "osens4", "ovar4", "fis4"]
    d["cells"] = [{**base, "layer": 1, "sae": {"width": "16k", "l0": 40, "rule": "explicit"}}] + \
                 [{**base, "layer": 1, "sae": {"width": "16k", "l0": 40, "rule": "explicit", "addon": m}} for m in modes]
    d["prompt_sets"] = {"people_cities": d["prompt_sets"]["people_cities"]}
    d.update(bootstrap=20, smoke_pairs=5)
    d["directions"].update(estimators=["das8v2_kl_s0", "das8v2_kl_s1"], random_seeds=2)
    cfg = tmp_path / "P23_tiny.yaml"
    cfg.write_text(yaml.safe_dump(d))
    assert R.main(["--config", str(cfg), "--profile", "dev", "--tiny", "--smoke", "--allow-dirty"]) == 0
    rc = subprocess.run([sys.executable, "scripts/run_dict_fidelity.py", "--config", str(cfg), "--tiny", "--n-windows", "4", "--n-boot", "20",
                         "--allow-dirty"], cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr[-800:]
    out = tmp_path / "repair"
    sys.argv = ["make_repair.py", "--runs", str(ROOT / "tests" / "_tiny_runs"), "--out", str(out), "--boot", "20"]
    assert mr.main() == 0
    t = pd.read_csv(out / "repair_cells_P23.csv")
    assert len(t) == 8 and set(t.kind) == {"base", "lin", "var", "sens", "rnd", "fis"} and t[t.addon == "osens4"].oracle.all()
    assert {"closure_ref", "closure_own", "dcapture", "ce_delta", "d_ce_delta"} <= set(t.columns)
    assert (out / "REPAIR.md").exists() and "H1 rule on P23" in (out / "REPAIR.md").read_text()


def test_capture_is_masked_where_the_subspace_has_no_effect():
    """P22: the difference-of-means direction does nothing below layer 6, so 'capture' there is a ratio of noise (-1.5 .. 7.9)."""
    import make_atlas as ma
    t = pd.DataFrame({"capture_feat_A_along_U": [3.2, 0.9, 0.8, 5.0], "full_nldr": [0.001, 0.6, 0.10, 0.5],
                      "random_p95_full_nldr": [0.0, 0.01, 0.0, 0.48]})
    m = ma.mask_low_effect(t)
    assert list(m.no_effect) == [True, False, True, True]                  # tiny effect / not above the null + 0.05
    assert np.isnan(m.capture_feat_A_along_U[[0, 2, 3]]).all() and m.capture_feat_A_along_U[1] == 0.9


def test_grid_effects_separate_width_and_sparsity():
    import make_atlas as ma
    rows = []
    for L in (7, 13):
        for w, dw in (("16k", -0.05), ("65k", 0.0), ("262k", 0.05)):
            for l0, dl in (("small", -0.1), ("medium", 0.0), ("big", 0.1)):
                rows.append({"task": "t", "layer": L, "sae": f"L{L}_gs2_1b_L{L}_{w}_{l0}", "capture": 0.6 + dw + dl})
    g = ma.grid_effects(pd.DataFrame(rows)).set_index(["factor", "level"]).capture_dev
    assert abs(g[("l0v", "big")] - 0.1) < 1e-9 and abs(g[("l0v", "small")] + 0.1) < 1e-9 and abs(g[("width", "262k")] - 0.05) < 1e-9

"""E8: vectorized metrics == harness.metrics, gap / share / specificity on constructed data, corrections."""
import numpy as np
import pandas as pd
import pytest

from harness import e8
from harness import metrics as M
from harness.analysis import _out

N = 80


def rows(condition, frac_A, estimator="", source="", r=-1, seed=-1, run_noise=0.0, rng=None):
    """N pairs; the first round(frac_A * N) pairs predict A's answer (id 1), the rest B's (id 2)."""
    rng = rng or np.random.default_rng(0)
    k = int(round(frac_A * N))
    hit = np.arange(N) < k
    la = np.where(hit, 5.0, 0.0) + run_noise * rng.standard_normal(N)
    lb = np.where(hit, 0.0, 5.0)
    lse = np.logaddexp(la, lb)
    return pd.DataFrame({"pair_id": [f"p{i:03d}" for i in range(N)], "condition": condition, "estimator": estimator,
                         "source": source, "r": r, "seed": seed, "ans_A_id": 1, "ans_B_id": 2, "logit_A": la,
                         "logit_B": lb, "lse": lse, "top1_id": np.where(hit, 1, 2), "f_pilot": True,
                         "task": "entity_track", "layer": 12, "sae": "L12_16k_l082", "fit_families": "pilot",
                         "cand_logits": [[a, b] for a, b in zip(la, lb)]})


def cell(same=0.9, other=0.2, generic=0.1, random_fracs=(0.2,) * 20, base=0.2, with_other=True):
    parts = [rows("clean_A", 1.0), rows("clean_B", 0.0), rows("full_A", 1.0), rows("feat_A_errB", base),
             rows("feat_A_zero", 0.5)]
    for r in (1, 2):
        parts.append(rows("repair", same, "pca_pooled", "same_task", r))
        if with_other:
            parts.append(rows("repair", other, "pca_pooled", "other_task:add_chain", r))
        parts.append(rows("repair", generic, "pca_pooled", "generic", r))
        parts += [rows("repair", f, "random", "random", r, s) for s, f in enumerate(random_fracs)]
    return pd.concat(parts, ignore_index=True)


def test_vectorized_metrics_equal_per_condition():
    rng = np.random.default_rng(1)
    g = pd.concat([rows("clean_A", 0.9, rng=rng, run_noise=1.0), rows("clean_B", 0.1, rng=rng, run_noise=1.0),
                   rows("feat_A_errB", 0.55, rng=rng, run_noise=2.0)])
    a, b, p = (g[g.condition == c].set_index("pair_id") for c in ("clean_A", "clean_B", "feat_A_errB"))
    ansA, ansB = a.ans_A_id.to_numpy(), a.ans_B_id.to_numpy()
    want = M.per_condition(_out(p), _out(a), _out(b), ansA, ansB)
    elig = (b.top1_id.to_numpy() == ansB).astype(float)
    got = e8.metric_samples(e8.Arr(p, ansA), e8.Arr(a, ansA), e8.Arr(b, ansA), elig, np.ones(N))
    for k in e8.METRICS:
        assert got[k] == pytest.approx(want[k], rel=1e-12, abs=1e-12), k


def test_gap_share_and_specificity_on_constructed_cell():
    gaps, shares, spec = e8.analyse_cell(cell(), "f_pilot", n_boot=2000)
    g = {(x["metric"], x["condition"]): x for x in gaps}
    assert g[("flip_rate", "feat_A_errB")]["gap"] == pytest.approx(0.8)
    assert g[("flip_rate", "feat_A_zero")]["gap"] == pytest.approx(0.5)
    s = {(x["metric"], x["source"], x["r"], x["seed"]): x for x in shares}
    assert s[("flip_rate", "same_task", 1, -1)]["share_closed"] == pytest.approx((0.9 - 0.2) / 0.8)
    sp = {(x["metric"], x["r"]): x for x in spec if x["estimator"] == "pca_pooled"}
    row = sp[("flip_rate", 1)]
    assert row["best_control"] == "other_task:add_chain"
    assert row["diff"] == pytest.approx((0.9 - 0.2) / 0.8 - 0.0)       # other task closes 0 of the gap
    assert row["p_margin"] < 0.01 and row["n_random_seeds"] == 20 and row["missing_controls"] == ""
    assert row["to_half_r"] == 1.0


def test_specificity_fails_margin_and_flags_missing_controls():
    _, _, spec = e8.analyse_cell(cell(same=0.4, with_other=False, random_fracs=(0.2, 0.3, 0.25)), "f_pilot", n_boot=2000)
    row = next(x for x in spec if x["metric"] == "flip_rate" and x["r"] == 1)
    assert row["missing_controls"] == "other_task" and row["n_random_seeds"] == 3
    assert row["share_same"] == pytest.approx(0.25) and row["p_margin"] > 0.5
    assert np.isnan(row["to_half_r"])


def test_random_p95_is_taken_over_seeds_per_resample():
    fr = tuple(0.2 + 0.01 * i for i in range(20))                      # shares 0.0 .. 0.2375
    _, shares, spec = e8.analyse_cell(cell(other=0.0, generic=0.0, random_fracs=fr), "f_pilot", n_boot=500)
    row = next(x for x in spec if x["metric"] == "flip_rate" and x["r"] == 1)
    want = np.percentile([(round(f * N) / N - 0.2) / 0.8 for f in fr], 95)        # realized fractions of N pairs
    assert row["random_p95"] == pytest.approx(want, abs=1e-9) and row["best_control"] == "random_p95"


def test_corrections_canonical_rows_and_roles():
    man = lambda h, cfg: {"exp_id": "E1", "run_hash": h, "config": {"name": cfg, "filters": ["f_pilot"]}}
    c_conf = cell()
    c_expl = cell().assign(task="var_binding", layer=8, sae="L8_16k_l071")
    c_conf_3 = cell(random_fracs=(0.2, 0.2, 0.2))                       # same cell, fewer seeds: not canonical
    out = e8.analyse_runs([(man("a", "x"), pd.concat([c_conf, c_expl])), (man("b", "y"), c_conf_3)], n_boot=300)
    sp = out["spec"]
    fl = sp[(sp.metric == "flip_rate") & (sp.r == 1) & (sp.estimator == "pca_pooled")]
    conf = fl[fl.task == "entity_track"]
    assert set(conf.role) == {"confirmatory"} and conf.set_index("run_hash").canonical.to_dict() == {"a": True, "b": False}
    assert conf[conf.canonical].p_holm.notna().all() and conf[~conf.canonical].p_holm.isna().all()
    assert fl[fl.task == "var_binding"].p_bh.notna().all() and set(fl[fl.task == "var_binding"].role) == {"exploratory"}
    assert bool(conf[conf.run_hash == "b"].underpowered_random.iloc[0])
    assert e8.role_of("add_chain", 18, "L18_16k_l074") == "confirmatory"
    assert e8.role_of("add_chain", 18, "L18_262k_l0100") == "exploratory"
    assert e8.role_of("entity_track", 12, "L12_16k_l082", "fresh") == "exploratory"


def test_no_base_gap_is_flagged_and_left_out_of_corrections():
    man = {"exp_id": "E2", "run_hash": "z", "config": {"name": "x", "filters": ["f_pilot"]}}
    c = cell(base=1.0)                                                  # features-only already flips every pair
    out = e8.analyse_runs([(man, c)], n_boot=300)
    sp = out["spec"]
    row = sp[(sp.metric == "flip_rate") & (sp.r == 1) & (sp.estimator == "pca_pooled")].iloc[0]
    assert row.base_gap == 0 and not row.base_gap_resolved
    assert np.isnan(row.share_same) and np.isnan(row.p_margin) and np.isnan(row.p_holm)


def test_weights_equal_fancy_indexing_and_cluster_bootstrap_resamples_families():
    rng = np.random.default_rng(3)
    g = pd.concat([rows("clean_A", 0.9, rng=rng, run_noise=1.0), rows("clean_B", 0.1, rng=rng, run_noise=1.0),
                   rows("feat_A_errB", 0.5, rng=rng, run_noise=2.0)])
    a, b, p = (g[g.condition == c].set_index("pair_id") for c in ("clean_A", "clean_B", "feat_A_errB"))
    ansA, ansB = a.ans_A_id.to_numpy(), a.ans_B_id.to_numpy()
    elig = (b.top1_id.to_numpy() == ansB).astype(float)
    ix = [rng.integers(0, N, N) for _ in range(5)]
    got = e8.metric_samples(e8.Arr(p, ansA), e8.Arr(a, ansA), e8.Arr(b, ansA), elig, e8.weights(ix, N))
    for k, i in enumerate(ix):
        want = M.per_condition(_out(p.iloc[i]), _out(a.iloc[i]), _out(b.iloc[i]), ansA[i], ansB[i])
        assert got["nldr"][k] == pytest.approx(want["nldr"], rel=1e-10)
        assert got["flip_rate"][k] == pytest.approx(want["flip_rate"], rel=1e-10)
    fam = np.repeat(["f1", "f2", "f3", "f4"], N // 4)
    c = cell().assign(family=lambda d: d.pair_id.map(dict(zip([f"p{i:03d}" for i in range(N)], fam))))
    gp, _, _ = e8.analyse_cell(c, "f_pilot", n_boot=400, cluster_col="family")
    gp_pairs, _, _ = e8.analyse_cell(c, "f_pilot", n_boot=400)
    w = lambda rows: next(x for x in rows if x["metric"] == "flip_rate" and x["condition"] == "feat_A_zero")
    assert w(gp)["ci"] == "cluster:family" and w(gp)["gap"] == pytest.approx(w(gp_pairs)["gap"])
    assert (w(gp)["gap_hi"] - w(gp)["gap_lo"]) > (w(gp_pairs)["gap_hi"] - w(gp_pairs)["gap_lo"])   # 4 clusters: wider


def test_gate_rows_only_fresh_families_are_the_gate():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    import make_gate as G

    fam = {f"p{i:03d}": ["ot_placed", "ot_bin", "ot_drawer", "ot_placed"][i % 4] for i in range(N)}
    fresh = pd.concat([rows("clean_A", 1.0), rows("clean_B", 0.0), rows("full_A", 1.0), rows("feat_A_errB", 0.9),
                       rows("feat_A_meanPos", 0.8)]).assign(fit_families="fresh", family=lambda d: d.pair_id.map(fam))
    pilot = fresh.assign(fit_families="fresh", family="pilot", pair_id=lambda d: "q" + d.pair_id)
    fresh = fresh.assign(f_A_correct=True)
    pilot = pilot.assign(f_A_correct=True)
    fresh_pilotfit = fresh.assign(fit_families="pilot")                    # E3 cell fit pilot -> eval fresh
    m = {"exp_id": "E3", "run_hash": "h3", "config": {"filters": ["f_pilot", "f_A_correct"]}}
    df = pd.DataFrame(G.gate_rows(m, pd.concat([fresh, fresh_pilotfit, pilot]), n_boot=300))
    prim = df[(df.metric == "flip_rate") & (df.condition == "feat_A_meanPos")]
    g1 = prim[prim.use == "gate"]
    assert len(g1) == 1 and g1["filter"].iloc[0] == "f_pilot" and g1.fit_families.iloc[0] == "fresh"
    sens = prim[prim.use.str.startswith("sensitivity")]
    assert set(zip(sens["filter"], sens.fit_families)) == {("f_A_correct", "fresh"), ("f_pilot", "pilot"),
                                                             ("f_A_correct", "pilot")}
    gate = prim[prim.use == "gate"].iloc[0]
    assert gate.eval_group == "fresh" and gate.shortfall == pytest.approx(0.2) and gate.criterion_met_pairs
    assert pd.notna(gate.lo_family_cluster) and gate.families == "ot_bin,ot_drawer,ot_placed"
    ref = prim[prim.use.str.startswith("reference")].iloc[0]
    assert ref.eval_group == "pilot" and pd.isna(ref.lo_family_cluster) and ref.criterion_met_family_cluster is None

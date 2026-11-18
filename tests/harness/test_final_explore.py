"""The last exploratory run (E17-E22, 2026-11-18): estimator / condition grammar, cache-key compatibility of plain DAS, the
E21 matched-L0 rule, width-aware E12 tables and exploratory alignment thresholds."""
import csv
import sys
from pathlib import Path

import pytest

from harness import final as FN
from harness.conditions import is_steer, steer_scale
from harness.config import ExpConfig, das_rank, is_estimator, is_u_condition, parse_das
from harness.profile import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_das_grammar():
    assert parse_das("das4") == {"k": 4, "epochs": 3, "lr": 1e-2, "seed": None}       # E9 defaults: cache keys unchanged
    assert parse_das("das2_e10_s1") == {"k": 2, "epochs": 10, "lr": 1e-2, "seed": 1}
    assert parse_das("das1_e10_lr3e-2")["lr"] == pytest.approx(0.03)
    assert parse_das("das16")["k"] == 16 and das_rank("das8_e10") == 8
    for bad in ("das", "das0", "dasx", "das1_x", "pca"):
        assert parse_das(bad) is None
    assert is_estimator("answer_sep") and is_estimator("das2_s2") and not is_estimator("das_e3")


def test_steer_grammar():
    assert is_steer("steer_x1") and is_steer("steer_x0.25") and is_steer("steer_x16")
    assert not is_steer("steer_x0") and not is_steer("steer_y")
    assert steer_scale("steer_x0.5") == 0.5 and steer_scale("steer_x4") == 4.0
    assert is_u_condition("steer_x8") and is_u_condition("repair") and not is_u_condition("feat_A_errB")


@pytest.mark.parametrize("exp", ["E17", "E18", "E19", "E20", "E22"])
def test_explore_configs_load(exp):
    c = ExpConfig.load(REPO_ROOT / "configs" / "experiments" / f"{exp}.yaml")
    assert all(cell.role == "exploratory" for cell in c.cells)
    assert c.directions.random_seeds >= 20                      # the E9/E10 rules need >= 20 random U
    for e in c.directions.estimators:                           # every DAS rank is among the evaluated ranks
        if e.startswith("das"):
            assert das_rank(e) in c.directions.ranks, e


def test_e17_reuses_e9_setup():
    """E17's plain das<k> must hit E9's cached rotations: same prompt set, layer, train size, fit filter and seed."""
    e9 = ExpConfig.load(REPO_ROOT / "configs" / "experiments" / "E9.yaml")
    e17 = ExpConfig.load(REPO_ROOT / "configs" / "experiments" / "E17.yaml")
    assert e17.prompt_sets["add_chain"] == e9.prompt_sets["add_chain"]
    assert (e17.n_train, e17.fit_filter, e17.seed) == (e9.n_train, e9.fit_filter, e9.seed)


def test_match16k_rule():
    from run_fidelity import l0_for
    assert l0_for(12, "16k") == 82 and l0_for(18, "16k") == 74
    assert l0_for(12, "65k", "match16k") == 72 and l0_for(18, "65k", "match16k") == 64
    assert l0_for(12, "262k", "match16k") == 67
    assert l0_for(12, "16k", "match16k") == 82


def _write(d: Path, rows):
    d.mkdir(parents=True)
    with (d / "fidelity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_make_e12_keys_on_sae(tmp_path):
    import make_e12
    base = {"recovered_mean_lo": 0, "recovered_mean_hi": 0, "fvu_lo": 0, "fvu_hi": 0, "fvu": 0}
    row = lambda s, sae, v: {**base, "set": s, "layer": 12, "sae": sae, "recovered_mean": v,
                             "recovered_mean_lo": v - .001, "recovered_mean_hi": v + .001}
    _write(tmp_path / "E12" / "a", [row("en_pile", "L12_16k_l082", .986), row("hi_deva_web", "L12_16k_l082", .95)])
    _write(tmp_path / "E12" / "b", [row("en_pile", "L12_65k_l072", .99), row("hi_deva_web", "L12_65k_l072", .97)])
    t = make_e12.build(make_e12.gather(tmp_path))
    assert len(t) == 4                                          # the 65k run does not replace the 16k rows
    hi65 = next(r for r in t if r["set"] == "hi_deva_web" and r["sae"] == "L12_65k_l072")
    assert hi65["recovered_mean_minus_en"] == pytest.approx(-0.02)   # compared with English at the same SAE


def test_alignment_exploratory_ranks():
    assert FN.alignment(0.12, 8) == "unrelated" and FN.alignment(0.14, 8) == "partial"   # frozen k <= 8 unchanged
    assert FN.alignment(0.17, 16) == "unrelated" and FN.alignment(0.30, 32) == "partial"
    assert FN.alignment(0.6, 64) == "aligned" and FN.alignment(0.2, 3) == ""            # unknown rank: no label


def test_plain_das_cache_key_unchanged(tmp_path):
    """Plain das<k> must resolve to the cache entry E9 wrote (k, epochs 3, lr 1e-2, config seed): reuse on gnode084."""
    import types
    from harness.cache import Cache
    from harness.run import Runner
    c = Cache(tmp_path)
    sp_entry = c.entry("spans", t="add_chain", l=18)
    seen = {}

    class Spy(Cache):
        def entry(self, kind, **kw):
            seen.setdefault(kind, []).append(kw)
            return super().entry(kind, **kw)

    fake = types.SimpleNamespace(tiny=False, smoke=False, finalizing=True, cache=Spy(tmp_path), cfg=types.SimpleNamespace(seed=0),
                                 spec=types.SimpleNamespace(key=lambda: {"m": 1}), DAS_VERSION=1, spans_entry=lambda t, l: sp_entry)
    fs = types.SimpleNamespace(digest="abc")
    for est in ("das4", "das4_e10_s1"):
        with pytest.raises(RuntimeError):
            Runner.das_direction(fake, "add_chain", 18, [], [0, 1], None, fs, est)
    old = dict(spans=sp_entry.key, fitset="abc", k=4, epochs=3, lr=1e-2, seed=0, model={"m": 1}, version=1)
    assert seen["das"][0] == old                                 # the exact E9 key
    assert c.entry("das", **seen["das"][0]).key == c.entry("das", **old).key
    assert seen["das"][1]["epochs"] == 10 and seen["das"][1]["seed"] == 1


def test_norm_matched_noise_and_extras_summary():
    import numpy as np
    import torch
    from harness import fidelity as F
    eps = torch.randn(50, 16) * torch.linspace(0.1, 3, 50)[:, None]
    eta = F.norm_matched_noise(eps, torch.Generator().manual_seed(0))
    assert torch.allclose(eta.norm(dim=-1), eps.norm(dim=-1), rtol=1e-5)
    assert abs(float((eta * eps).sum(-1).div(eps.norm(dim=-1) ** 2).mean())) < 0.2        # direction unrelated to eps
    n = 30
    per = {k: np.ones(n) for k in ("pred", "pos", "fvu_num", "fvu_den", "eps_ratio_sum", "l0_sum")}
    per.update(nll_clean=np.full(n, 2.0), nll_splice=np.full(n, 2.3), nll_mean=np.full(n, 9.0), nll_zero=np.full(n, 9.5),
               nll_noise=np.full(n, 2.1), nll_perturb=np.full(n, 2.05))
    s = F.summarize(per, n, n_boot=20)
    assert s["ce_delta"] == pytest.approx(0.3) and s["noise_delta"] == pytest.approx(0.1)
    assert s["splice_over_noise"] == pytest.approx(3.0) and s["perturb_delta"] == pytest.approx(0.05)
    per.pop("nll_noise"); per.pop("nll_perturb")
    assert "noise_delta" not in F.summarize(per, n, n_boot=5)                          # E12 rows unchanged without extras


def test_das_learned_removes_shared_init():
    import numpy as np
    import make_final as mf
    I = mf.das_init(2304, 2, 0)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2304, 1)); v -= I @ (I.T @ v); v /= np.linalg.norm(v)
    U = np.linalg.qr(np.c_[I[:, :1], v])[0]                     # half init, half a learned direction
    L, cap = mf.das_learned(U, "das2", 0)
    assert L.shape[1] == 1 and abs(abs(float(L[:, 0] @ v[:, 0])) - 1) < 1e-6
    assert cap == pytest.approx(0.5, abs=1e-6)

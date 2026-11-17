"""Pieces added for the final compute pass: steering (E10), the mean A-B difference, DAS rotations (E9), language families in
configs (E14/E11), and the configs and queue themselves."""
from pathlib import Path

import numpy as np
import pytest
import torch

from harness import conditions as C
from harness import families as F
from harness.config import ExpConfig
from harness.directions import FitSet
from harness.queue import Queue

ROOT = Path(__file__).resolve().parents[2]


def spans(n_pairs=3, rows_per=2, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = n_pairs * rows_per
    x_A, x_B = torch.randn(m, d, generator=g), torch.randn(m, d, generator=g)
    sp = C.Spans.__new__(C.Spans)
    sp.pair_ids = [f"p{i}" for i in range(n_pairs)]
    sp.row_ptr = np.arange(0, m + 1, rows_per)
    sp.x_A, sp.x_B = x_A, x_B
    sp.slot = ["s0", "final"] * n_pairs
    sp.off = np.array([1, 0] * n_pairs)
    return sp


def test_steer_adds_scale_times_mean_diff_norm_inside_span_U_only():
    sp = spans()
    U = torch.linalg.qr(torch.randn(8, 2, generator=torch.Generator().manual_seed(1)))[0]
    m = torch.randn(sp.x_B.shape, generator=torch.Generator().manual_seed(2))
    out = C.steer(sp, U, m, 2.0)
    push = out - sp.x_B
    assert torch.allclose(push.norm(dim=-1), 2.0 * m.norm(dim=-1), atol=1e-5)          # norm-matched by construction
    assert torch.allclose(push - (push @ U) @ U.T, torch.zeros_like(push), atol=1e-5)    # inside span(U)
    m_perp = m - (m @ U) @ U.T
    assert torch.allclose(C.steer(sp, U, m_perp, 2.0), sp.x_B, atol=1e-5)               # nothing to push along


def test_steer_never_reads_x_A():
    sp = spans()
    U = torch.eye(8)[:, :2]
    m = torch.ones(sp.x_B.shape)
    a = C.steer(sp, U, m, 1.0)
    sp.x_A = sp.x_A * 100
    assert torch.equal(a, C.steer(sp, U, m, 1.0))


def test_mean_diff_is_per_key_mean_and_requires_train():
    sp = spans()
    fs = FitSet(tuple(sp.pair_ids))
    md = C.MeanDiff.fit(fs, sp)
    D = sp.x_A - sp.x_B
    assert torch.allclose(md.by_key["final"], D[1::2].mean(0), atol=1e-6)
    assert torch.allclose(md.task, D.mean(0), atol=1e-6)
    with pytest.raises(Exception):
        C.MeanDiff.fit(FitSet(("other",)), sp)


def test_language_families_expand_and_fresh_is_unchanged():
    assert F.expand("entity_track", ["ot_hi"]) == ["ot_hi"]
    assert F.expand("people_cities", ["pc_en"]) == ["pc_en"]
    assert F.expand("entity_track", ["fresh"]) == ["ot_placed", "ot_bin", "ot_drawer"]
    with pytest.raises(ValueError):
        F.expand("entity_track", ["ot_klingon"])


@pytest.mark.parametrize("exp", ["E9", "E10", "E11", "E14", "E4_1m", "E9b", "E10b", "E2r"])
def test_final_configs_validate(exp):
    c = ExpConfig.load(ROOT / "configs" / "experiments" / f"{exp}.yaml")
    c.validate()
    roles = {x.role for x in c.cells}
    if exp in ("E9", "E10"):
        conf = [x for x in c.cells if x.role == "confirmatory"]
        assert [(x.task, x.layer) for x in conf] == [("add_chain", 18)]         # RS L18 is the GO cell (humans, 2026-11-16)
    else:
        assert roles == {"exploratory"}


def test_e14_covers_every_language_native_and_english_transfer():
    c = ExpConfig.load(ROOT / "configs" / "experiments" / "E14.yaml")
    pairs = {(x.task, x.fit_families[0], x.eval_families[0]) for x in c.cells}
    assert len(pairs) == 27
    for pre, task in (("ot", "entity_track"), ("rs", "add_chain"), ("pc", "people_cities")):
        for l in ("en", "hi", "hi_latn", "hinglish", "fr"):
            assert (task, f"{pre}_{l}", f"{pre}_{l}") in pairs
            if l != "en":
                assert (task, f"{pre}_en", f"{pre}_{l}") in pairs


def test_final_queue_loads_and_chains_shared_stage1():
    q = Queue.load(ROOT / "queue" / "final-server.yaml")
    ids = {j.id: j for j in q.jobs}
    assert {"E9_s3", "E10_s3", "E11_s3", "E14_s3", "E4_1m_s3", "bundle", "publish"} <= set(ids)
    assert "E9_s1" in ids["E10_s1"].depends_on and "E9_s3" in ids["E10_s3"].depends_on   # DAS trained once, by E9
    assert all(ids[f"{e}_fin"].gpu == "none" for e in ("E9", "E10", "E11", "E14", "E4_1m"))


@pytest.mark.parametrize("queue", ["final-server", "release3-server", "release2-server"])
def test_every_queued_experiment_is_allowed_by_every_profile(queue):
    from harness.profile import load_profile
    q = Queue.load(ROOT / "queue" / f"{queue}.yaml")
    ids = {ExpConfig.load(ROOT / c).exp_id for c in q.configs()}
    for prof in ("server", "run", "dev"):
        allowed = set(load_profile(prof, env={}).allowed_experiments)
        assert ids <= allowed, (prof, ids - allowed)


def test_bonus_jobs_are_lowest_priority_and_chained():
    q = Queue.load(ROOT / "queue" / "final-server.yaml")
    ids = {j.id: j for j in q.jobs}
    core = min(ids[f"{e}_s3"].priority for e in ("E9", "E10", "E14", "E11"))
    assert all(ids[f"{e}_s3"].priority < core for e in ("E9b", "E10b", "E2r"))
    assert "E9b_s3" in ids["E10b_s3"].depends_on and "E9b_fin" in ids["bundle"].depends_on

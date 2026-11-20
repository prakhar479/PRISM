"""Pure tensor modules: conditions, directions, metrics, stats (handoff tests 4-11, 14)."""
import numpy as np
import pytest
import torch

from harness import conditions as C
from harness import directions as Dr
from harness import metrics as M
from harness import sae as S
from harness import stats as St

D, N = 24, 12


def make_spans(seed=0, n=N, d=D, prefix="t-train-", slots=True):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = np.full(n, 20)
    fd = rng.integers(12, 17, size=n)
    lens = T - fd
    ptr = np.concatenate([[0], np.cumsum(lens)])
    pos = torch.cat([torch.arange(f, t) for f, t in zip(fd, T)])
    off = torch.cat([torch.arange(t - f - 1, -1, -1) for f, t in zip(fd, T)])
    m = int(ptr[-1])
    sp = C.Spans([f"{prefix}{i:05d}" for i in range(n)], ptr, pos, off, torch.randn(m, d, generator=g),
                 torch.randn(m, d, generator=g),
                 slot=[("final" if o == 0 else ("query_obj" if o == 1 else f"off:{int(o)}")) for o in off] if slots else None,
                 meta={"T": T, "first_diff": fd, "ans_A": rng.integers(0, 3, size=n), "ans_B": rng.integers(3, 6, size=n),
                       "family": np.array(["f0"] * n, dtype=object)})
    sp.xh_A, sp.xh_B = 0.8 * sp.x_A, 0.8 * sp.x_B
    sp.e_A, sp.e_B = sp.x_A - sp.xh_A, sp.x_B - sp.xh_B
    return sp


def orth(d, r, seed=1):
    return Dr.random(d, r, seed).r(r)


# ---------------------------------------------------------------- conditions
def test_repair_r0_and_full_basis():
    sp = make_spans()
    assert torch.equal(C.build("repair", sp, U=orth(D, 0)), C.build("feat_A_errB", sp))
    full = C.build("repair", sp, U=torch.eye(D))
    assert torch.allclose(full, sp.x_A, atol=1e-5)
    assert torch.allclose(C.build("full_along_U", sp, U=torch.eye(D)), sp.x_A, atol=1e-5)
    assert torch.allclose(C.build("full_along_U", sp, U=orth(D, 0)), sp.x_B)


def test_repair_matches_pilot_hybrid_patch():
    sp = make_spans()
    U = orth(D, 3)
    pilot = sp.xh_A + sp.e_B + ((sp.e_A - sp.e_B) @ U) @ U.T       # exp3_hybrid_repair._unit
    assert torch.allclose(C.build("repair", sp, U=U), pilot, atol=1e-6)


def test_projection_idempotent_symmetric_orthonormal():
    U = orth(D, 5).double()
    P = U @ U.T
    assert torch.allclose(U.T @ U, torch.eye(5, dtype=torch.float64), atol=1e-6)     # U was cast from float32
    assert torch.allclose(P @ P, P, atol=1e-6) and torch.allclose(P, P.T)


def test_noise_norms_and_pilot_identity():
    sp = make_spans()
    eta = C.gaussian_noise(sp.e_A, torch.Generator().manual_seed(7))
    assert torch.allclose(eta.norm(dim=-1), sp.e_A.norm(dim=-1), rtol=1e-5)
    from src.experiments.common import rescaled_noise
    assert torch.equal(eta, rescaled_noise(sp.e_A, torch.Generator().manual_seed(7)))


def test_cov_noise_converges():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(D, D, generator=g) * 0.3
    E = torch.randn(4000, D, generator=g) @ A.T + 2.0
    fs = Dr.FitSet(tuple(f"x-train-{i}" for i in range(3)))
    cn = C.CovNoise.fit(fs, E, fs.pair_ids)
    smp = cn.sample(200_000, torch.Generator().manual_seed(1))
    target = torch.cov(E.T)
    assert torch.allclose(torch.cov(smp.T), target, atol=0.02 * float(target.abs().max()))
    assert smp.mean(0).abs().max() < 0.02


def test_mean_error_train_only_and_slot_alignment():
    tr = make_spans(seed=2)
    fs = Dr.FitSet(tuple(tr.pair_ids))
    me = C.MeanError.fit(fs, tr)
    keys = tr.keys()
    fin = [j for j, k in enumerate(keys) if k == "final"]
    want = torch.cat([tr.e_A[fin], tr.e_B[fin]]).double().mean(0).float()
    assert torch.allclose(me.by_key["final"], want, atol=1e-6)
    te = make_spans(seed=3, prefix="t-test-")
    rows = me.per_row(te)
    j = te.keys().index("final")
    assert torch.allclose(rows[j], me.by_key["final"]) and me.last_coverage["task"] == 0
    with pytest.raises(Dr.LeakageError):
        C.MeanError.fit(fs, te)                          # data outside the FitSet
    with pytest.raises(Dr.LeakageError):
        Dr.FitSet(tuple(te.pair_ids))                    # test ids cannot form a FitSet


def test_mean_pos_offset_fallback():
    tr = make_spans(seed=2, slots=False)
    me = C.MeanError.fit(Dr.FitSet(tuple(tr.pair_ids)), tr)
    te = make_spans(seed=3, prefix="t-test-", slots=False)
    me.per_row(te)
    assert me.last_coverage["offset"] == te.m


def test_resample_donors_constraints():
    sp = make_spans(seed=4, n=40)
    sp.meta["ans_A"] = np.arange(40) % 4
    sp.meta["ans_B"] = (np.arange(40) + 1) % 4
    for same in (True, False):
        ds = C.select_donors(sp.meta, same, seed=0)
        for i, d in enumerate(ds):
            if d["j"] < 0:
                continue
            assert d["j"] != i                                   # never A or B of the same pair
            ans = sp.meta["ans_A" if d["prompt"] == "A" else "ans_B"][d["j"]]
            assert sp.meta["first_diff"][d["j"]] <= sp.meta["first_diff"][i]
            if same:
                assert ans == sp.meta["ans_A"][i]
            elif d["status"] == "ok":
                assert ans not in (sp.meta["ans_A"][i], sp.meta["ans_B"][i])
        eC = C.donor_errors(sp, ds)
        i = next(k for k, d in enumerate(ds) if d["j"] >= 0)
        d = ds[i]
        src = sp.e_A if d["prompt"] == "A" else sp.e_B
        start = int(sp.row_ptr[d["j"]]) + int(sp.meta["first_diff"][i] - sp.meta["first_diff"][d["j"]])
        assert torch.equal(eC[sp.rows(i)], src[start:start + (sp.rows(i).stop - sp.rows(i).start)])


def test_subthreshold_addback(tmp_path):
    p = S.write_random_params(tmp_path / "p.npz", d_model=D, d_sae=300, seed=1)
    sae = S.load_params(p)
    X = torch.randn(30, D, generator=torch.Generator().manual_seed(0)) * 2
    codes = S.encode_sparse(sae, X, alpha_min=0.05)
    assert torch.equal(C.subthreshold_addback(codes, sae, 1.0), torch.zeros(30, D))
    small = C.subthreshold_addback(codes, sae, 0.05)
    m = (codes.z < codes.threshold) & (codes.z > 0)
    assert m.any() and torch.allclose(small, S.decode_entries(sae, codes, m), atol=1e-6)
    with pytest.raises(ValueError):
        C.subthreshold_addback(codes, sae, 0.01)


# ---------------------------------------------------------------- directions
def test_pilot_estimators_and_guard():
    g = torch.Generator().manual_seed(0)
    E = torch.randn(80, D, generator=g)
    y = np.arange(80) % 5
    ids = tuple(f"t-train-{i:05d}" for i in range(40))
    fs = Dr.FitSet(ids)
    from src.experiments.exp3_hybrid_repair import fit_supervised, fit_unsupervised
    assert np.allclose(Dr.pca_pooled(fs, ids, E, 8).U, fit_unsupervised(E, 8))
    a = Dr.answer_sep(fs, ids, E, y, 8)
    assert np.allclose(a.U, fit_supervised(E, y, 8)) and a.meta["centroid_dims"] == 4
    for d in (Dr.pca_pooled(fs, ids, E, 8), a, Dr.random(D, 8, 3)):
        assert np.allclose(d.U.T @ d.U, np.eye(8), atol=1e-8)
    with pytest.raises(Dr.LeakageError):
        Dr.pca_pooled(fs, ids + ("t-train-99999",), E, 8)
    with pytest.raises(Dr.LeakageError):
        Dr.FitSet(ids, split="test")
    with pytest.raises(Dr.LeakageError):
        fs.assert_disjoint(["t-train-00003"])


def test_centered_estimators_remove_key_means():
    sp = make_spans(seed=5)
    keys = sp.keys()
    shift = torch.zeros(D)
    shift[0] = 50.0
    E = sp.e_A + torch.tensor([1.0 if k == "final" else 0.0 for k in keys])[:, None] * shift
    fs = Dr.FitSet(tuple(sp.pair_ids))
    pooled = Dr.pca_pooled(fs, sp.pair_ids, E, 3)
    centred = Dr.pca_pos_centered(fs, sp.pair_ids, E, keys, 3)
    e0 = np.eye(D)[:, :1]
    assert Dr.principal_cosines(pooled.U[:, :1], e0)[0] > 0.9      # a slot mean dominates pooled PCA
    assert Dr.principal_cosines(centred.U[:, :1], e0)[0] < 0.9
    pd = Dr.pca_slot_centered_pairdiff(fs, sp.pair_ids, sp.e_A - sp.e_B, keys, 3)
    assert np.allclose(pd.U.T @ pd.U, np.eye(3), atol=1e-8)


# ---------------------------------------------------------------- metrics
def test_metrics_hand_computed():
    lA = np.array([2.0, 0.0])
    lB = np.array([0.0, 1.0])
    lse = np.log(np.exp(lA) + np.exp(lB) + 1.0)          # one extra token with logit 0
    o = M.Out(lA, lB, lse, np.array([10, 11]), np.stack([lA, lB], 1))
    assert np.allclose(o.p_A + o.p_B + 1 / np.exp(lse), 1)
    assert np.allclose(o.p_candidates(), o.p_A + o.p_B)
    assert np.allclose(o.cand_entropy()[0], -(np.log(np.exp(2) / (np.exp(2) + 1)) * np.exp(2) / (np.exp(2) + 1)
                                              + np.log(1 / (np.exp(2) + 1)) / (np.exp(2) + 1)))
    assert list(M.outcome(o, np.array([10, 5]), np.array([3, 11]))) == ["A", "B"]


def test_nldr_and_flip_rate_conventions():
    rng = np.random.default_rng(0)
    n = 50
    ansA, ansB = np.full(n, 1), np.full(n, 2)
    def out(dA):
        lA = rng.normal(5, 1, n) + dA
        lB = np.full(n, 5.0)
        top = np.where(lA > lB, 1, 2)
        return M.Out(lA, lB, np.logaddexp(lA, lB) + 0.1, top, np.stack([lA, lB], 1))
    A, B, P = out(3.0), out(-3.0), out(0.5)
    m = M.per_condition(P, A, B, ansA, ansB)
    from src.experiments.common import Baseline, summarize
    import torch as T
    base = Baseline(items=[None] * n, spans=[], dl_clean=T.tensor(A.D), dl_corrupt=T.tensor(B.D),
                    top1_clean_run=T.ones(n, dtype=T.bool), top1_corrupt_run=T.zeros(n, dtype=T.bool), resid_clean={},
                    resid_corrupt={}, clean_ans=T.tensor(ansA), corrupt_ans=T.tensor(ansB), states_clean=ansA, states_corrupt=ansB)
    pil = summarize(T.tensor(P.D), T.tensor(P.top1_id == 1), base, n_boot=1000)
    assert m["nldr"] == pytest.approx(pil["nldr"], abs=1e-9)
    assert m["nldr_pair_mean"] == pytest.approx(pil["nldr_pair_mean"], abs=1e-6)
    elig = B.top1_id == 2
    assert m["flip_rate"] == pytest.approx((P.top1_id[elig] == 1).mean()) and m["flip_n_eligible"] == elig.sum()
    # the pilot's bootstrap CI is reproduced exactly (same index matrix: default_rng(0), shape (n_boot, n))
    res = St.bootstrap(lambda idx: {"nldr": M.per_condition(P.take(idx), A.take(idx), B.take(idx), ansA[idx], ansB[idx])["nldr"]},
                       n, n_boot=1000, seed=0)
    assert res["nldr"]["lo"] == pytest.approx(pil["nldr_lo"], abs=1e-6) and res["nldr"]["hi"] == pytest.approx(pil["nldr_hi"], abs=1e-6)
    g = M.gaps(m, M.per_condition(A, A, B, ansA, ansB))
    assert g["gap_nldr"] == pytest.approx(1 - m["nldr"], abs=1e-6)
    sc = M.share_closed({"nldr": 0.9}, {"nldr": 0.7}, {"nldr": 1.0})
    assert sc["share_closed_nldr"] == pytest.approx(2 / 3)   # (0.9 - 0.7) / (1 - 0.7)


# ---------------------------------------------------------------- stats
def test_bootstrap_reproducible_and_cluster():
    x = np.random.default_rng(1).normal(size=60)
    f = lambda idx: {"m": x[idx].mean()}
    a, b = St.bootstrap(f, 60, 500, seed=3), St.bootstrap(f, 60, 500, seed=3)
    assert np.array_equal(a["m"]["samples"], b["m"]["samples"]) and a["m"]["lo"] < x.mean() < a["m"]["hi"]
    fam = np.repeat(np.arange(6), 10)
    idx = St.boot_indices(60, 50, seed=0, clusters=fam)
    for ix in idx:
        labs, cnt = np.unique(fam[ix], return_counts=True)
        assert (cnt % 10 == 0).all()                      # whole families only


def test_holm_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(St.holm(p), [0.04, 0.09, 0.09, 0.2])
    assert np.allclose(St.bh(p), [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)
    assert St.boot_pvalue(np.array([1.0, 2.0, -1.0]), 0) == pytest.approx(2 / 4)
    d = St.distribution(np.arange(20))
    assert d["n"] == 20 and d["p95"] == pytest.approx(np.percentile(np.arange(20), 95))

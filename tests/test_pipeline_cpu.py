"""CPU-only tests of alignment, decomposition, SVD subspaces and patching (mock model, no weights)."""
import numpy as np
import pytest
import torch

from src.data.synthetic_tasks import (FallbackTokenizer, TASKS, TASK_STATES, assert_token_alignment, build_task_splits,
                                      generate_pairs, parse_state, solve_dyck, solve_graph, solve_mod_sum, tokenize_pairs)
from src.experiments.common import (RunContext, cache_baseline, decompose_batched, rescaled_noise, run_patched,
                                    summarize)
from src.experiments.exp1_causal_split import build_vectors
from src.experiments.exp2_geometry import centered_svd, fit_probes, _stack
from src.experiments.exp3_hybrid_repair import fit_supervised, fit_unsupervised
from src.models.sae_loader import MockConfig, load_model, load_sae
from src.utils import get_device, get_dtype, load_config

LAYER = 12


@pytest.fixture(scope="module")
def tok():
    return FallbackTokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return load_model(mock=True, mock_cfg={"d_model": 256}, tokenizer=tok)


@pytest.fixture(scope="module")
def data(model, tok):
    out = {}
    for t in TASKS:
        sp = build_task_splits(t, tok, 64, 32, seed=0, mock=True)
        out[t] = {k: cache_baseline(model, v, [LAYER], 16) for k, v in sp.items()}
    return out


# ------------------------------------------------------------------ data / alignment
def test_alignment_assert():
    assert_token_alignment([1, 2, 3], [4, 5, 6])
    with pytest.raises(AssertionError):
        assert_token_alignment([1, 2, 3], [4, 5])


def test_fallback_tokenizer_deterministic_roundtrip(tok):
    s = "State = 0. Add 3. Add 4. Modulo 7, State ="
    assert tok.encode(s) == FallbackTokenizer().encode(s)
    assert tok.decode(tok.encode(s)) == s
    assert tok.encode(s)[0] == tok.BOS


def test_solvers_match_spec_examples():
    assert solve_mod_sum(0, [3, 4, 2, 1])[0] == 3
    assert solve_dyck(2, [1, 1, -1, 1])[0] == 4
    assert solve_graph([(1, 2), (3, 4), (2, 3)], n_nodes=4)[0] == 4


@pytest.mark.parametrize("task", TASKS)
def test_pairs_valid_deterministic_aligned(task, tok):
    a = generate_pairs(task, 50, seed=3)
    b = generate_pairs(task, 50, seed=3)
    assert [p.clean_prompt for p in a] == [p.clean_prompt for p in b]
    assert len(a) == 50
    for p in a:
        assert p.clean_state_id != p.corrupt_state_id
        assert len(tok.encode(p.clean_prompt)) == len(tok.encode(p.corrupt_prompt))
        assert parse_state(p.clean_prompt) == (task, p.clean_state_id)
        assert parse_state(p.corrupt_prompt) == (task, p.corrupt_state_id)
        assert p.clean_state_id in TASK_STATES[task]
    assert len(tokenize_pairs(tok, a)) == 50  # all answers are single tokens


def test_dyck_depth_stays_bounded():
    for p in generate_pairs("dyck_depth", 100, seed=1):
        assert all(1 <= s <= 6 for s in p.step_states_clean)


@pytest.mark.parametrize("task", TASKS)
def test_train_test_disjoint(task, tok):
    sp = build_task_splits(task, tok, 100, 50, seed=0, mock=True)
    tr = {p.pair.clean_prompt for p in sp["train"]} | {p.pair.corrupt_prompt for p in sp["train"]}
    te = {p.pair.clean_prompt for p in sp["test"]} | {p.pair.corrupt_prompt for p in sp["test"]}
    assert not (tr & te)


# ------------------------------------------------------------------ models
def test_device_dtype():
    assert get_device().type in ("cuda", "mps", "cpu")
    assert get_dtype(torch.device("cpu")) == torch.float32
    assert get_dtype(torch.device("cuda")) == torch.bfloat16


@pytest.mark.parametrize("width", ["16k", "65k"])
def test_decomposition_identity(model, width):
    sae = load_sae(LAYER, width, model=model, mock=True)
    x = torch.randn(5, 7, 256)
    xh, e, acts = sae.decompose(x)
    assert xh.dtype == e.dtype == torch.float32
    assert torch.allclose(xh + e, x, atol=1e-5)
    assert ((acts > 0).sum(-1) <= sae.k).all()


def test_mock_sae_blind_to_dark_subspace(model):
    sae = load_sae(LAYER, "16k", model=model, mock=True)
    D = model.world.D
    x = torch.randn(10, 256)
    xh, _, _ = sae.decompose(x)
    assert (xh @ D).abs().max() < 1e-5


def test_model_runs_and_hooks(model, tok):
    ids = torch.tensor([tok.encode("State = 0. Add 3. Add 4. Add 2. Add 1. Modulo 7, State =")] * 2)
    assert model(ids).shape == (2, ids.shape[1], tok.vocab_size)
    with pytest.raises(ZeroDivisionError):
        model.run_with_hooks(ids, [("blocks.3.hook_resid_post", lambda a, h: 1 / 0)])


# ------------------------------------------------------------------ patching
@pytest.mark.parametrize("task", TASKS)
def test_baseline_clean_beats_corrupt(data, task):
    b = data[task]["test"]
    assert (b.dl_clean > 0).all() and (b.dl_corrupt < 0).all()


def test_nldr_endpoints(model, data):
    b = data["mod_sum"]["test"]
    full_clean = run_patched(model, b, LAYER, b.resid_clean[LAYER], None)
    assert summarize(*full_clean, b)["nldr"] == pytest.approx(1.0, abs=0.15)  # later layers add corrupt signal
    noop = run_patched(model, b, 25, b.resid_corrupt[LAYER], None) if False else run_patched(model, b, LAYER, b.resid_corrupt[LAYER], None)
    assert summarize(*noop, b)["nldr"] == pytest.approx(0.0, abs=1e-4)


def test_causal_split_signs(model, data):
    sae = load_sae(LAYER, "16k", model=model, mock=True)
    res = {}
    for task in ("mod_sum", "ioi_control"):
        b = data[task]["test"]
        xh_c, e_c, _ = decompose_batched(sae, b.resid_clean[LAYER])
        xh_k, e_k, _ = decompose_batched(sae, b.resid_corrupt[LAYER])
        gen = torch.Generator().manual_seed(0)
        for name, v in build_vectors(xh_c, e_c, xh_k, e_k, rescaled_noise(e_c, gen)).items():
            res[task, name] = summarize(*run_patched(model, b, LAYER, v), b)["nldr"]
    assert res["mod_sum", "error_only"] > 0.6 > 0.2 > res["mod_sum", "sae_only"]
    assert res["ioi_control", "sae_only"] > 0.6 > 0.3 > res["ioi_control", "error_only"]


def test_noise_norm_matching():
    e = torch.randn(20, 64) * torch.rand(20, 1) * 3
    eta = rescaled_noise(e, torch.Generator().manual_seed(1))
    assert torch.allclose(eta.norm(dim=-1), e.norm(dim=-1), rtol=1e-4)


# ------------------------------------------------------------------ SVD / hybrid
def test_subspaces_orthonormal_nested_and_train_only(model, data):
    sae = load_sae(LAYER, "16k", model=model, mock=True)
    Xtr, ytr = _stack(data["mod_sum"]["train"], LAYER)
    _, E, _ = decompose_batched(sae, Xtr)
    for U in (fit_unsupervised(E, 16), fit_supervised(E, ytr, 16)):
        assert U.shape == (256, 16)
        assert np.allclose(U.T @ U, np.eye(16), atol=1e-5)
    # changing held-out data cannot change U (it is a pure function of train errors)
    assert np.allclose(fit_unsupervised(E, 8), fit_unsupervised(E.clone(), 8))


def test_svd_concentrates_on_mock_manifold(model, data):
    sae = load_sae(LAYER, "16k", model=model, mock=True)
    Xtr, _ = _stack(data["mod_sum"]["train"], LAYER)
    _, E, _ = decompose_batched(sae, Xtr)
    S, Vt = centered_svd(E)
    assert (S[:2] ** 2).sum() / (S ** 2).sum() > 0.35 and S[1] > 2 * S[2]   # clear 2-D spectral gap
    # the top-2 subspace should lie inside the mod_sum dark plane
    plane = model.world.D[:, :2].numpy()
    assert np.linalg.norm(plane.T @ Vt[:2].T) ** 2 / 2 > 0.9


def test_hybrid_rank0_equals_sae_only_and_rank_helps(model, data):
    sae = load_sae(LAYER, "16k", model=model, mock=True)
    b, tr = data["mod_sum"]["test"], data["mod_sum"]["train"]
    Xtr, ytr = _stack(tr, LAYER)
    _, E, _ = decompose_batched(sae, Xtr)
    xh_c, e_c, _ = decompose_batched(sae, b.resid_clean[LAYER])
    xh_k, e_k, _ = decompose_batched(sae, b.resid_corrupt[LAYER])
    out = {}
    for r in (0, 4):
        U = torch.from_numpy(fit_supervised(E, ytr, 4)[:, :r]).float()
        v = xh_c + e_k + ((e_c - e_k) @ U) @ U.T
        out[r] = summarize(*run_patched(model, b, LAYER, v), b)["nldr"]
    base = summarize(*run_patched(model, b, LAYER, xh_c + e_k), b)["nldr"]
    assert out[0] == pytest.approx(base, abs=1e-6)
    assert out[4] > out[0] + 0.5


def test_probes_train_test_separation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 200)
    X = rng.normal(size=(200, 32))
    X[:, 0] = np.cos(2 * np.pi * y / 7) * 5
    X[:, 1] = np.sin(2 * np.pi * y / 7) * 5
    r = fit_probes(X[:140], y[:140], X[140:], y[140:], 7)
    assert r["r2_circular"] > 0.9
    assert r["acc"] > 0.4  # chance is 1/7; a linear readout of a 7-point ring is imperfect by construction


def test_config_loads():
    cfg = load_config()
    assert cfg["layers"] == [4, 8, 12, 16, 20, 24] and cfg["widths"] == ["16k", "65k"]


# ----------------------------------------------------------------------------- v2 features
def test_from_diff_full_clean_patch_and_rank0(model, data):
    from src.experiments.common import summarize
    b = data["mod_sum"]["test"]
    xc = b.acts(12, "clean", "from_diff")
    assert xc.shape[0] == int(b.row_ptr[-1]) and torch.allclose(xc[b.row_ptr[1:] - 1], b.resid_clean[12], atol=1e-5)
    dl, top = run_patched(model, b, 12, xc, positions="from_diff")
    dl2, _ = run_patched(model, b, 12, b.resid_clean[12], positions="last")
    assert torch.allclose(dl, dl2, atol=1e-4)
    s = summarize(dl, top, b)
    assert s["nldr_lo"] <= s["nldr"] <= s["nldr_hi"]


def test_bootstrap_ci_contains_point(data):
    from src.experiments.common import summarize
    b = data["dyck_depth"]["test"]
    noisy = b.dl_corrupt + 0.5 * (b.dl_clean - b.dl_corrupt) + torch.randn(len(b))
    s = summarize(noisy, b.top1_clean_run, b, n_boot=500)
    assert s["nldr_lo"] < s["nldr"] < s["nldr_hi"]


def test_resample_partners():
    from src.experiments.exp1_causal_split import resample_partners
    st = np.array([0, 0, 1, 1, 2])
    same, diff = resample_partners(st, True, 0), resample_partners(st, False, 0)
    assert all(st[same[i]] == st[i] and (same[i] != i or i == 4) for i in range(5))
    assert all(st[diff[i]] != st[i] for i in range(5))


def test_random_subspace_is_null_in_mock(model, data):
    from src.experiments.exp3_hybrid_repair import random_subspace
    from src.experiments.common import summarize
    b = data["mod_sum"]["test"]
    sae = load_sae(12, "16k", model=model, mock=True)
    xh_c, e_c, _ = decompose_batched(sae, b.resid_clean[12])
    xh_k, e_k, _ = decompose_batched(sae, b.resid_corrupt[12])
    U = torch.from_numpy(random_subspace(e_c.shape[1], 4, 0)).float()
    dl, top = run_patched(model, b, 12, xh_c + e_k + ((e_c - e_k) @ U) @ U.T)
    assert abs(summarize(dl, top, b)["nldr"]) < 0.15


def test_valid_mask_modes():
    from src.data.synthetic_tasks import valid_mask
    sc = {"dl_clean": np.array([3.0, 3.0, 1.0]), "dl_corrupt": np.array([-3.0, 2.0, -5.0]),
          "clean_top1": np.array([True, True, True]), "corrupt_top1": np.array([True, False, True])}
    assert valid_mask(sc, "strict").tolist() == [True, False, False]
    assert valid_mask(sc, "clean_only").tolist() == [True, False, True]


def test_answer_format_space_prompt(tok):
    from src.data.synthetic_tasks import generate_pairs, tokenize_pairs
    p = generate_pairs("mod_sum", 4, 0)
    it = tokenize_pairs(tok, p, "space_prompt")
    assert len(it) == 4 and it[0].pair.clean_prompt.endswith("= ") and not it[0].pair.clean_target_str.startswith(" ")


def test_unit_log_resume_and_failure_isolation(tmp_path, model, tok):
    from src.experiments.common import UnitLog
    ctx = RunContext(model=model, tok=tok, cfg=load_config(), mock=True, device="cpu", out_dir=tmp_path)
    u = ctx.unit_log("x")
    assert ctx.run_unit("x", "a", lambda: [{"v": 1}], u)
    assert not ctx.run_unit("x", "b", lambda: 1 / 0, u)
    assert (tmp_path / "metrics" / "failures.jsonl").exists()
    u2 = UnitLog(tmp_path / "metrics" / "x_units.jsonl", resume=True)
    assert u2.done("a") and not u2.done("b")
    calls = []
    ctx.resume = True
    ctx.run_unit("x", "a", lambda: calls.append(1) or [], u2)
    assert not calls and u2.rows() == [{"v": 1}]

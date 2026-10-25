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
from src.models.sae_loader import _PLANES, MockConfig, load_model, load_sae
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
    xh, e, l0 = sae.decompose(x)
    assert xh.dtype == e.dtype == torch.float32
    assert torch.allclose(xh + e, x, atol=1e-5)
    assert l0.shape == (5, 7) and (l0 <= sae.k).all()


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
    i, j = _PLANES["mod_sum"]
    plane = model.world.D[:, [i, j]].numpy()
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


def test_few_shot_prefix_shared_and_leak_free(tok):
    s = build_task_splits("mod_sum", tok, 60, 60, 0, n_shots=3)
    pre = lambda it: it.pair.clean_prompt.rsplit("\n", 1)[0]
    prefixes = {pre(it) for sp in s.values() for it in sp}
    assert len(prefixes) == 1
    shot_prompts = {l.rsplit("=", 1)[0] + "=" for l in next(iter(prefixes)).split("\n")}
    queries = {it.pair.clean_prompt.rsplit("\n", 1)[1] for sp in s.values() for it in sp}
    queries |= {it.pair.corrupt_prompt.rsplit("\n", 1)[1] for sp in s.values() for it in sp}
    assert not (shot_prompts & queries)


def test_batches_respect_token_budget(tok):
    from src.data.synthetic_tasks import batch_by_length, generate_pairs, tokenize_pairs
    items = tokenize_pairs(tok, generate_pairs("mod_sum", 64, 0))
    L = len(items[0].clean_ids)
    for b in batch_by_length(items, 32, max_tokens=4 * L):
        assert len(b) <= 4 and len({len(i.clean_ids) for i in b}) == 1


def test_real_sae_jumprelu_encode():
    import types
    from src.models.sae_loader import RealSAE
    d, n = 8, 16
    fake = torch.nn.Module()
    fake.W_enc, fake.b_enc = torch.nn.Parameter(torch.randn(d, n)), torch.nn.Parameter(torch.zeros(n))
    fake.W_dec, fake.b_dec = torch.nn.Parameter(torch.randn(n, d)), torch.nn.Parameter(torch.zeros(d))
    fake.threshold = torch.nn.Parameter(torch.full((n,), 0.5))
    sae = RealSAE(fake, "16k", 0)
    x = torch.randn(5, d)
    pre = x @ fake.W_enc
    a = sae.encode(x)
    assert torch.all((a > 0) == (pre > 0.5)) and torch.allclose(a[a > 0], pre[pre > 0.5])
    xh, e, _ = sae.decompose(x)
    assert torch.allclose(xh + e, x, atol=1e-6)


# ------------------------------------------------------------------ memory-lean runtime (8 GB GPUs)
def test_last_logits_and_nll_match_full_forward(model, tok):
    from src.models import runtime
    ids = torch.tensor([tok.encode("State = 0. Add 3. Add 4. Add 2. Add 1. Modulo 7, State =")] * 3)
    full = model.run_with_hooks(ids, fwd_hooks=[])
    assert torch.allclose(runtime.last_logits(model, ids), full[:, -1].float(), atol=1e-5)
    lp = torch.log_softmax(full[:, 2:-1].float(), -1)
    ref = -float(lp.gather(-1, ids[:, 3:, None]).sum())
    assert runtime.token_nll_sum(model, ids, skip=2) == pytest.approx(ref, rel=1e-5)


@pytest.mark.parametrize("positions", ["last", "from_diff"])
def test_resume_from_layer_matches_hooked_forward(model, data, positions):
    b = data["mod_sum"]["test"]
    sae = load_sae(LAYER, "16k", model=model, mock=True)
    xh_c, _, _ = decompose_batched(sae, b.acts(LAYER, "clean", positions))
    _, e_k, _ = decompose_batched(sae, b.acts(LAYER, "corrupt", positions))
    v = xh_c + e_k
    dl_a, top_a = run_patched(model, b, LAYER, v, None, positions=positions, resume=False)
    dl_b, top_b = run_patched(model, b, LAYER, v, None, positions=positions, resume=True)
    assert torch.allclose(dl_a, dl_b, atol=1e-4) and torch.equal(top_a, top_b)
    # the cached corrupt residual must not be modified by patching
    dl_c, _ = run_patched(model, b, LAYER, b.resid_corrupt[LAYER], None, resume=True)
    assert torch.allclose(dl_c, b.dl_corrupt, atol=1e-4)


def _toy_jumprelu(d=16, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc, W_dec = torch.randn(d, n, generator=g), torch.randn(n, d, generator=g) / 4
    b_enc, b_dec, thr = torch.randn(n, generator=g) / 4, torch.randn(d, generator=g), torch.rand(n, generator=g)
    return W_enc, b_enc, thr, W_dec, b_dec


def test_jumprelu_streamed_equals_resident_and_reference():
    from src.models.sae_loader import JumpReLUSAE
    W_enc, b_enc, thr, W_dec, b_dec = _toy_jumprelu()
    x = torch.randn(33, 16)
    pre = x @ W_enc + b_enc
    f = torch.relu(pre) * (pre > thr)
    ref = f @ W_dec + b_dec
    res = JumpReLUSAE(W_enc.T.contiguous(), b_enc, thr, W_dec, b_dec, "16k", 0)
    st = JumpReLUSAE(W_enc.T.contiguous(), b_enc, thr, W_dec, b_dec, "16k", 0).stream_to("cpu", chunk_latents=7, pin=False)
    for sae in (res, st):
        xh, e, l0 = sae.decompose(x)
        assert torch.allclose(xh, ref, atol=1e-5) and torch.allclose(xh + e, x, atol=1e-6)
        assert torch.equal(l0, (f > 0).sum(-1))
        assert torch.allclose(sae.encode(x), f, atol=1e-6)
        assert torch.allclose(sae.decode(f), ref, atol=1e-5)
    assert st.streamed and len(st._host) == 6


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma_scope_npz_loader(tmp_path, dtype):
    from src.models.sae_loader import JumpReLUSAE, load_gemma_scope_npz
    W_enc, b_enc, thr, W_dec, b_dec = _toy_jumprelu()
    path = tmp_path / "params.npz"
    np.savez(path, W_enc=W_enc.numpy(), b_enc=b_enc.numpy(), threshold=thr.numpy(), W_dec=W_dec.numpy(), b_dec=b_dec.numpy())
    sae = load_gemma_scope_npz(str(path), "16k", 3, dtype)
    assert sae.dtype == dtype and sae.d_sae == 40 and sae.d_model == 16
    assert torch.equal(sae.W_enc_T, W_enc.T.to(dtype)) and torch.equal(sae.W_dec, W_dec.to(dtype))
    ref = JumpReLUSAE(W_enc.T.contiguous().to(dtype), b_enc.to(dtype), thr.to(dtype), W_dec.to(dtype), b_dec, "16k", 3)
    x = torch.randn(9, 16)
    assert torch.equal(sae.decompose(x)[0], ref.decompose(x)[0])


def test_token_budget_is_runtime_configurable(tok):
    from src.data import synthetic_tasks as st
    items = tokenize_pairs(tok, generate_pairs("mod_sum", 64, 0))
    L = len(items[0].clean_ids)
    old = st.MAX_BATCH_TOKENS
    try:
        st.MAX_BATCH_TOKENS = 3 * L
        assert max(len(b) for b in st.batch_by_length(items, 128)) == 3
    finally:
        st.MAX_BATCH_TOKENS = old


# ------------------------------------------------------------------ state-tracking suite (2026-10-20 task analysis)
def _independent_var(program: str) -> int:
    env = {}
    for stmt in program.split(".")[:-1]:
        k, v = (t.strip() for t in stmt.split("="))
        env[k] = int(v) if v.isdigit() else env[v]
    return env[program.split(".")[-1].split("=")[0].strip()]


@pytest.mark.parametrize("task", ["add_chain", "entity_track", "var_binding"])
def test_state_task_answers_and_alignment(task, tok):
    import re
    pairs = generate_pairs(task, 300, seed=3)
    for p in pairs:
        assert p.clean_target_str != p.corrupt_target_str
        for prompt, tgt, state in ((p.clean_prompt, p.clean_target_str, p.clean_state_id),
                                   (p.corrupt_prompt, p.corrupt_target_str, p.corrupt_state_id)):
            assert parse_state(prompt) == (task, state)
            if task == "add_chain":   # independent re-derivation from the text
                nums = [int(x) for x in re.findall(r"(\d)", prompt)]
                assert int(tgt) == sum(nums)
            elif task == "var_binding":
                assert int(tgt) == _independent_var(prompt)
    assert len(tokenize_pairs(tok, pairs)) == len(pairs)          # token-aligned, single-token answers


def test_prompt_spaces_fill_the_tier_splits():
    from src.data import synthetic_tasks as st
    for task, kw in (("add_chain", {"depth": 3}), ("entity_track", {"depth": 2}), ("var_binding", {"depth": 1}),
                     ("ioi_control", None)):
        assert len(st.generate_pairs(task, 2500, 0, gen_kwargs=kw)) == 2500, task
    # the validated wording alone still fills a tier (~9.4k pairs) if the 'story' template must be turned off
    assert len(st.generate_pairs("add_chain", 3000, 0, gen_kwargs={"templates": ["calc"]})) == 3000


def test_few_shot_prefix_is_never_starved(tok):
    # dyck_depth with 2 ops has only 12 pairs: shots are drawn first, so they always come out complete
    s = build_task_splits("dyck_depth", tok, 4, 2, 0, n_shots=2, gen_kwargs={"n_ops": 2})
    p = s["train"][0].pair
    assert p.meta["n_shots"] == 2 and p.clean_prompt.count("\n") == 2
    shots = set(p.clean_prompt.split("\n")[:2])
    for split in s.values():
        for it in split:
            assert it.pair.clean_prompt.split("\n")[-1] + it.pair.clean_target_str not in shots


def test_fallback_tokenizer_keeps_newlines_for_few_shot(tok):
    from src.data.synthetic_tasks import generate_with_shots
    p = generate_with_shots("ioi_control", 4, 2, 0, 77)[0]
    decoded = tok.decode(tok.encode(p.clean_prompt))
    assert decoded == p.clean_prompt and parse_state(decoded) == ("ioi_control", p.clean_state_id)


def test_task_override_merge_keeps_gen_kwargs():
    from src.experiments.common import merge_task_overrides
    cfg = {"add_chain": {"gen_kwargs": {"depth": 3}, "n_shots": 2}}
    pre = {"add_chain": {"n_shots": 4, "filter_mode": "strict"}, "var_binding": {"n_shots": 8}}
    m = merge_task_overrides(cfg, pre)
    assert m["add_chain"] == {"gen_kwargs": {"depth": 3}, "n_shots": 4, "filter_mode": "strict"}
    assert m["var_binding"] == {"n_shots": 8} and cfg["add_chain"]["n_shots"] == 2


def test_preflight_choice_and_fingerprint():
    import importlib
    import sys
    sys.path.insert(0, "scripts")
    pf = importlib.import_module("preflight")
    rows = [dict(n_shots=0, clean_only=0.50, strict=0.10), dict(n_shots=2, clean_only=0.99, strict=0.90),
            dict(n_shots=8, clean_only=1.00, strict=0.95)]
    best, mode, ret = pf.choose(rows, 0.3)
    assert best["n_shots"] == 2 and mode == "strict" and ret == 0.90     # 0.99 ~ 1.00 -> fewer shots
    best, mode, _ = pf.choose([dict(n_shots=4, clean_only=0.6, strict=0.1)], 0.3)
    assert mode == "clean_only"
    cfg = load_config(None)
    a = pf.fingerprint(cfg, mock=True)
    assert a == pf.fingerprint(load_config(None), mock=True) != pf.fingerprint(cfg, mock=False)
    cfg["task_overrides"]["add_chain"]["gen_kwargs"] = {"depth": 2}
    assert pf.fingerprint(cfg, mock=True) != a


def test_default_config_uses_the_validated_suite():
    from src.data.synthetic_tasks import LEGACY_TASKS
    cfg = load_config(None)
    assert set(cfg["tasks"]) == {"add_chain", "entity_track", "var_binding", "ioi_control"}
    assert not set(cfg["tasks"]) & set(LEGACY_TASKS)
    assert all(t in cfg["task_overrides"] for t in cfg["tasks"])


def test_surface_token_baseline(data):
    from src.experiments.exp2_geometry import surface_features
    tr, te = data["add_chain"]["train"].items, data["add_chain"]["test"].items
    Xtr, Xte = surface_features(tr, te)
    assert Xtr.shape[0] == 2 * len(tr) and Xte.shape == (2 * len(te), Xtr.shape[1])
    ytr = np.concatenate([data["add_chain"]["train"].states_clean, data["add_chain"]["train"].states_corrupt])
    yte = np.concatenate([data["add_chain"]["test"].states_clean, data["add_chain"]["test"].states_corrupt])
    # the running sum is linear in token counts: the prompt-token probe gets a high ordinal R^2 with no model at all
    assert fit_probes(Xtr, ytr, Xte, yte, None)["r2_ordinal"] > 0.8
    # entity_track needs binding (which object moved where): the prompt-token probe stays far from solving it
    b_tr, b_te = data["entity_track"]["train"], data["entity_track"]["test"]
    Xe, Xet = surface_features(b_tr.items, b_te.items)
    ye = np.concatenate([b_tr.states_clean, b_tr.states_corrupt])
    yet = np.concatenate([b_te.states_clean, b_te.states_corrupt])
    assert fit_probes(Xe, ye, Xet, yet, None)["acc"] < 0.6


def test_pilot_config_is_small_and_complete():
    pilot, full = load_config("configs/pilot.yaml"), load_config("configs/tier1_core.yaml")
    assert pilot["tasks"] == full["tasks"] and pilot["task_overrides"] == full["task_overrides"]
    assert pilot["widths"] == ["16k"] and len(pilot["layers"]) <= 3 and 12 in pilot["geometry_layers"]
    assert pilot["n_train"] + pilot["n_test"] < full["n_train"] + full["n_test"]
    assert "controls" in pilot["exp1_extras"] and "random_subspace" in pilot["exp3_methods"]


@pytest.mark.parametrize("facts,expected", [
    (dict(blockers=["IOI gate failed"], h1={"add_chain": ("supported", 12, 0.9, 0.1)}, ioi_ok=True), "BLOCKED"),
    (dict(blockers=[], h1={"add_chain": ("supported", 12, 0.9, 0.1)}, ioi_ok=True), "GO"),
    (dict(blockers=[], h1={"add_chain": ("supported", 12, 0.9, 0.1)}, ioi_ok=False), "MAYBE"),   # SAE lossy everywhere
    (dict(blockers=[], h1={"add_chain": ("weak", 12, 0.4, 0.3)}, ioi_ok=True), "MAYBE"),
    (dict(blockers=[], h1={"add_chain": ("rejected", 12, 0.1, 0.8)}, ioi_ok=True), "NO-GO"),
    # SAE carries most of the state but a rank-1 error direction closes half of a 0.25 gap: refine, not reject
    (dict(blockers=[], h1={"entity_track": ("rejected", 12, 0.61, 0.75)}, ioi_ok=True,
          h3={"entity_track": (1, 12, 0.25, 0.51)}), "REFINE"),
    (dict(blockers=[], h1={"entity_track": ("rejected", 12, 0.61, 0.75)}, ioi_ok=True,
          h3={"entity_track": (32, 12, 0.25, 0.51)}), "NO-GO"),                # high-rank residue: not a finding
    (dict(blockers=[], h1={}, ioi_ok=None), "BLOCKED"),
])
def test_decide_viability_verdict(facts, expected):
    import importlib
    import sys
    sys.path.insert(0, "scripts")
    decide = importlib.import_module("decide")
    decide.FACTS.clear()
    decide.FACTS.update({"h2": {}, "h3": {}, **facts})
    assert f"**{expected}**" in decide.verdict(mock=False)[1]

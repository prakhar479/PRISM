"""Breadth pilot (docs/PILOT_BREADTH_PLAN.md): dictionaries and their formats, add-ons, code-level write-back, new estimators and
conditions, DAS v2, the counterfactual families, generated configs and the queue. Tiny / synthetic data only."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from harness import conditions as C
from harness import dicts as Dct
from harness.config import ExpConfig, das_rank, das_spec, is_estimator, parse_das, parse_das2
from harness.sae import decompose, encode_sparse

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- grammar
def test_das2_grammar_and_v1_unchanged():
    assert parse_das("das4") == {"k": 4, "epochs": 3, "lr": 1e-2, "seed": None}
    s = parse_das2("das8v2_kl_s1")
    assert s["k"] == 8 and s["objective"] == "kl" and s["seed"] == 1 and s["version"] == 2 and s["epochs"] == 10
    assert parse_das2("das4v2_ld_e3_lr3e-2")["lr"] == pytest.approx(0.03)
    assert parse_das("das8v2_kl") is None and parse_das2("das8") is None
    for bad in ("das0v2_kl", "das4v2_xx", "das4v2"):
        assert das_spec(bad) is None
    assert das_rank("das16v2_ce_s2") == 16 and das_spec("das4")["version"] == 1
    assert is_estimator("das8v2_kl") and is_estimator("grad_err") and is_estimator("fourier_T10")


# --------------------------------------------------------------------------- dictionaries
def _dense_ref(d, X):
    z = X @ d.W_enc_T.T
    if d.kind == "topk":
        r = torch.relu(z + d.b_enc)
        v, i = r.topk(d.k, -1)
        f = torch.zeros_like(r).scatter(-1, i, v)
    elif d.kind == "gated":
        f = (z + d.b_enc > 0) * torch.relu(d.mag_scale * z + d.b_mag)
    else:
        zz = z + d.b_enc
        f = zz * (zz > d.threshold)
    return f @ d.W_dec + d.b_dec, f


@pytest.mark.parametrize("kind", ["jumprelu", "topk", "gated"])
def test_dictionary_kinds_match_dense_reference_and_codes(kind):
    d = Dct.random_dict(16, 64, kind, seed=3, k=5, thr=0.3)
    X = torch.randn(40, 16)
    xh, eps, l0 = decompose(d, X)
    xr, fr = _dense_ref(d, X)
    assert torch.allclose(xh, xr, atol=1e-5) and torch.allclose(eps, X - xr, atol=1e-5)
    assert torch.equal(l0, (fr > 0).sum(-1))
    c = encode_sparse(d, X, alpha_min=1.0)                                  # active codes (custom kinds) or near-threshold
    from harness.model.stages import _active
    a = _active(d, X, None, "cpu")
    dec = Dct.Dictionary.__mro__[1]                                         # JumpReLUSAE
    rows = torch.zeros(40, 16).index_add_(0, a.row, a.z[:, None] * d.W_dec[a.col])
    assert torch.allclose(rows + d.b_dec, xh, atol=1e-5)
    assert c.n_rows == 40 and dec is not None


def _synthetic_state(variant, d=12, S=20, g=None):
    g = g or torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g)
    if variant in ("standard", "p_anneal"):
        return {"bias": r(d), "encoder.weight": r(S, d), "encoder.bias": r(S), "decoder.weight": r(d, S)}
    if variant in ("topk", "batchtopk"):
        return {"b_dec": r(d), "encoder.weight": r(S, d), "encoder.bias": r(S), "decoder.weight": r(d, S), "k": torch.tensor(4),
                "threshold": torch.tensor(0.7)}
    if variant == "matryoshka":
        return {"W_enc": r(d, S), "b_enc": r(S), "W_dec": r(S, d), "b_dec": r(d), "k": torch.tensor(4), "threshold": torch.tensor(0.7),
                "group_sizes": torch.tensor([10, 10])}
    if variant == "jumprelu":
        return {"W_enc": r(d, S), "b_enc": r(S), "W_dec": r(S, d), "b_dec": r(d), "threshold": r(S).abs() * 0.5 - 0.1}
    return {"decoder_bias": r(d), "encoder.weight": r(S, d), "r_mag": r(S) * 0.1, "gate_bias": r(S), "mag_bias": r(S),
            "decoder.weight": r(d, S)}


def _ref_formula(sd, variant, x):
    if variant in ("standard", "p_anneal"):
        f = torch.relu((x - sd["bias"]) @ sd["encoder.weight"].T + sd["encoder.bias"])
        return f @ sd["decoder.weight"].T + sd["bias"]
    if variant == "topk":
        r = torch.relu((x - sd["b_dec"]) @ sd["encoder.weight"].T + sd["encoder.bias"])
        v, i = r.topk(int(sd["k"]), -1)
        return torch.zeros_like(r).scatter(-1, i, v) @ sd["decoder.weight"].T + sd["b_dec"]
    if variant == "batchtopk":
        r = torch.relu((x - sd["b_dec"]) @ sd["encoder.weight"].T + sd["encoder.bias"])
        return (r * (r > sd["threshold"])) @ sd["decoder.weight"].T + sd["b_dec"]
    if variant == "matryoshka":
        r = torch.relu((x - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"])
        return (r * (r > sd["threshold"])) @ sd["W_dec"] + sd["b_dec"]
    if variant == "jumprelu":
        p = x @ sd["W_enc"] + sd["b_enc"]
        return (torch.relu(p) * (p > sd["threshold"])) @ sd["W_dec"] + sd["b_dec"]
    h = (x - sd["decoder_bias"]) @ sd["encoder.weight"].T
    f = (h + sd["gate_bias"] > 0) * torch.relu(torch.exp(sd["r_mag"]) * h + sd["mag_bias"])
    return f @ sd["decoder.weight"].T + sd["decoder_bias"]


@pytest.mark.parametrize("variant", ["standard", "p_anneal", "topk", "batchtopk", "matryoshka", "jumprelu", "gated"])
def test_dictionary_learning_formats_fold_exactly(tmp_path, variant):
    from harness.dict_formats import from_dictionary_learning
    sd = _synthetic_state(variant)
    p = tmp_path / "ae.pt"
    torch.save(sd, p)
    d = from_dictionary_learning([p], variant, 12, "16k", "k")
    x = torch.randn(30, 12) * 2
    xh, _, _ = decompose(d, x)
    assert torch.allclose(xh, _ref_formula(sd, variant, x), atol=1e-4)


def test_gemma_scope_2_format(tmp_path):
    from safetensors.torch import save_file
    from harness.dict_formats import from_gemma_scope_2
    g = torch.Generator().manual_seed(0)
    t = {"w_enc": torch.randn(10, 30, generator=g), "w_dec": torch.randn(30, 10, generator=g), "b_enc": torch.randn(30, generator=g),
         "b_dec": torch.randn(10, generator=g), "threshold": torch.rand(30, generator=g)}
    save_file(t, str(tmp_path / "params.safetensors"))
    d = from_gemma_scope_2([tmp_path / "params.safetensors"], 13, "16k", "k")
    x = torch.randn(25, 10)
    z = x @ t["w_enc"] + t["b_enc"]
    assert torch.allclose(decompose(d, x)[0], (z * (z > t["threshold"])) @ t["w_dec"] + t["b_dec"], atol=1e-5)


def test_npz_roundtrip_all_kinds(tmp_path):
    for kind in ("jumprelu", "topk", "gated"):
        d = Dct.random_dict(8, 32, kind, seed=1, k=3)
        p = Dct.save_npz_dict(tmp_path / f"{kind}.npz", d)
        e = Dct.load_npz_dict(p, 1, "16k")
        x = torch.randn(9, 8)
        assert e.kind == kind and torch.allclose(decompose(d, x)[0], decompose(e, x)[0], atol=1e-6)


def test_dict_ids_keep_gemma_scope_keys_and_resolve():
    from harness.config import SAESpec
    from harness.sae import SAEId
    inv = json.loads((ROOT / "configs" / "sae_inventory.json").read_text())
    did = Dct.resolve(12, SAESpec(width="16k"), inv["inventory"], inv["revision"], {})
    old = SAEId(12, "16k", 82, revision=inv["revision"])
    assert did.key == old.key and did.to_dict() == old.to_dict() and did.path_in_repo == old.path_in_repo
    add = Dct.resolve(12, SAESpec(width="16k", addon="pcx24"), inv["inventory"], inv["revision"], {})
    assert add.key == old.key + "__pcx24" and add.base().key == old.key and add.to_dict()["addon"] == "pcx24"
    reg = {"x": {"layer": 12, "width": "4k", "l0": 40, "repo": "r/r", "revision": "abc", "arch": "dl_topk", "files": ["a/ae.pt"]}}
    rd = Dct.resolve(12, SAESpec(dict="x"), inv["inventory"], inv["revision"], reg)
    assert rd.key == "L12_x" and rd.repo_files() == (("r/r", "a/ae.pt", "abc"),)
    with pytest.raises(ValueError):
        Dct.resolve(18, SAESpec(dict="x"), inv["inventory"], inv["revision"], reg)        # layer mismatch
    with pytest.raises(ValueError):
        Dct.resolve(12, SAESpec(width="16k", addon="foo"), inv["inventory"], inv["revision"], {})
    t = Dct.resolve(12, SAESpec(dict="trained:klft_L12_16k", width="16k"), inv["inventory"], inv["revision"], {})
    assert t.source == "trained" and t.repo_files() == ()


def test_registry_entries_are_well_formed():
    reg = Dct.load_registry()
    assert reg, "configs/dict_registry.yaml is empty"
    for name, e in reg.items():
        assert {"group", "layer", "width", "arch", "repo", "revision", "files"} <= set(e), name
        assert len(e["revision"]) == 40 and e["files"], name
        assert e["arch"] in ("gs2",) or e["arch"][3:] in ("standard", "p_anneal", "topk", "batchtopk", "matryoshka", "jumprelu", "gated")


# --------------------------------------------------------------------------- add-ons
def test_addon_lin_and_pcx():
    torch.manual_seed(0)
    base = Dct.random_dict(16, 64, "jumprelu", seed=2, thr=0.5)
    X = torch.randn(4000, 16) @ torch.randn(16, 16) + 3.0
    fit = Dct.fit_addon(base, iter(X.split(1000)), "lin")
    ad = Dct.AddonDict(base, "lin", fit["W"], fit["c"])
    e0 = decompose(base, X)[1]
    e1 = decompose(ad, X)[1]
    assert (e1 ** 2).sum() < 0.5 * (e0 ** 2).sum()                        # the linear part of eps is removed
    assert torch.allclose(decompose(ad, X)[0] + e1, X, atol=1e-4)
    fp = Dct.fit_addon(base, iter(X.split(1000)), "pcx3")
    P = fp["W"]
    assert torch.allclose(P @ P, P, atol=1e-4) and abs(float(torch.trace(P)) - 3) < 1e-3
    ap = Dct.AddonDict(base, "pcx3", P)
    xh, ep, _ = decompose(ap, X)
    assert torch.allclose(ep, e0 - e0 @ P, atol=1e-4)


# --------------------------------------------------------------------------- conditions
def _spans(n=5, rows_per=3, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = n * rows_per
    sp = C.Spans([f"p{i}" for i in range(n)], np.arange(0, m + 1, rows_per), torch.arange(m), torch.zeros(m, dtype=torch.long),
                 torch.randn(m, d, generator=g), torch.randn(m, d, generator=g))
    return sp


@pytest.mark.parametrize("kind", ["jumprelu", "topk"])
def test_code_writeback_with_all_latents_is_the_feature_difference(kind):
    from harness.model.stages import _active
    from harness.sae import DecoderTable
    d = Dct.random_dict(16, 64, kind, seed=5, k=6, thr=0.2)
    sp = _spans().decompose_with(d)
    cA, cB = _active(d, sp.x_A, None, "cpu"), _active(d, sp.x_B, None, "cpu")
    dec = DecoderTable.from_sae(d, torch.arange(64))
    w = C.codes_writeback(cA, cB, dec, torch.arange(64))
    assert torch.allclose(w, sp.xh_A - sp.xh_B, atol=1e-5)
    full = C.build("feat_codes_along_U", sp, eps_C=w)
    assert torch.allclose(full, sp.xh_A + sp.e_B, atol=1e-5)               # = feat_A_errB when every latent is swapped
    U = torch.linalg.qr(torch.randn(16, 3))[0]
    lat, info = C.select_latents_along_U(cA, cB, dec, U, coverage=0.9)
    assert 0 < info["n_latents"] <= info["n_candidates"] and info["coverage"] >= 0.9 - 1e-6


def test_select_latents_finds_planted_latents():
    from harness.sae import DecoderTable, SparseCodes
    W = torch.eye(8)
    dec = DecoderTable(torch.arange(8), W)
    # A and B differ only on latents 2 and 5; U spans e2, e5
    cA = SparseCodes(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 5]), torch.tensor([1.0, 3.0, 2.0]), torch.zeros(3), 1, 1.0)
    cB = SparseCodes(torch.tensor([0]), torch.tensor([1]), torch.tensor([1.0]), torch.zeros(1), 1, 1.0)
    U = W[:, [2, 5]]
    lat, info = C.select_latents_along_U(cA, cB, dec, U, coverage=0.99)
    assert set(lat.tolist()) == {2, 5}


def test_resampU_with_full_basis_is_the_resample_condition():
    sp = _spans()
    d = Dct.random_dict(16, 64, "jumprelu", seed=1)
    sp.decompose_with(d)
    eC = torch.randn(sp.m, 16)
    I = torch.eye(16)
    assert torch.allclose(C.build("resampU_sameAns", sp, U=I, eps_C=eC), C.build("feat_A_resampC_sameAns", sp, eps_C=eC), atol=1e-5)
    U = torch.linalg.qr(torch.randn(16, 2))[0]
    out = C.build("resampU_diffAns", sp, U=U, eps_C=eC)
    assert torch.allclose(out - (sp.xh_A + sp.e_B), (eC - sp.e_B) @ U @ U.T, atol=1e-5)


# --------------------------------------------------------------------------- estimators
def test_value_directions_recover_planted_structure():
    from harness.directions import FitSet, value_direction
    rng = np.random.default_rng(0)
    d, n = 32, 600
    v = rng.integers(0, 10, n)
    a, b1, b2 = (np.linalg.qr(rng.standard_normal((d, 3)))[0].T)
    X = (v[:, None] * a + np.cos(2 * np.pi * v / 10)[:, None] * b1 + np.sin(2 * np.pi * v / 10)[:, None] * b2
         + 0.05 * rng.standard_normal((n, d)))
    fs = FitSet(tuple(f"t{i}" for i in range(n)))
    ids = list(fs.pair_ids)
    U = value_direction(fs, ids, torch.from_numpy(X).float(), v, "ridge_value").U
    assert U.shape == (d, 1) and abs(float(U[:, 0] @ a)) > 0.95
    P = value_direction(fs, ids, torch.from_numpy(X).float(), v, "fourier_T10").U
    assert P.shape == (d, 2)
    cos = np.linalg.svd(P.T @ np.stack([b1, b2], 1), compute_uv=False)
    assert cos.min() > 0.9
    assert value_direction(fs, ids, torch.from_numpy(X).float(), v, "fourier_T2").U.shape[1] == 1


# --------------------------------------------------------------------------- families
def test_pilot_families():
    from harness import families as F
    from harness import pilotfamilies as PF
    rng = np.random.default_rng(0)
    for fid in PF.P1_FAMILIES[1:]:
        g = F.generator("people_cities", fid, depth=3)
        for _ in range(50):
            p = g(rng)
            if fid == "pc_en_swap":
                assert sorted(p.clean_prompt.split()) == sorted(p.corrupt_prompt.split())   # same tokens, other pairing
            else:
                assert p.clean_prompt.rsplit(".", 1)[0] == p.corrupt_prompt.rsplit(".", 1)[0]  # only the query differs
            assert p.clean_target_str != p.corrupt_target_str
    from src.data import synthetic_tasks as st
    for fid in PF.P11_FAMILIES:
        g = F.generator("entity_track", fid, depth=2)
        f = F._fam("entity_track", fid)
        for _ in range(50):
            p = g(rng)
            if p is None:
                continue
            fb = p.meta["fallback_label"]
            assert fb not in (p.clean_target_str.strip(), p.corrupt_target_str.strip())
            assert f.labels[p.meta["fallback_index"]] == fb
    assert F.expand("people_cities", ["pc_en_swap"]) == ["pc_en_swap"]
    assert F.expand("entity_track", list(PF.P11_FAMILIES)) == list(PF.P11_FAMILIES)


def test_committed_pilot_prompt_sets_load_and_hold_invariants():
    from harness import tasks as T
    for f in ("people_cities_p1", "entity_track_p11", "people_cities_p1_g3", "entity_track_p11_g3", "add_chain_p8_g3"):
        rows, meta = T.load_set(ROOT / "data" / "prompt_sets" / f"{f}.jsonl.gz")
        assert not T.check_invariants(rows), f
        assert len({len(r["cand_ids"]) for r in rows}) == 1, f                    # one candidate count per set (stage 1)


# --------------------------------------------------------------------------- configs, queue, prefetch
def test_pilot_configs_are_generated_and_valid():
    import subprocess
    import sys
    rc = subprocess.run([sys.executable, "scripts/make_pilot_configs.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    for p in sorted((ROOT / "configs" / "pilot").glob("P*.yaml")):
        c = ExpConfig.load(p)
        assert all(cell.role == "exploratory" for cell in c.cells)


def test_pilot_configs_dry_run(capsys):
    from harness import run as R
    for p in sorted((ROOT / "configs" / "pilot").glob("P*.yaml")):
        rc = R.main(["--config", str(p), "--profile", "server", "--dry-run"])
        plan = json.loads(capsys.readouterr().out)
        assert rc in (0, 2) and plan["jobs"], p


def test_pilot_queue_and_prefetch_targets():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_prefetch as rp
    from harness.queue import Queue
    q = Queue.load(ROOT / "queue" / "pilot-breadth-server.yaml")
    for c in q.configs():
        assert (ROOT / c).exists(), c
    for j in q.jobs:
        if j.kind == "script":
            assert (ROOT / j.script).exists(), j.script
    t = rp.targets(q)
    repos = {m[0] for m in t["models"]}
    assert {"google/gemma-2-2b", "google/gemma-3-1b-pt"} <= repos
    keys = {s.key for s in t["saes"]}
    assert "L12_16k_l082" in keys and any("sb_topk" in k for k in keys) and not any("trained" in k or "klft" in k for k in keys)


# --------------------------------------------------------------------------- model-backed (tiny)
@pytest.fixture(scope="module")
def tiny():
    from harness.model import tiny_gemma2
    m = tiny_gemma2(300, n_layers=4, seed=0)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(14):
        T, fd = 12, 8
        a = [2] + list(rng.integers(3, 300, T - 1))
        b = list(a)
        b[fd] = int((a[fd] + 7) % 297 + 3)
        rows.append({"clean_ids": a, "corrupt_ids": b, "first_diff": fd, "T": T, "ans_A": 11, "ans_B": 12, "cand_ids": [11, 12, 13],
                     "pair_id": f"x-train-{i}"})
    from src.models import runtime
    with torch.no_grad():
        xA = torch.cat([runtime.resid_at(m, torch.tensor([r["clean_ids"]]), 1)[0, r["first_diff"]:] for r in rows])
        xB = torch.cat([runtime.resid_at(m, torch.tensor([r["corrupt_ids"]]), 1)[0, r["first_diff"]:] for r in rows])
    sp = C.Spans([r["pair_id"] for r in rows], np.arange(0, 4 * 14 + 1, 4), torch.arange(56), torch.zeros(56, dtype=torch.long), xA, xB)
    sp.decompose_with(Dct.random_dict(64, 128, "jumprelu", seed=1, thr=0.05))
    return m, rows, sp


@pytest.mark.slow
@pytest.mark.parametrize("objective", ["ce", "ld", "kl"])
def test_das_v2_trains_with_holdout(tiny, objective):
    from harness.model.das import fit_das_v2, init_seed_v2
    m, rows, sp = tiny
    U, info = fit_das_v2(m, 1, rows, list(range(14)), sp, 2, objective=objective, epochs=2, seed=0, init_seed=5, step_tokens=48)
    assert U.shape == (64, 2) and np.allclose(U.T @ U, np.eye(2), atol=1e-4)
    assert info["n_val"] >= 2 and info["n_fit"] + info["n_val"] == 14 and info["objective"] == objective
    assert {"iia", "nldr_mean", "share_above_full"} <= set(info["val"])
    assert init_seed_v2(0, "a", 12, "d") != init_seed_v2(0, "b", 12, "d")


@pytest.mark.slow
def test_grad_dirs_first_order_matches_finite_difference(tiny):
    from harness.model.grads import fit_grad_dirs
    from src.models import runtime
    m, rows, sp = tiny
    U, info = fit_grad_dirs(m, 1, rows, list(range(3)), sp, 4)
    assert U["grad_err"].shape == (64, 4) and np.allclose(U["grad_err"].T @ U["grad_err"], np.eye(4), atol=1e-4)
    r, sl = rows[0], sp.rows(0)
    tB = torch.tensor([r["corrupt_ids"]])

    def f(t):
        with torch.no_grad():
            x = runtime.resid_at(m, tB, 1).clone()
            x[0, r["first_diff"]:r["T"]] = sp.xh_A[sl] + sp.e_B[sl] + t * (sp.e_A[sl] - sp.e_B[sl])
            lg = runtime.last_logits_from(m, x, 1, tB)[0]
            return float(lg[r["ans_A"]] - lg[r["ans_B"]])
    one = fit_grad_dirs(m, 1, rows[:1], [0], sp, 2)[1]["first_order_full_mean"]
    fd = (f(1e-2) - f(-1e-2)) / 2e-2
    assert one == pytest.approx(fd, rel=0.05, abs=1e-3)


@pytest.mark.slow
def test_train_dictionary_tiny_both_modes(tmp_path):
    import subprocess
    import sys
    import shutil
    for mode in ("klft", "frozen_decoder"):
        name = f"pytest_{mode}"
        shutil.rmtree(ROOT / "tests" / "_tiny_cache" / "trained_dicts" / name, ignore_errors=True)
        rc = subprocess.run([sys.executable, "scripts/train_dictionary.py", "--tiny", "--mode", mode, "--layer", "1", "--name", name,
                             "--steps", "2"], cwd=ROOT, capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr[-2000:]
        d = Dct.trained_dir(ROOT / "tests" / "_tiny_cache", name)
        assert (d / "complete.json").exists() and (ROOT / "tests" / "_tiny_runs" / "PD" / name / "manifest.json").exists()
        did = Dct.DictId(1, "16k", 0, repo="", source="trained", name=name)
        t = Dct.load_trained(ROOT / "tests" / "_tiny_cache", did)
        assert t.kind == ("topk" if mode == "frozen_decoder" else "jumprelu")
        if mode == "klft":                      # the optimizer must have updated the tensors that were saved
            init = Dct.random_dict(64, 128, "jumprelu", seed=1, layer=1)
            assert not torch.allclose(t.W_enc_T, init.W_enc_T)
        else:                                   # decoder frozen (random unit rows), encoder trained away from its init
            assert torch.allclose(t.W_dec.norm(dim=1), torch.ones(t.W_dec.shape[0]), atol=1e-5)
            assert not torch.allclose(t.W_enc_T, 0.1 * t.W_dec)


@pytest.mark.slow
def test_das_v2_backs_off_on_oom(tiny, monkeypatch):
    """A CUDA OOM on a multi-pair batch splits it (and halves later batches) instead of failing the fit."""
    from harness.model import das as D
    from src.models import runtime
    m, rows, sp = tiny
    real = runtime.last_logits_from

    def flaky(model, resid, layer, tokens):
        if resid.shape[0] > 1 and torch.is_grad_enabled():
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(model, resid, layer, tokens)
    monkeypatch.setattr(D.runtime, "last_logits_from", flaky)
    U, info = D.fit_das_v2(m, 1, rows, list(range(14)), sp, 2, objective="kl", epochs=1, seed=0, init_seed=3, step_tokens=96)
    assert info["oom_backoffs"] >= 1 and info["step_tokens"] < 96 and np.isfinite(U).all()


# --------------------------------------------------------------------------- review fixes (2026-11-22)
def test_signoffs_survive_the_pilot_cf_field():
    """pack_hash ignores the pilot's `cf` selector: every recorded sign-off still matches its pack."""
    from harness import langfamilies as LF
    so = LF.signoffs()
    signed = [f for f in so if any(f in fams for fams in LF.FAMILIES.values())]
    assert signed, "no sign-offs recorded"
    for fid in signed:
        assert LF.is_validated(fid), fid


def test_addon_on_jumprelu_keeps_the_base_code_path():
    """An add-on over a JumpReLU base has no encode_active of its own: encode_sparse keeps alpha_min, and the codes are the
    base's (the add-on changes x_hat, not the features)."""
    base = Dct.random_dict(16, 64, "jumprelu", seed=4, thr=0.3)
    ad = Dct.AddonDict(base, "pcx2", torch.eye(16)[:, :2] @ torch.eye(16)[:2])
    assert getattr(ad, "encode_active", None) is None
    X = torch.randn(30, 16)
    a, b = encode_sparse(ad, X, alpha_min=0.5), encode_sparse(base, X, alpha_min=0.5)
    assert torch.equal(a.col, b.col) and torch.allclose(a.z, b.z) and a.alpha_min == 0.5
    tk = Dct.AddonDict(Dct.random_dict(16, 64, "topk", seed=4, k=3), "pcx2", torch.zeros(16, 16))
    assert tk.encode_active is not None


def test_trained_digest_follows_the_weights(tmp_path):
    d = Dct.trained_dir(tmp_path, "x")
    assert Dct.trained_digest(tmp_path, "x") is None                       # not trained yet
    Dct.save_npz_dict(d / "params.npz", Dct.random_dict(8, 16, "jumprelu", seed=1))
    (d / "complete.json").write_text("{}")
    h1 = Dct.trained_digest(tmp_path, "x")
    Dct.save_npz_dict(d / "params.npz", Dct.random_dict(8, 16, "jumprelu", seed=2))
    assert h1 and Dct.trained_digest(tmp_path, "x") != h1                  # retrained -> new revision -> new cache keys


def test_make_pilot_backfills_resampU_donor_status():
    """Runs from the run-20261122-pilot tag (2df949e) record resampU_* donor_status as 'n/a'; make_pilot copies the cell's
    feat_A_resampC_* status (same donor draw) so no-donor placeholders are excluded."""
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location("make_pilot", Path(__file__).resolve().parents[2] / "scripts" / "make_pilot.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    base = {"task": "t", "layer": 16, "sae": "s", "fit_families": "f"}
    rows = []
    for pid, same, diff in (("p0", "ok", "none"), ("p1", "none", "relaxed")):
        rows += [{**base, "pair_id": pid, "condition": "feat_A_resampC_sameAns", "donor_status": same},
                 {**base, "pair_id": pid, "condition": "feat_A_resampC_diffAns", "donor_status": diff},
                 {**base, "pair_id": pid, "condition": "resampU_sameAns", "donor_status": "n/a"},
                 {**base, "pair_id": pid, "condition": "resampU_diffAns", "donor_status": "n/a"},
                 {**base, "pair_id": pid, "condition": "clean_A", "donor_status": "n/a"}]
    out = mp.backfill_donor_status(pd.DataFrame(rows)).set_index(["pair_id", "condition"]).donor_status
    assert out[("p0", "resampU_sameAns")] == "ok" and out[("p0", "resampU_diffAns")] == "none"
    assert out[("p1", "resampU_sameAns")] == "none" and out[("p1", "resampU_diffAns")] == "relaxed"
    assert out[("p0", "clean_A")] == "n/a"

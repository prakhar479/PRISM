"""Batch-2 add-ons (harness/dicts.py): var / fis / sens / rnd subspaces (predicted and oracle), clin, mlp / cmlp."""
import numpy as np
import pytest
import torch

from harness import dicts as Dct

D, N = 16, 64


@pytest.fixture(scope="module")
def base():
    return Dct.random_dict(D, N, "jumprelu", seed=3, thr=0.3)


def chunks(seed=0, n=3000, d=D):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g) / d ** 0.5
    return [(torch.randn(n, d, generator=g) @ A * 2 + 0.5) for _ in range(3)]


def F_psd(seed=1, d=D):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(d, d, generator=g, dtype=torch.float64)
    return B @ B.T / d


def test_parse_addon_grammar():
    assert Dct.parse_addon("lin")["kind"] == "lin" and Dct.parse_addon("clin")["needs_codes"]
    s = Dct.parse_addon("osens32")
    assert (s["kind"], s["k"], s["oracle"], s["needs_F"]) == ("sens", 32, True, True)
    assert Dct.parse_addon("var8") == {"kind": "var", "k": 8, "oracle": False, "needs_F": False, "needs_codes": False}
    assert Dct.parse_addon("pcx24")["oracle"] and Dct.parse_addon("cmlp128")["needs_codes"]
    for bad in ("", "sens", "sens0", "xvar8", "mlp", "ovarx", "oclin", "pcx0"):
        with pytest.raises(ValueError):
            Dct.parse_addon(bad)


def test_subspace_choice_variance_vs_sensitivity_on_a_planted_case():
    """Error varies most along e0 (insensitive) and little along e1 (very sensitive): var picks e0, fis and sens pick e1."""
    Cee = torch.diag(torch.tensor([100.0, 1.0] + [0.01] * (D - 2), dtype=torch.float64))
    F = torch.diag(torch.tensor([0.01, 100.0] + [0.01] * (D - 2), dtype=torch.float64))
    pick = lambda kind: int(Dct.error_subspace(kind, 1, Cee, F)[:, 0].abs().argmax())
    assert (pick("var"), pick("fis"), pick("sens")) == (0, 1, 1)
    for kind in ("var", "fis", "sens", "rnd"):
        U = Dct.error_subspace(kind, 5, Cee, F, seed=2)
        assert U.shape == (D, 5) and torch.allclose(U.T @ U, torch.eye(5, dtype=torch.float64), atol=1e-9)
    a, b = Dct.error_subspace("rnd", 3, Cee, F, seed=1), Dct.error_subspace("rnd", 3, Cee, F, seed=2)
    assert not torch.allclose(a, b) and torch.allclose(a, Dct.error_subspace("rnd", 3, Cee, F, seed=1))


@pytest.mark.parametrize("mode", ["var4", "fis4", "sens4", "rnd4", "ovar4", "osens4", "ofis4", "ornd4", "lin", "pcx4"])
def test_addon_keeps_the_decomposition_exact(base, mode):
    """x_hat' + eps' = x for every add-on, and the oracle of a full-rank subspace removes the error entirely."""
    F = F_psd()
    fit = Dct.fit_addon(base, chunks(), mode, F=F, seed=5)
    a = Dct.AddonDict(base, mode, fit["W"], fit.get("c"), fit["info"])
    x = chunks(seed=9, n=500)[0]
    xh, eps, l0 = a.decompose_rows(x)
    assert torch.allclose(xh + eps, x, atol=1e-4)
    if mode.startswith("o") or mode.startswith("pcx"):
        assert fit["c"] is None
    elif mode != "lin":
        assert int(torch.linalg.matrix_rank(fit["W"])) <= 4          # the predictor lives in the k-dim subspace
        assert 0.0 < fit["info"]["r2_predictor"] <= 1.0 and 0.0 < fit["info"]["eps_var_share"] <= 1.0 + 1e-9


def test_full_rank_subspace_equals_lin_and_full_oracle_removes_the_error(base):
    F = F_psd()
    lin = Dct.fit_addon(base, chunks(), "lin")
    full = Dct.fit_addon(base, chunks(), f"var{D}")
    assert torch.allclose(lin["W"], full["W"], atol=1e-4) and torch.allclose(lin["c"], full["c"], atol=1e-4)
    o = Dct.AddonDict(base, f"osens{D}", *[Dct.fit_addon(base, chunks(), f"osens{D}", F=F)[k] for k in ("W", "c")])
    x = chunks(seed=4, n=300)[0]
    _, eps, _ = o.decompose_rows(x)
    assert float(eps.abs().max()) < 1e-4


def test_oracle_variance_subspace_removes_the_top_variance_share(base):
    fit = Dct.fit_addon(base, chunks(), "ovar4")
    a = Dct.AddonDict(base, "ovar4", fit["W"], None, fit["info"])
    x = chunks(seed=11, n=4000)[0]
    _, e0, _ = Dct.decompose(base, x) if hasattr(Dct, "decompose") else __import__("harness.sae", fromlist=["decompose"]).decompose(base, x)
    _, e1, _ = a.decompose_rows(x)
    removed = 1 - float(e1.pow(2).sum() / e0.pow(2).sum())
    assert removed > 0.6 * fit["info"]["eps_var_share"] and removed <= 1.0           # held-out rows, same share up to sampling


def test_sens_and_fis_require_the_fisher(base):
    with pytest.raises(ValueError):
        Dct.fit_addon(base, chunks(), "sens4")
    Dct.fit_addon(base, chunks(), "var4")            # var / rnd do not


def test_clin_equals_an_independent_least_squares_on_the_codes(base):
    xs = chunks(n=4000)
    fit = Dct.fit_addon(base, xs, "clin", lam=1e-9)
    tmp = Dct.AddonDict(base, "clin", torch.zeros(1, 1))
    X = torch.cat(xs)
    Fm = tmp.codes(X).double().numpy()
    _, E, _ = __import__("harness.sae", fromlist=["decompose"]).decompose(base, X)
    A = np.concatenate([Fm, np.ones((len(Fm), 1))], 1)
    sol, *_ = np.linalg.lstsq(A, E.double().numpy(), rcond=None)
    pred_ref = A @ sol
    a = Dct.AddonDict(base, "clin", fit["W"], fit["c"], fit["info"])
    pred = a.addon(X, E).double().numpy()
    # dead / constant codes make the system rank-deficient: compare predictions (identical), not coefficients
    assert np.abs(pred - pred_ref).max() < 5e-3 * max(1.0, np.abs(pred_ref).max())
    assert float(((E.double().numpy() - pred) ** 2).sum()) < float((E.double().numpy() ** 2).sum())


@pytest.mark.parametrize("mode", ["mlp32", "cmlp32"])
def test_mlp_addons_fit_and_stay_exact(base, mode):
    fit = Dct.fit_addon(base, chunks(n=2500), mode, seed=1)
    assert fit["info"]["r2_heldout"] > -0.05 and set(fit["extra"]) == {"b1", "W2", "b2", "scale_in", "scale_out"}
    a = Dct.AddonDict(base, mode, fit["W"], fit["c"], fit["info"], fit["extra"])
    x = chunks(seed=8, n=400)[0]
    xh, eps, _ = a.decompose_rows(x)
    assert torch.allclose(xh + eps, x, atol=1e-4)
    a.to_device(torch.device("cpu"))


def test_codes_addons_need_a_jumprelu_base():
    topk = Dct.random_dict(D, N, "topk", seed=1, k=4)
    with pytest.raises(ValueError):
        Dct.AddonDict(topk, "clin", torch.zeros(1, 1))
    Dct.AddonDict(topk, "var4", torch.zeros(D, D), torch.zeros(D))        # predictor-only add-ons work on any base


@pytest.mark.slow
def test_tiny_run_with_batch2_addons_end_to_end(tmp_path):
    """fit_sensitivity (tiny, smoke corpus) then a tiny run whose cells use every new add-on kind; the oracle of the full
    space makes feat_A_errB equal full_A."""
    import sys
    import pandas as pd
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    import fit_sensitivity as fs
    from harness import run as R
    assert fs.main(["--tiny", "--smoke", "--layers", "1", "--generic", "generic/pile10k_6000x128_skip100.json"]) == 0
    d = yaml.safe_load((root / "configs" / "pilot" / "P4.yaml").read_text())
    modes = ["lin", "var8", "fis8", "sens8", "rnd8", "ovar8", "osens8", "osens64", "clin", "mlp32", "cmlp32"]
    cell = d["cells"][1]
    d["cells"] = [{**d["cells"][0], "layer": 1, "sae": {"width": "16k", "l0": 40, "rule": "explicit"}}] + \
                 [{**d["cells"][0], "layer": 1, "sae": {"width": "16k", "l0": 40, "rule": "explicit", "addon": m}} for m in modes]
    d.update(exp_id="P4", name="tiny_b2", bootstrap=20, smoke_pairs=5, conditions=["feat_A_errB"])
    d["prompt_sets"] = {k: v for k, v in d["prompt_sets"].items() if k == d["cells"][0]["task"]}
    d["directions"].update(ranks=[2], estimators=["das4v2_kl"], random_seeds=2, r_max=2)
    p = tmp_path / "b2.yaml"
    p.write_text(yaml.safe_dump(d))
    assert R.main(["--config", str(p), "--profile", "dev", "--tiny", "--smoke", "--allow-dirty"]) == 0
    run = sorted((root / "tests" / "_tiny_runs" / "P4").glob("*/manifest.json"), key=lambda q: q.stat().st_mtime)[-1].parent
    pp = pd.read_parquet(run / "per_pair.parquet")
    assert {f"__{m}" for m in modes} <= {s[s.index("__"):] for s in set(pp.sae) if "__" in s}
    ref = pp[(pp.condition == "full_A") & pp.sae.str.endswith("__osens64")].set_index("pair_id")
    got = pp[(pp.condition == "feat_A_errB") & pp.sae.str.endswith("__osens64")].set_index("pair_id").loc[ref.index]
    assert np.allclose(ref.logit_A.to_numpy(), got.logit_A.to_numpy(), atol=1e-3)

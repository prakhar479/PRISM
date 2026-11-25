"""Composite dictionaries (P12/P13, harness/composite.py) on the tiny model: stage-1 sublayer sites, the eps of every part
kind against a direct reference, and an end-to-end tiny run with composite cells and the diffmean estimator."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from harness import composite as Cm
from harness.batching import Budget
from harness.cache import Cache
from harness.model import ModelSpec, tiny_gemma2
from harness.model import stages as S

ROOT = Path(__file__).resolve().parents[2]
L = 2
XC = "crosscoder/layer_1_2_3_width_65k_l0_medium"
RECIPES = {
    "tc_all": [f"transcoder_all/layer_{l}_width_16k_l0_small/params.safetensors" for l in range(L + 1)],
    "tc_aff_all+attn": [f"transcoder_all/layer_{l}_width_16k_l0_small_affine/params.safetensors" for l in range(L + 1)]
                       + [f"attn_out_all/layer_{l}_width_16k_l0_small/params.safetensors" for l in range(L + 1)],
    "mlpsae_all": [f"mlp_out_all/layer_{l}_width_16k_l0_small/params.safetensors" for l in range(L + 1)],
    "clt_aff": [f"clt/width_262k_l0_medium_affine/params_layer_{l}.safetensors" for l in range(L + 1)],
    "tc_single": [f"transcoder/layer_{L}_width_16k_l0_medium/params.safetensors"],
    "xc": [f"{XC}/params_layer_{i}.safetensors" for i in range(2)],
}


def fake_files(root: Path, n_layers: int, d=64, dz=64, n=96):
    """Random Gemma-Scope-2-shaped files (same construction as harness.run._tiny_comp_fetch)."""
    from safetensors.torch import save_file

    def fetch(f):
        path = root / f
        if not path.exists():
            part = Cm.parse_parts([f])[0]
            g = torch.Generator().manual_seed(int(hashlib.sha256(f.encode()).hexdigest()[:8], 16))
            d_in = dz if part.kind == "attn" else d
            t = {"w_enc": torch.randn(d_in, n, generator=g) / d_in ** 0.5, "b_enc": torch.zeros(n),
                 "threshold": torch.full((n,), 0.05), "b_dec": torch.randn(d_in, generator=g) * 0.01}
            if part.kind == "clt":
                W = torch.randn(n, n_layers, d, generator=g) / n ** 0.5
                W[:, :part.layer] = 0
                t["w_dec"] = W
            elif part.kind == "xc":
                W = torch.randn(n, len(part.xc_layers), d, generator=g) / n ** 0.5
                W[:, :part.block] = 0
                t["w_dec"] = W
            else:
                t["w_dec"] = torch.randn(n, d_in, generator=g) / n ** 0.5
            if part.affine:
                t["affine_skip_connection"] = torch.randn(d, d, generator=g) * 0.01
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(t, str(path))
        return path
    return fetch


def test_parse_validate_and_sites():
    ps = Cm.parse_parts(RECIPES["tc_aff_all+attn"])
    assert {p.kind for p in ps} == {"tc", "attn"} and all(p.affine for p in ps if p.kind == "tc")
    assert Cm.needed_sites(ps, L) == {(l, s) for l in range(L + 1) for s in ("mlp_in", "mlp_out", "attn_z")}
    xc = Cm.parse_parts(RECIPES["xc"])
    assert [p.layer for p in xc] == [1, 2] and Cm.needed_sites(xc, L) == {(1, "resid_post")}
    Cm.validate(xc, L)
    with pytest.raises(ValueError):
        Cm.validate(xc[:1], L)                                    # a crosscoder at L2 needs blocks 0 and 1
    with pytest.raises(ValueError):
        Cm.validate(Cm.parse_parts(RECIPES["clt_aff"][1:]), L)    # a CLT needs every source layer 0..L
    assert Cm.needed_sites(Cm.parse_parts(RECIPES["tc_single"]), L) == {(L, "mlp_in"), (L, "mlp_out")}
    # the committed registry's composites are well formed at their layers
    reg = yaml.safe_load((ROOT / "configs" / "dict_registry.yaml").read_text())["dictionaries"]
    comps = {k: v for k, v in reg.items() if v["arch"] == "gs2c"}
    assert len(comps) >= 20
    for k, v in comps.items():
        Cm.validate(Cm.parse_parts(v["files"]), int(v["layer"]))


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("comp")
    n_layers = 4
    m = tiny_gemma2(300, n_layers=n_layers, seed=0)
    spec = ModelSpec(name="tiny", dtype="float32", tiny_vocab=300, tiny_layers=n_layers)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        T, fd = 12, 8
        a = [2] + list(map(int, rng.integers(3, 300, T - 1)))
        b = list(a)
        b[fd] = int((a[fd] + 7) % 297 + 3)
        rows.append({"clean_ids": a, "corrupt_ids": b, "first_diff": fd, "T": T, "ans_A": 11, "ans_B": 12, "cand_ids": [11, 12, 13],
                     "pair_id": f"x-train-{i}"})
    cache = Cache(tmp / "cache")
    sites = set()
    for files in RECIPES.values():
        sites |= Cm.needed_sites(Cm.parse_parts(files), L)
    S.stage1(m, spec, rows, "h0", [L] + sorted(sites), cache, Budget(max_tokens=64, max_rows=4), torch.device("cpu"), shard_pairs=4)
    S.write_sublayer_params(m, spec, cache, list(range(L + 1)))
    return m, spec, rows, cache, fake_files(tmp / "gs2", n_layers)


def _ref_eps(m, rows, name, fetch):
    """Direct eps at layer L from a TransformerLens cache of the A prompts (float64), rows in pair order."""
    from safetensors.torch import load_file
    out = []
    for r in rows:
        _, c = m.run_with_cache(torch.tensor([r["clean_ids"]]))
        sl = slice(r["first_diff"], r["T"])
        g = lambda k: c[k][0, sl].double()
        mlp_in = {l: (m.blocks[l].ln2(c[f"blocks.{l}.hook_resid_mid"]))[0, sl].double() for l in range(L + 1)}
        W = lambda f: {k: v.double() for k, v in load_file(str(fetch(f))).items()}
        jr = lambda x, w: torch.relu(x @ w["w_enc"] + w["b_enc"]) * ((x @ w["w_enc"] + w["b_enc"]) > w["threshold"])
        eps = torch.zeros_like(g(f"blocks.{L}.hook_resid_post"))
        for p in Cm.parse_parts(RECIPES[name]):
            w = W(p.file)
            if p.kind == "tc":
                eps += g(f"blocks.{p.layer}.hook_mlp_out") - (jr(mlp_in[p.layer], w) @ w["w_dec"] + w["b_dec"]
                                                              + (mlp_in[p.layer] @ w["affine_skip_connection"] if p.affine else 0))
            elif p.kind == "mlpsae":
                y = g(f"blocks.{p.layer}.hook_mlp_out")
                eps += y - (jr(y, w) @ w["w_dec"] + w["b_dec"])
            elif p.kind == "attn":
                b = m.blocks[p.layer]
                z = g(f"blocks.{p.layer}.attn.hook_z").flatten(-2)
                o = lambda zz: b.ln1_post((zz @ b.attn.W_O.double().reshape(-1, 64) + b.attn.b_O.double()).float()).double()
                assert torch.allclose(o(z), g(f"blocks.{p.layer}.hook_attn_out"), atol=1e-4)     # the attention output map
                eps += o(z) - o(jr(z, w) @ w["w_dec"] + w["b_dec"])
        if name == "clt_aff":
            ws = [W(f) for f in RECIPES[name]]
            for j in range(L + 1):
                yh = sum(jr(mlp_in[l], ws[l]) @ ws[l]["w_dec"][:, j, :] for l in range(j + 1)) + ws[j]["b_dec"] \
                    + mlp_in[j] @ ws[j]["affine_skip_connection"]
                eps += g(f"blocks.{j}.hook_mlp_out") - yh
        if name == "xc":
            ws = [W(f) for f in RECIPES[name]]
            xh = sum(jr(g(f"blocks.{l}.hook_resid_post"), ws[i]) @ ws[i]["w_dec"][:, i_L, :]
                     for i, l, i_L in ((0, 1, 1), (1, 2, 1))) + ws[1]["b_dec"]
            eps = g(f"blocks.{L}.hook_resid_post") - xh
        out.append(eps)
    return torch.cat(out)


@pytest.mark.parametrize("name", list(RECIPES))
def test_composite_eps_matches_direct_reference(staged, name):
    m, spec, rows, cache, fetch = staged
    comp = Cm.Composite(Cm.parse_parts(RECIPES[name]), L, "16k", name, fetch, row_budget_bytes=2048)   # many row chunks
    base = S.site_entry(cache, spec, "h0", L, "resid_post")
    ents = {k: S.site_entry(cache, spec, "h0", *k) for k in comp.sites()}
    sub = S.load_sublayer_params(cache, spec, Cm.attn_layers(comp.parts)) if Cm.attn_layers(comp.parts) else None
    acc = comp.eps_for_entries(base, ents, sub)
    order, eps = [], []
    for si in base.shards():
        order += base.read_shard(si)["pair_index"].tolist()
        eps.append(acc[si]["eps_A"])
    assert order == sorted(order)                                # equal T: shard plan keeps pair order
    got = torch.cat(eps).double()
    ref = _ref_eps(m, [rows[i] for i in order], name, fetch)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-4, rtol=1e-4), float((got - ref).abs().max())
    # stage 2 writes x_hat = x - eps into the ordinary sae_out entry
    e = S.stage2_composite(cache, base, ents, comp, {"name": name}, sub)
    s0, b0 = e.read_shard(base.shards()[0]), base.read_shard(base.shards()[0])
    assert torch.allclose(b0["x_A"].float() - s0["xh_A"], acc[base.shards()[0]]["eps_A"], atol=1e-6)


def test_sites_decompose_the_residual_exactly(staged):
    """resid_post[L] - resid_pre[0] = sum over l <= L of (attn_out_l + mlp_out_l): what makes eps a decomposition."""
    m, *_ = staged
    _, c = m.run_with_cache(torch.tensor([[2, 5, 9, 33, 70, 101]]))
    tot = sum(c[f"blocks.{l}.hook_attn_out"] + c[f"blocks.{l}.hook_mlp_out"] for l in range(L + 1))
    assert torch.allclose(c[f"blocks.{L}.hook_resid_post"] - c["blocks.0.hook_resid_pre"], tot, atol=1e-4)


@pytest.mark.slow
def test_tiny_run_with_composite_cells_and_diffmean(tmp_path, monkeypatch):
    from harness import run as R
    reg = {"dictionaries": {}}
    for name in ("tc_aff_all+attn", "clt_aff", "xc", "tc_single"):
        reg["dictionaries"][f"t_{name}"] = {"group": "t", "layer": L, "width": "16k", "l0": 10, "arch": "gs2c", "repo": "x/y",
                                            "revision": "r0", "files": RECIPES[name]}
    rp = tmp_path / "reg.yaml"
    rp.write_text(yaml.safe_dump(reg))
    monkeypatch.setenv("HL_DICT_REGISTRY", str(rp))
    d = yaml.safe_load((ROOT / "configs" / "pilot" / "P13.yaml").read_text())
    cells = [{"task": "people_cities", "layer": L, "sae": {"width": "16k", "l0": 24, "rule": "explicit"}, "role": "exploratory",
              "fit_families": ["pc_en"], "eval_families": ["pc_en"]}]
    cells += [{"task": "people_cities", "layer": L, "sae": {"dict": f"t_{n}"}, "role": "exploratory", "fit_families": ["pc_en"],
               "eval_families": ["pc_en"]} for n in ("tc_aff_all+attn", "clt_aff", "xc", "tc_single")]
    d.update(exp_id="P12", name="tiny_composite", model="gemma-2-2b", model_revision="c5ebcd40d208330abc697524c919956e692655cf",
             prompt_sets={"people_cities": "prompt_sets/people_cities_p1.jsonl.gz"}, cells=cells, bootstrap=20, smoke_pairs=5)
    d["directions"].update(ranks=[1, 2], estimators=["diffmean", "pca_pooled"], random_seeds=2, r_max=2)
    p = tmp_path / "P12_tiny.yaml"
    p.write_text(yaml.safe_dump(d))
    assert R.main(["--config", str(p), "--profile", "dev", "--tiny", "--smoke"]) == 0
    run_dir = sorted((ROOT / "tests" / "_tiny_runs" / "P12").glob("*/manifest.json"), key=lambda q: q.stat().st_mtime)[-1].parent
    pp = pd.read_parquet(run_dir / "per_pair.parquet")
    assert {f"L{L}_t_{n}" for n in ("tc_aff_all+attn", "clt_aff", "xc", "tc_single")} <= set(pp.sae)
    est = set(pp[pp.condition == "feat_A_along_U"].estimator)
    assert {"diffmean", "pca_pooled"} <= est
    assert not len(pp[(pp.estimator == "diffmean") & (pp.r > 1)])          # 1 column


def test_g3_queue_configs_and_prefetch_targets():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_prefetch as rp
    from harness.config import ExpConfig
    from harness.queue import Queue
    q = Queue.load(ROOT / "queue" / "pilot-g3-server.yaml")
    assert set(q.configs()) == {"configs/pilot/P12.yaml", "configs/pilot/P13.yaml", "configs/pilot/P14.yaml"}
    t = rp.targets(q)
    assert {"google/gemma-3-1b-pt", "google/gemma-3-1b-it", "google/gemma-2-2b"} <= {m[0] for m in t["models"]}
    files = {f for s in t["saes"] for f in s.repo_files()}
    assert any("clt/width_262k_l0_medium/params_layer_13" in f[1] for f in files)
    assert any("crosscoder/" in f[1] for f in files) and any("attn_out_all/" in f[1] for f in files)
    p12, p13 = ExpConfig.load(ROOT / "configs/pilot/P12.yaml"), ExpConfig.load(ROOT / "configs/pilot/P13.yaml")
    assert p12.model == "gemma-3-1b-pt" and p13.model == "gemma-3-1b-it"
    assert sum(c.sae.dict.startswith("gs2_1bc_") for c in p12.cells) == 2 * 26 and "diffmean" in p13.directions.estimators


def test_only_sae_filters_cells_and_changes_the_run_hash(capsys):
    from harness import run as R
    p = str(ROOT / "configs/pilot/P12.yaml")
    assert R.main(["--config", p, "--profile", "server", "--dry-run"]) in (0, 2)
    full = json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        R.main(["--config", p, "--profile", "server", "--dry-run", "--only-sae", "gs2_1bc_L13_xc"])   # real runs keep every cell
    capsys.readouterr()
    assert R.main(["--config", p, "--profile", "server", "--dry-run", "--smoke", "--only-sae", "gs2_1bc_L13_xc"]) in (0, 2)
    part = json.loads(capsys.readouterr().out)
    assert part["run_hash"] != full["run_hash"] and len(part["jobs"]) < len(full["jobs"])


def test_preflight_checks_the_gemma2_revision_of_a_gemma3_first_queue():
    """pilot-g3's first config is a Gemma 3 one: preflight must still load gemma-2-2b at its own pinned commit."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_preflight as pf
    from harness.queue import Queue
    g3 = Queue.load(ROOT / "queue" / "pilot-g3-server.yaml")
    assert pf.queue_revision(g3) == "c5ebcd40d208330abc697524c919956e692655cf"
    assert pf.queue_revision(Queue.load(ROOT / "queue" / "pilot-breadth-server.yaml")) == pf.queue_revision(g3)
    assert pf.queue_revision(Queue.load(ROOT / "queue" / "release2-server.yaml")) == "c5ebcd40d208330abc697524c919956e692655cf"


def test_every_committed_pilot_set_has_one_candidate_count():
    """Stage 1 stacks each row's cand_ids into one tensor: a set with rows of different candidate counts fails at the first
    batch that mixes them (P13's refusal set did: 15 vs 16)."""
    from harness import tasks as T
    for f in sorted((ROOT / "data" / "prompt_sets").glob("*.jsonl.gz")):
        rows, _ = T.load_set(f)
        assert len({len(r["cand_ids"]) for r in rows}) == 1, f.name
        assert all(r["ans_A"] in r["cand_ids"] and r["ans_B"] in r["cand_ids"] for r in rows), f.name


def test_refusal_builder_unifies_candidates():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_refusal_set as b
    rows = b.unify_candidates([{"cand_ids": [1, 2, 3, 9]}, {"cand_ids": [1, 2, 3]}, {"cand_ids": [1, 2, 7, 3]}])
    assert {tuple(r["cand_ids"]) for r in rows} == {(1, 2, 3, 9, 7)}


def test_verify_pilot_runs_accepts_a_consistent_tiny_run_and_rejects_it_as_a_real_one():
    """scripts/verify_pilot_runs.py: a tiny smoke run passes only with --allow-smoke (it is not a real run); strict mode fails it."""
    import subprocess
    import sys
    runs = ROOT / "tests" / "_tiny_runs" / "P13"
    done = [m for m in runs.glob("*/manifest.json") if json.loads(m.read_text()).get("status") == "complete"]
    if not done:
        pytest.skip("no tiny P13 run in tests/_tiny_runs (test_tiny_run_with_composite_cells_and_diffmean builds P12 only)")
    cmd = [sys.executable, "scripts/verify_pilot_runs.py", "--queue", "queue/pilot-g3-server.yaml", "--runs", "tests/_tiny_runs", "--only", "P13"]
    ok = subprocess.run(cmd + ["--allow-smoke"], cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout[-1500:] + ok.stderr[-500:]
    strict = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert strict.returncode == 1 and "not a real run" in strict.stdout


def test_verify_classifies_capability_skips_apart_from_failures():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import verify_pilot_runs as v
    cells = [{"cell": "add_chain_L13_x", "skipped": "add_chain L13: only 3 train pairs pass f_pilot in families ['rs_en'] (need 20)"},
             {"cell": "people_cities_L13_x", "skipped": "people_cities_L13_x: 4 test pairs"},
             {"cell": "refusal_L7_y", "skipped": "needs 31.0 GiB host RAM (x1.5 margin), 20.0 GiB available"},
             {"cell": "refusal_L13_z"}]
    cap, other = v.classify_skips(cells)
    assert [c["cell"] for c in cap] == ["add_chain_L13_x", "people_cities_L13_x"] and [c["cell"] for c in other] == ["refusal_L7_y"]
    assert v.cell_task("people_cities_L13_x") == "people_cities" and v.cell_task("add_chain_L13_x") == "add_chain"


def test_p14_runs_the_p12_dictionaries_on_object_tracking():
    from harness.config import ExpConfig
    p12, p14 = ExpConfig.load(ROOT / "configs/pilot/P12.yaml"), ExpConfig.load(ROOT / "configs/pilot/P14.yaml")
    assert {c.task for c in p14.cells} == {"entity_track"} and len(p14.cells) == len(p12.cells) // 2
    assert {c.sae.dict for c in p14.cells} == {c.sae.dict for c in p12.cells}            # the same dictionaries
    assert p14.model == p12.model and p14.model_revision == p12.model_revision
    assert all(c.fit_families == ["ot_placed_dest", "ot_bin_dest", "ot_drawer_dest"] for c in p14.cells)


def test_select_cells_selectors():
    from harness.config import ExpConfig
    from harness.run import select_cells
    c = ExpConfig.load(ROOT / "configs/batch2/P23.yaml").cells
    assert len(select_cells(c, ["addon=sens8"])) == 3 and len(select_cells(c, ["addon=osens"])) == 12
    assert len(select_cells(c, ["addon=NONE"])) == 3 and {x.task for x in select_cells(c, ["task=entity_track"])} == {"entity_track"}
    assert len(select_cells(c, ["addon=NONE", "layer=18"])) == 36 and select_cells(c, ["width=65k"]) == []
    d = ExpConfig.load(ROOT / "configs/batch2/P20.yaml").cells
    assert len(select_cells(d, ["gs2_1b_all_L13_"])) == 6 and len(select_cells(d, ["gs2_1b_all_L0_", "gs2_1b_all_L25_"])) == 12
    with pytest.raises(ValueError):
        select_cells(c, ["bogus=1"])

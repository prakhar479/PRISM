"""E4 (matched-L0 widths) and E5 (subthreshold add-back): config generation, sparse codes cache, decoder table,
subthresh condition semantics (handoff test 14), greedy latent selection (train only), tiny runner end to end."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from harness import conditions as C
from harness import sae as S
from harness.cache import Cache
from harness.directions import FitSet, LeakageError
from harness.model import stages as ST

ROOT = Path(__file__).resolve().parents[2]
D, N = 24, 96


@pytest.fixture(scope="module")
def sae(tmp_path_factory):
    p = S.write_random_params(tmp_path_factory.mktemp("sae") / "p.npz", D, N, seed=3, thr=0.6)
    return S.load_params(p, 0, "96")


def _x(n=40, seed=0):
    return torch.randn(n, D, generator=torch.Generator().manual_seed(seed)) * 1.5


# --------------------------------------------------------------------------- configs
def test_configs_match_generator():
    r = subprocess.run([sys.executable, "scripts/make_e4_config.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_e4_variants_come_from_the_inventory_by_match_l0():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_e4_config as M

    inv = json.loads((ROOT / "configs" / "sae_inventory.json").read_text())["inventory"]
    v = M.variants_for(inv, 12, bracket=True)
    assert v[0] == ("16k", S.canonical_l0(inv["12"]["16k"]), "pilot (canonical)")
    target = v[0][1]
    for w in inv["12"]:
        if w == "16k":
            continue
        m = S.match_l0(target, inv["12"][w])
        picked = {l0 for ww, l0, _ in v if ww == w}
        assert m["closest"] in picked and picked <= {m["closest"], m["below"], m["above"]}
    closest_only = M.variants_for(inv, 12, bracket=False)
    assert len(closest_only) == len(inv["12"])                                  # one variant per width
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "E4.yaml").read_text())
    assert not any(c["sae"]["width"] == "1m" for c in cfg["cells"])           # 1m lives in E4_1m.yaml
    assert all(c["sae"]["width"] == "1m" for c in yaml.safe_load(
        (ROOT / "configs" / "experiments" / "E4_1m.yaml").read_text())["cells"])


# --------------------------------------------------------------------------- codes, decoder table
def test_decoder_table_decodes_like_the_sae(sae):
    codes = S.encode_sparse(sae, _x(), alpha_min=0.3)
    m = codes.z > 0
    tab = S.DecoderTable.from_sae(sae, codes.col)
    assert torch.allclose(S.decode_entries(tab, codes, m), S.decode_entries(sae, codes, m), atol=1e-6)
    with pytest.raises(KeyError):
        S.DecoderTable.from_sae(sae, torch.tensor([1, 2])).rows(torch.tensor([3]))


def _spans_entry(cache, X_A, X_B, lens):
    e = cache.entry("spans", model="t", set_hash="h", layer=0, site="resid_post").open()
    a = 0
    for si, k in enumerate((2, len(lens) - 2)):                              # two shards
        ls = lens[si * 2: si * 2 + k] if si == 0 else lens[2:]
        n = int(sum(ls))
        idx = list(range(0, 2)) if si == 0 else list(range(2, len(lens)))
        e.write_shard(si, {"pair_index": torch.tensor(idx), "span_len": torch.tensor(ls), "x_A": X_A[a:a + n],
                           "x_B": X_B[a:a + n]})
        a += n
    e.mark_complete({})
    return e


@pytest.mark.parametrize("block", [None, 17])
def test_stage2_codes_equal_direct_encoding_subset_and_blockwise(tmp_path, sae, block):
    lens = [3, 5, 2, 4, 6]
    X_A, X_B = _x(sum(lens), 1), _x(sum(lens), 2)
    cache = Cache(tmp_path)
    sp = _spans_entry(cache, X_A, X_B, lens)
    ce, de = ST.stage2_codes(cache, sp, sae, {"id": "s"}, torch.device("cpu"), 0.5, block_latents=block)
    pick = [3, 0]                                                             # pair order as in load_spans
    got = ST.load_codes(ce, sp, pick)
    ptr = np.concatenate([[0], np.cumsum(lens)])
    rows = torch.cat([torch.arange(ptr[i], ptr[i + 1]) for i in pick])
    ref = S.subthreshold_only(S.encode_sparse(sae, X_A[rows], alpha_min=0.5))
    order = lambda c: torch.argsort(c.row * 10_000 + c.col)
    g, r = order(got), order(ref)
    assert got.n_rows == len(rows)
    assert torch.equal(got.row[g], ref.row[r]) and torch.equal(got.col[g], ref.col[r])       # same entries
    assert torch.allclose(got.z[g], ref.z[r], atol=1e-5)                                   # fp32 batch rounding
    assert not got.active.any()                                               # only inactive entries are stored
    tab = ST.load_decoder(de)
    assert set(got.col.tolist()) <= set(tab.cols.tolist())


# --------------------------------------------------------------------------- subthresh semantics (handoff test 14)
def _spans_from(sae, X_A, X_B):
    sp = C.Spans(["p-train-0"], np.array([0, len(X_A)]), torch.arange(len(X_A)), torch.arange(len(X_A)).flip(0), X_A, X_B)
    return sp.decompose_with(sae)


def test_subthresh_alpha_one_adds_nothing_and_small_alpha_adds_every_positive_subthreshold_feature(sae):
    X_A, X_B = _x(30, 4), _x(30, 5)
    sp = _spans_from(sae, X_A, X_B)
    codes = S.subthreshold_only(S.encode_sparse(sae, X_A, alpha_min=1e-6))
    one = C.build("subthresh", sp, codes_A=codes, sae=sae, alpha=1.0)
    assert torch.equal(one, C.build("feat_A_errB", sp))
    small = C.build("subthresh", sp, codes_A=codes, sae=sae, alpha=1e-6)
    z = X_A @ sae.W_enc_T.T + sae.b_enc                                       # dense reference
    keep = (z > 0) & (z < sae.threshold)
    ref = sp.xh_A + (z * keep) @ sae.W_dec + sp.e_B
    assert torch.allclose(small, ref, atol=1e-4)
    m = torch.randn(30, D)
    mp = C.build("subthresh", sp, codes_A=codes, sae=sae, alpha=1.0, slot_error="meanPos", eps_mean_pos=m)
    assert torch.equal(mp, sp.xh_A + m)


# --------------------------------------------------------------------------- greedy selection
def test_greedy_picks_the_explaining_latents_on_train_only(sae):
    X_A = _x(60, 6)
    codes = S.subthreshold_only(S.encode_sparse(sae, X_A, alpha_min=0.5))
    m = (codes.z >= 0.5 * codes.threshold) & (codes.z < codes.threshold) & (codes.z > 0)
    cols, counts = torch.unique(codes.col[m], return_counts=True)
    a, b = [int(c) for c in cols[torch.argsort(counts, descending=True)[:2]]]
    tab = S.DecoderTable.from_sae(sae, codes.col)
    only = lambda i: S.decode_entries(tab, codes, m & (codes.col == i))
    target = only(a) + only(b)                                                # explained exactly by {a, b}
    ids = [f"t-train-{i}" for i in range(3)]
    fs = FitSet(tuple(ids))
    g = C.greedy_subthreshold(fs, ids, codes, tab, target, 0.5, k_max=5)
    assert set(g.latents[:2]) == {a, b}
    assert g.objective[2] < 1e-6 * g.objective[0]                             # nothing left to explain
    assert all(x >= y - 1e-6 for x, y in zip(g.objective, g.objective[1:]))  # non-increasing
    g_half = C.greedy_subthreshold(fs, ids, codes, tab, only(a) + 0.5 * only(b), 0.5, k_max=5)
    assert g_half.latents[0] == a and b not in g_half.latents                 # adding all of b does not help
    with pytest.raises(LeakageError):
        C.greedy_subthreshold(fs, ids + ["t-train-9"], codes, tab, target, 0.5)
    with pytest.raises(LeakageError):
        FitSet(("t-test-0",))


def test_subthresh_config_validation(tmp_path):
    from harness.config import ExpConfig

    d = yaml.safe_load((ROOT / "configs" / "experiments" / "E5.yaml").read_text())
    cfg = ExpConfig.load(ROOT / "configs" / "experiments" / "E5.yaml")
    assert cfg.subthresh.alpha_min == 0.5 and cfg.subthresh.g_alpha == 0.5
    d["subthresh"]["greedy_ks"] = [11]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="greedy_ks"):
        ExpConfig.load(p)


# --------------------------------------------------------------------------- tiny runner
def _tiny_cfg(tmp_path, name, mut):
    d = yaml.safe_load((ROOT / "configs" / "experiments" / f"{name}.yaml").read_text())
    mut(d)
    p = tmp_path / f"{name}_tiny.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.mark.slow
def test_tiny_e5_alpha_one_equals_features_only_and_greedy_saved(tmp_path):
    from harness import run as R
    from harness.schema import validate_per_pair

    def mut(d):
        d.update(name="tiny_test", cells=d["cells"][:1], bootstrap=20, smoke_pairs=5, conditions=["feat_A_errB"])
        d["prompt_sets"] = {"entity_track": d["prompt_sets"]["entity_track"]}
        d["subthresh"].update(alphas=[0.5, 1.0], greedy_ks=[1, 2])
    assert R.main(["--config", str(_tiny_cfg(tmp_path, "E5", mut)), "--profile", "dev", "--tiny", "--smoke"]) == 0
    run_dir = sorted((ROOT / "tests" / "_tiny_runs" / "E5").glob("*/manifest.json"), key=lambda q: q.stat().st_mtime)[-1].parent
    pp = pd.read_parquet(run_dir / "per_pair.parquet")
    assert not validate_per_pair(pp)
    sub = pp[pp.condition == "subthresh"]
    assert set(sub.source) == {f"alpha={a:.2f}|slot={s}" for a in (0.5, 1.0) for s in ("B", "meanPos")}
    one = sub[(sub.source == "alpha=1.00|slot=B") & (sub.r == 0)].set_index("pair_id")
    ref = pp[pp.condition == "feat_A_errB"].set_index("pair_id").loc[one.index]
    assert np.allclose(one.logit_A - one.logit_B, ref.logit_A - ref.logit_B, atol=1e-5)
    js = list((run_dir / "subthresh").glob("*.json"))
    assert js and json.loads(js[0].read_text())["latents"] is not None


@pytest.mark.slow
def test_tiny_e4_several_widths_per_layer_generic_l0_and_no_model_in_stage2(tmp_path, monkeypatch):
    from harness import run as R

    def mut(d):
        cells = [c for c in d["cells"] if c["task"] == "entity_track"][:3]          # 16k + two 32k variants
        d.update(name="tiny_test", cells=cells, bootstrap=20, smoke_pairs=4)
        d["prompt_sets"] = {"entity_track": d["prompt_sets"]["entity_track"]}
        d["directions"].update(ranks=[0, 1])
    seen = []
    orig_stage2, orig_run2, orig_init = R.S.stage2, R.Runner.run_stage2, R.Runner.__init__
    holder = {}

    def init(self, *a, **k):                                                    # own cache: stage 2 must really run
        orig_init(self, *a, **k)
        self.cache = Cache(tmp_path / "cache")
    monkeypatch.setattr(R.Runner, "__init__", init)

    def run2(self):
        holder["r"] = self
        return orig_run2(self)

    def stage2(*a, **k):
        seen.append(holder["r"].model is None)
        return orig_stage2(*a, **k)
    monkeypatch.setattr(R.Runner, "run_stage2", run2)
    monkeypatch.setattr(R.S, "stage2", stage2)
    assert R.main(["--config", str(_tiny_cfg(tmp_path, "E4", mut)), "--profile", "dev", "--tiny", "--smoke"]) == 0
    assert seen and all(seen)                                                   # the model was never loaded in stage 2
    run_dir = sorted((ROOT / "tests" / "_tiny_runs" / "E4").glob("*/manifest.json"), key=lambda q: q.stat().st_mtime)[-1].parent
    pp = pd.read_parquet(run_dir / "per_pair.parquet")
    assert pp.sae.nunique() == 3
    cells = json.loads((run_dir / "manifest.json").read_text())["cells"]
    assert all("generic" in c and c["generic"]["rows"] > 0 for c in cells)

"""T1: golden-fixture tests against real pilot tensors exported on RUN (RUN_REQUESTS R2). Skipped until
incoming/R2/ holds the unpacked export (manifest.json with kind golden_fixture)."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harness import metrics as M
from harness import sae as S
from harness import tasks as T
from harness.tolerances import CPU_VS_GOLDEN, U_COLUMN_ABS_COS

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.golden


def _golden_dir():
    for man in sorted((ROOT / "incoming" / "R2").glob("**/manifest.json")):
        try:
            if json.loads(man.read_text()).get("kind") == "golden_fixture" and not json.loads(man.read_text())["mock"]:
                return man.parent
        except Exception:  # noqa: BLE001
            continue
    return None


GOLD = _golden_dir()
if GOLD is None:
    pytest.skip("no golden export in incoming/R2 (RUN_REQUESTS R2 not returned yet)", allow_module_level=True)
MAN = json.loads((GOLD / "manifest.json").read_text())
CELLS = list(MAN["cells"].values())


def g(z, k, key):
    return torch.from_numpy(z[f"p{k:02d}/{key}"])


@pytest.fixture(scope="module", params=[c["task"] for c in CELLS])
def cell(request):
    info = next(c for c in CELLS if c["task"] == request.param)
    return info, np.load(GOLD / info["file"])


@pytest.fixture(scope="module")
def inventory():
    return json.loads((ROOT / "configs" / "sae_inventory.json").read_text())


def _sae(info, inventory):
    L = info["layer"]
    sid = S.SAEId(L, "16k", S.canonical_l0(inventory["inventory"][str(L)]["16k"]), revision=inventory["revision"])
    try:
        return sid, S.load_params(S.params_path(sid), L, "16k")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"16k SAE params unavailable on this machine: {type(e).__name__}")


@pytest.mark.parametrize("task", T.PILOT_TASKS)
def test_pilot_candidates_and_filter_reproduced(task):
    """Every candidate the pilot scored has the same ids in our prompt set, and the pilot rule applied to the pilot's
    own scores selects exactly the pilot's kept pairs."""
    f = GOLD / f"pairs_{task}.jsonl.gz"
    if not f.exists():
        pytest.skip(f"no dump for {task}")
    with gzip.open(f, "rt") as fh:
        dump = [json.loads(l) for l in fh]
    rows, meta = T.load_set(ROOT / "data" / "prompt_sets" / f"{task}_pilot.jsonl.gz")
    by_id = {r["pair_id"]: r for r in rows}
    for d in dump:
        if d.get("clean_ids"):
            r = by_id[d["pair_id"]]
            assert r["clean_ids"] == d["clean_ids"] and r["corrupt_ids"] == d["corrupt_ids"], d["pair_id"]
            assert (r["ans_A"], r["ans_B"], r["first_diff"]) == (d["clean_ans"], d["corrupt_ans"], d["first_diff"])
    scored = [d for d in dump if d["scores"]]
    if not scored:
        pytest.skip("dump has no filter scores")
    idx = [rows.index(by_id[d["pair_id"]]) for d in scored]
    sub = [rows[i] for i in idx]
    sc = {k: np.array([d["scores"][k] for d in scored]) for k in scored[0]["scores"]}
    top1_A = np.where(sc["clean_top1"], [r["ans_A"] for r in sub], -1)
    top1_B = np.where(sc["corrupt_top1"], [r["ans_B"] for r in sub], -1)
    fl = T.filter_flags(sub, top1_A, top1_B, sc["dl_clean"], sc["dl_corrupt"], meta["settings"]["filter_mode"])
    for split in ("train", "test"):
        want = [d["pair_id"] for d in sorted((d for d in dump if d["split"] == split), key=lambda d: d["split_rank"])]
        n = len(want)
        got = [sub[i]["pair_id"] for i in T.select(sub, fl, split, "f_pilot", n)]
        assert sorted(got) == sorted(want), f"{task}/{split}"


def test_sae_reproduces_golden_xhat_eps_and_codes(cell, inventory):
    info, z = cell
    _, sae = _sae(info, inventory)
    for k in range(len(info["golden_pairs"])):
        for tag in ("A", "B"):
            x = g(z, k, f"span_{tag}_pilot")
            xh, e, l0 = S.decompose(sae, x)
            assert torch.allclose(xh, g(z, k, f"xhat_{tag}"), rtol=CPU_VS_GOLDEN.rtol, atol=CPU_VS_GOLDEN.atol)
            assert torch.allclose(e, g(z, k, f"eps_{tag}"), rtol=CPU_VS_GOLDEN.rtol, atol=CPU_VS_GOLDEN.atol)
            codes = S.encode_sparse(sae, x, alpha_min=0.5)
            ref = set(zip(z[f"p{k:02d}/z_{tag}_row"].tolist(), z[f"p{k:02d}/z_{tag}_col"].tolist()))
            ours = set(zip(codes.row.tolist(), codes.col.tolist()))
            assert len(ours ^ ref) <= max(2, len(ref) // 1000)       # entries exactly at the 0.5 * threshold boundary
            assert torch.equal(codes.l0(), l0)


def test_golden_U_equals_committed_pilot_U(cell):
    info, z = cell
    pilot = np.load(ROOT / "results" / "paper_20261026_001341" / "paper_residue" / "tensors" /
                    f"exp3_U_{info['task']}_L{info['layer']}_16k.npz")
    for k_g, k_p in (("unsupervised_svd", "unsupervised_svd"), ("supervised_centroid_svd", "supervised_centroid_svd")):
        U, P = z[f"U/{k_g}"], pilot[k_p][:, : z[f"U/{k_g}"].shape[1]]
        cos = np.abs((U * P).sum(0) / (np.linalg.norm(U, axis=0) * np.linalg.norm(P, axis=0)))
        assert cos.min() > U_COLUMN_ABS_COS, (k_g, cos)


def test_metrics_on_golden_logits(cell):
    info, z = cell
    gp = info["golden_pairs"]
    for k, p in enumerate(gp):
        cand = z[f"p{k:02d}/cand_ids"]
        for name in ("clean_A", "clean_B", "feat_A_errB"):
            cl = z[f"p{k:02d}/logits/{name}/cand"]
            iA, iB = list(cand).index(p["clean_ans"]), list(cand).index(p["corrupt_ans"])
            assert cl[iA] - cl[iB] == pytest.approx(p["D_batch1"][name], abs=1e-4)
    o = lambda name: M.Out(np.array([z[f"p{k:02d}/logits/{name}/cand"][list(z[f'p{k:02d}/cand_ids']).index(p['clean_ans'])] for k, p in enumerate(gp)]),
                           np.array([z[f"p{k:02d}/logits/{name}/cand"][list(z[f'p{k:02d}/cand_ids']).index(p['corrupt_ans'])] for k, p in enumerate(gp)]),
                           np.array([float(z[f"p{k:02d}/logits/{name}/lse"]) for k in range(len(gp))]),
                           np.array([int(z[f"p{k:02d}/logits/{name}/topk_ids"][0]) for k in range(len(gp))]),
                           np.stack([z[f"p{k:02d}/logits/{name}/cand"] for k in range(len(gp))]))
    ansA = np.array([p["clean_ans"] for p in gp])
    ansB = np.array([p["corrupt_ans"] for p in gp])
    m = M.per_condition(o("full_A"), o("clean_A"), o("clean_B"), ansA, ansB)
    assert m["nldr"] == pytest.approx(1.0, abs=0.05)
    full = z["p00/logits_full_clean_A"]
    assert float(z["p00/logits/clean_A/lse"]) == pytest.approx(float(torch.logsumexp(torch.from_numpy(full), -1)), abs=1e-3)

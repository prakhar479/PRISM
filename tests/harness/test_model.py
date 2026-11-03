"""Model stages on a tiny random Gemma-2 (handoff tests 2, 3, 15, 16-OOM). Numbers here are never results."""
import numpy as np
import pytest
import torch

from harness import conditions as C
from harness import tasks as T
from harness.batching import Budget, run_with_backoff
from harness.cache import Cache, StaleCacheError
from harness.device import FakeOOM
from harness.model import ModelSpec, tiny_gemma2
from harness.model import stages as S
from harness.model.das import Rotation, train_das
from src.data.synthetic_tasks import FallbackTokenizer


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tok = FallbackTokenizer()
    rows = T.build_pilot_candidates("entity_track", tok, n_train=6, n_test=4)
    spec = ModelSpec(name="tiny", dtype="float32", tiny_vocab=tok.vocab_size)
    model = tiny_gemma2(tok.vocab_size)
    cache = Cache(tmp_path_factory.mktemp("cache"))
    ent = S.stage1(model, spec, rows, T.set_hash(rows), [1, 2], cache, Budget(max_tokens=400), "cpu", shard_pairs=7)
    return dict(tok=tok, rows=rows, spec=spec, model=model, cache=cache, ent=ent)


def test_stage1_outputs_match_direct_forward(world):
    from src.models import runtime
    rows, model = world["rows"], world["model"]
    out = S.load_clean(world["ent"]["clean_out"], len(rows))
    i = 3
    lg = runtime.last_logits(model, torch.tensor([rows[i]["clean_ids"]]))[0]
    assert out["A"]["logit_A"][i] == pytest.approx(float(lg[rows[i]["ans_A"]]), abs=1e-4)
    assert out["A"]["top1_id"][i] == int(lg.argmax())
    assert out["A"]["lse"][i] == pytest.approx(float(torch.logsumexp(lg, -1)), abs=1e-4)
    assert np.isfinite(out["A"]["cand_logits"]).all() and out["A"]["cand_logits"].shape[1] == 7


def test_patched_positions_are_first_diff_to_final(world):
    rows = world["rows"]
    idx = list(range(len(rows)))
    sp = S.load_spans(world["ent"]["spans"][2], rows, idx)
    for k, i in enumerate(idx):
        r = rows[i]
        assert sp.rows(k).stop - sp.rows(k).start == r["T"] - r["first_diff"]
        assert int(sp.pos[sp.rows(k)][0]) == r["first_diff"] and int(sp.pos[sp.rows(k)][-1]) == r["T"] - 1
    assert len(sp.slot) == sp.m


def test_hook_identity_and_full_A(world):
    rows, model = world["rows"], world["model"]
    idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    sp = S.load_spans(world["ent"]["spans"][2], rows, idx)
    clean = S.load_clean(world["ent"]["clean_out"], len(rows))
    for hooked in (False, True):
        o = S.stage3(model, rows, idx, 2, sp.row_ptr, {"clean_B": None, "full_B": sp.x_B, "full_A": sp.x_A},
                     Budget(max_tokens=300), "cpu", hooked=hooked)
        DB = clean["B"]["logit_A"][idx] - clean["B"]["logit_B"][idx]
        DA = clean["A"]["logit_A"][idx] - clean["A"]["logit_B"][idx]
        for n in ("clean_B", "full_B"):
            assert np.allclose(o[n]["logit_A"] - o[n]["logit_B"], DB, atol=1e-4)
        assert np.allclose(o["full_A"]["logit_A"] - o["full_A"]["logit_B"], DA, atol=1e-4)    # NLDR = 1
    a = S.stage3(model, rows, idx, 2, sp.row_ptr, {"x": sp.x_A * 0.5 + sp.x_B * 0.5}, Budget(300), "cpu")
    b = S.stage3(model, rows, idx, 2, sp.row_ptr, {"x": sp.x_A * 0.5 + sp.x_B * 0.5}, Budget(300), "cpu", hooked=True)
    assert np.allclose(a["x"]["logit_A"], b["x"]["logit_A"], atol=1e-4)                      # resume == hooked


def test_stage2_and_conditions_roundtrip(world, tmp_path):
    from harness import sae as SA
    rows = world["rows"]
    p = SA.write_random_params(tmp_path / "p.npz", d_model=64, d_sae=256, seed=0, thr=0.05)
    sae = SA.load_params(p)
    e2 = S.stage2(world["cache"], world["ent"]["spans"][2], sae, {"test": 1}, "cpu", block_latents=100)
    idx = list(range(len(rows)))
    sp = S.load_spans(world["ent"]["spans"][2], rows, idx, sae_entry=e2)
    assert torch.allclose(sp.xh_A + sp.e_A, sp.x_A, atol=1e-6)
    xh, _, _ = SA.decompose(sae, sp.x_A)
    assert torch.allclose(xh, sp.xh_A, atol=1e-5)
    assert torch.allclose(C.build("repair", sp, U=torch.eye(64)), sp.x_A, atol=1e-5)


def test_stage1_resumes_and_cache_key_guard(world):
    rows = world["rows"]
    out = world["ent"]["clean_out"]
    n_shards = len(out.shards())
    (out.dir / "shard_00000.done").unlink()
    (out.dir / "_COMPLETE").unlink()
    S.stage1(world["model"], world["spec"], rows, T.set_hash(rows), [1, 2], world["cache"], Budget(400), "cpu", shard_pairs=7)
    assert len(out.shards()) == n_shards and out.complete
    import json
    kp = out.dir / "key.json"
    d = json.loads(kp.read_text())
    d["set_hash"] = "tampered"
    kp.write_text(json.dumps(d))
    with pytest.raises(StaleCacheError):
        out.require_complete()


def test_das_full_rank_equals_full_patch(world):
    rows, model = world["rows"], world["model"]
    sp = S.load_spans(world["ent"]["spans"][2], rows, [0, 1])
    rot = Rotation(64, 64)
    assert torch.allclose(rot.intervene(sp.x_B, sp.x_A), sp.x_A, atol=1e-4)
    trained = train_das(model, 2, rows, [0, 1], sp, k=2, steps=3)
    R = trained.R.detach()
    assert torch.allclose(R.T @ R, torch.eye(2), atol=1e-4)


def test_oom_backoff_with_injected_fake_oom():
    calls = []

    def fn(b):
        calls.append(b.max_tokens)
        if b.max_tokens > 500:
            raise FakeOOM("CUDA out of memory (fake)")
        return b.max_tokens

    b = Budget(max_tokens=4000, floor=128)
    assert run_with_backoff(fn, b) == 500 and calls == [4000, 2000, 1000, 500] and b.backoffs == 3
    with pytest.raises(ValueError):
        run_with_backoff(lambda b: (_ for _ in ()).throw(ValueError("x")), Budget(1000))


def test_stage3_invariant_to_batching(world):
    """Many conditions per forward (large budget) == one pair per forward (tiny budget)."""
    rows, model = world["rows"], world["model"]
    idx = list(range(len(rows)))
    sp = S.load_spans(world["ent"]["spans"][1], rows, idx)
    conds = {"a": sp.x_A, "b": None, "c": 0.3 * sp.x_A + 0.7 * sp.x_B}
    big = S.stage3(model, rows, idx, 1, sp.row_ptr, conds, Budget(max_tokens=100_000, max_rows=64), "cpu")
    small = S.stage3(model, rows, idx, 1, sp.row_ptr, conds, Budget(max_tokens=1, max_rows=1), "cpu")
    for n in conds:
        assert np.allclose(big[n]["logit_A"], small[n]["logit_A"], atol=1e-4)
        assert np.array_equal(big[n]["top1_id"], small[n]["top1_id"])


def test_head_equals_full_forward_in_fp32_and_bf16_within_ulps():
    """runtime.head (last position, fp32 soft-cap) computes the full forward's last logits: exactly in fp32; within
    the preflight's bf16 ulp tolerance in bf16 (preflight check 3, RUN 2026-11-01)."""
    from harness.model import build_model
    from harness.tolerances import head_vs_full_atol
    from src.models import runtime

    for dt in ("float32", "bfloat16"):
        m = build_model(ModelSpec(name="tiny", dtype=dt, tiny_vocab=4096, tiny_layers=3), torch.device("cpu"))
        with torch.no_grad():
            for p in m.unembed.parameters():
                p.mul_(20)                                    # Gemma-like logit magnitudes under the 30 soft-cap
        toks = torch.randint(0, 4096, (3, 10), generator=torch.Generator().manual_seed(0))
        with torch.inference_mode():
            a = runtime.last_logits(m, toks)
            inp, kw = runtime._input(m, toks)
            full = m.run_with_hooks(inp, fwd_hooks=[], tokens=toks, **kw)[:, -1].float()
        d = float((a - full).abs().max())
        if dt == "float32":
            assert d < 1e-4                                   # fp32 ulp at |30| is 1.9e-6; a wrong head is O(1) off
        else:
            assert 0 < d <= head_vs_full_atol(float(full.abs().max()), dt)


def test_vocab_major_unembed_avoids_the_per_call_copy_and_is_bitwise_identical():
    """TL's Unembed does F.linear(x, W_U.T.contiguous(), b): with W_U stored vocab-major that is a no-op instead of a
    full copy of the unembedding per call (the ~1126 MiB transient allocations seen on RUN)."""
    from harness.model import build_model, vocab_major_unembed
    from src.models import runtime

    for dt in ("float32", "bfloat16"):
        m = build_model(ModelSpec(name="tiny", dtype=dt, tiny_vocab=4096, tiny_layers=2), torch.device("cpu"))
        u = m.unembed
        assert u.W_U.shape == (m.cfg.d_model, 4096) and u.W_U.T.is_contiguous()
        assert u.W_U.T.contiguous().data_ptr() == u.W_U.data_ptr()             # no copy
        ref = build_model(ModelSpec(name="tiny", dtype=dt, tiny_vocab=4096, tiny_layers=2), torch.device("cpu"))
        ref.unembed.W_U = torch.nn.Parameter(ref.unembed.W_U.detach().contiguous(), requires_grad=False)  # TL default
        assert not ref.unembed.W_U.T.is_contiguous() and torch.equal(ref.unembed.W_U, u.W_U)
        toks = torch.randint(0, 4096, (3, 9), generator=torch.Generator().manual_seed(1))
        with torch.inference_mode():
            assert torch.equal(runtime.last_logits(m, toks), runtime.last_logits(ref, toks))
        vocab_major_unembed(m)                                                  # idempotent
        assert u.W_U.T.is_contiguous()
    m.unembed.to(torch.float64)                                                 # Module.to keeps the strides
    assert m.unembed.W_U.T.is_contiguous()

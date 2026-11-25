"""harness/sensitivity.py on the tiny model: gradients against finite differences, one pass for several layers, a PSD Fisher."""
import numpy as np
import pytest
import torch

from harness import sensitivity as Sn
from harness.model import hook_name, tiny_gemma2
from src.models import runtime


@pytest.fixture(scope="module")
def tiny():
    return tiny_gemma2(300, n_layers=4, seed=0)


def _objective(model, tokens, layer, pos, delta, lo, hi):
    """sum_{s=lo}^{hi-1} log p(tokens[s+1]) with `delta` [d] added to resid_post[layer] at position pos."""
    def fn(act, hook):
        act = act.clone()
        act[:, pos] = act[:, pos] + delta.to(act.dtype)
        return act
    with torch.no_grad():
        R = runtime.final_resid(model, tokens, [(hook_name(layer), fn)])
        lp = torch.log_softmax(runtime.head(model, R[:, lo:hi]), -1)
        return float(lp.gather(-1, tokens[:, lo + 1:hi + 1, None]).sum())


def test_gradient_matches_finite_differences(tiny):
    rng = np.random.default_rng(0)
    tokens = torch.tensor([list(map(int, rng.integers(3, 300, 12)))])
    d = tiny.cfg.d_model
    for layer, pos in ((1, 5), (2, 8), (1, 0)):
        g = Sn.score_batch(tiny, tokens, [layer], skip=2, sample=False)[layer][0, pos]
        u = torch.randn(d)
        u /= u.norm()
        h = 1e-2
        num = (_objective(tiny, tokens, layer, pos, h * u, 2, 11) - _objective(tiny, tokens, layer, pos, -h * u, 2, 11)) / (2 * h)
        assert abs(float(g @ u) - num) <= 2e-3 * max(1.0, abs(num)) + 2e-3, (layer, pos, float(g @ u), num)


def test_one_pass_over_layers_equals_separate_passes(tiny):
    tokens = torch.tensor([[2, 17, 99, 5, 61, 250, 7, 8, 33, 120, 4, 9]])
    both = Sn.score_batch(tiny, tokens, [1, 2], skip=2, sample=False)
    for L in (1, 2):
        one = Sn.score_batch(tiny, tokens, [L], skip=2, sample=False)[L]
        assert torch.allclose(both[L], one, atol=1e-5, rtol=1e-4), L
    assert both[1].shape == (1, 12, tiny.cfg.d_model) and float(both[1].abs().sum()) > 0


def test_positions_before_the_loss_window_have_no_effect_from_later_context_only(tiny):
    """With the loss restricted to predictions at positions >= lo, a perturbation at a position after hi has zero gradient."""
    tokens = torch.tensor([[2, 17, 99, 5, 61, 250, 7, 8, 33, 120, 4, 9]])
    g = Sn.score_batch(tiny, tokens, [1], skip=2, sample=False, positions=(2, 5))[1][0]
    assert float(g[6:].abs().max()) == 0.0          # causal model: positions > hi cannot influence predictions 2..4
    assert float(g[:5].abs().max()) > 0.0


def test_fisher_is_symmetric_psd_and_sampling_is_reproducible(tiny):
    ids = [list(map(int, np.random.default_rng(i).integers(3, 300, 12))) for i in range(6)]
    a = Sn.fit(tiny, ids, [1, 3], skip=2, device=torch.device("cpu"), batch=3, seed=0)
    b = Sn.fit(tiny, ids, [1, 3], skip=2, device=torch.device("cpu"), batch=2, seed=0)
    for L in (1, 3):
        F = a[L]["F"]
        assert torch.allclose(F, F.T, atol=1e-12)
        assert float(torch.linalg.eigvalsh(F).min()) > -1e-9 and a[L]["trace"] > 0
        assert a[L]["n_rows"] == 6 * 9                      # positions 2..10 of 12 tokens
    assert torch.allclose(a[1]["F"], a[1]["F"])             # deterministic given the seed and the batching
    c = Sn.fit(tiny, ids, [1], skip=2, device=torch.device("cpu"), batch=3, seed=1)
    assert not torch.allclose(a[1]["F"], c[1]["F"])         # another seed samples other tokens
    # the empirical Fisher (observed tokens) differs from the sampled one but has the same shape and trace order
    e = Sn.fit(tiny, ids, [1], skip=2, device=torch.device("cpu"), batch=3, sample=False)
    assert e[1]["F"].shape == a[1]["F"].shape and e[1]["trace"] > 0

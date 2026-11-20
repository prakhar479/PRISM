"""SAE math (handoff tests 1, 13, 14-part), inventory parsing and L0 matching."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from harness import sae as S
from harness.tolerances import BLOCKWISE, RECON_IDENTITY

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def small_sae(tmp_path_factory):
    p = S.write_random_params(tmp_path_factory.mktemp("sae") / "params.npz", d_model=32, d_sae=200, seed=3)
    return S.load_params(p, layer=1, width="200")


@pytest.fixture(scope="module")
def X():
    g = torch.Generator().manual_seed(0)
    return torch.randn(57, 32, generator=g) * 2


def test_recon_identity_all_paths(small_sae, X):
    for blk in (None, 7, 64, 200):
        xh, e, l0 = S.decompose(small_sae, X, block_latents=blk)
        assert torch.allclose(xh + e, X, atol=RECON_IDENTITY.atol)
        assert (l0 > 0).any()


def test_blockwise_equals_oneshot(small_sae, X):
    a = S.decompose(small_sae, X)
    for blk in (1, 13, 100):
        b = S.decompose(small_sae, X, block_latents=blk)
        assert torch.allclose(a[0], b[0], rtol=BLOCKWISE.rtol, atol=BLOCKWISE.atol)
        assert torch.equal(a[2], b[2])


def test_matches_pilot_jumprelu(small_sae, X):
    """Same numbers as the pilot's JumpReLUSAE.decompose (src/, frozen)."""
    xh_p, e_p, l0_p = small_sae.decompose(X)
    xh, e, l0 = S.decompose(small_sae, X)
    assert torch.allclose(xh, xh_p, atol=1e-5) and torch.equal(l0.long(), l0_p.long())


def test_reference_formula(small_sae, X):
    We = small_sae.W_enc_T.T
    z = X @ We + small_sae.b_enc
    f = torch.where(z > small_sae.threshold, z, torch.zeros_like(z))
    ref = f @ small_sae.W_dec + small_sae.b_dec
    assert torch.allclose(S.decompose(small_sae, X)[0], ref, atol=1e-5)


def test_sparse_codes_and_decode(small_sae, X):
    codes = S.encode_sparse(small_sae, X, alpha_min=0.5)
    xh, _, l0 = S.decompose(small_sae, X)
    assert torch.equal(codes.l0(), l0)
    full = S.decode_entries(small_sae, codes, codes.active) + small_sae.b_dec
    assert torch.allclose(full, xh, atol=1e-5)
    assert (codes.z >= 0.5 * codes.threshold).all() and (codes.z > 0).all()
    blk = S.encode_sparse(small_sae, X, alpha_min=0.5, block_latents=17)
    assert len(blk.z) == len(codes.z)
    sub = codes.subset_rows(torch.tensor([3, 5]))
    assert sub.n_rows == 2 and torch.equal(sub.l0(), l0[[3, 5]])


def test_parse_tree_fixture():
    paths = ["layer_12/width_16k/average_l0_82/params.npz", "layer_12/width_16k/average_l0_22",
             "layer_12/width_1m/average_l0_107/params.npz", "layer_12/width_262k/average_l0_121",
             "layer_3/width_65k/average_l0_9/x", "README.md", "layer_12/width_16k"]
    inv = S.parse_tree(paths)
    assert inv == {3: {"65k": [9]}, 12: {"16k": [22, 82], "262k": [121], "1m": [107]}}
    assert list(inv[12]) == ["16k", "262k", "1m"]


def test_match_l0():
    m = S.match_l0(82, [11, 21, 36, 67, 121, 243])
    assert m["closest"] == 67 and m["below"] == 67 and m["above"] == 121
    assert S.match_l0(500, [10, 20])["above"] is None


def test_canonical_rule_matches_every_pilot_sae():
    inv = json.loads((ROOT / "configs" / "sae_inventory.json").read_text())["inventory"]
    pilot = yaml.safe_load((ROOT / "configs" / "pilot_saes.yaml").read_text())["saes"]
    for s in pilot:
        if s["pilot_release"].endswith("canonical"):
            assert S.canonical_l0(inv[str(s["layer"])][s["width"]]) == s["l0"], s


def test_saeid_paths():
    i = S.SAEId(12, "16k", 82, revision="abc")
    assert i.path_in_repo == "layer_12/width_16k/average_l0_82/params.npz" and i.d_sae == 16384 and i.key == "L12_16k_l082"

"""Artifact check: every DAS rotation starts from the same seed-0 init (harness/model/das.py Rotation: QR of randn(d,k) with a
seed-0 generator, then torch's orthogonal parametrization). How far did training move each rotation from its init?"""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, f"/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/hw")
from geo_lib import *
import torch
from harness.model.das import Rotation
cells = [("E9", "add_chain_L18_L18_16k_l074"), ("E9", "entity_track_L12_L12_16k_l082"), ("E10", "people_cities_L12_L12_16k_l082__fit-pc_en"),
         ("E16", "add_chain_L12_L12_16k_l082"), ("E16", "add_chain_L14_L14_16k_l084"), ("E16", "add_chain_L16_L16_16k_l078"), ("E16", "add_chain_L20_L20_16k_l071")]
inits = {k: Rotation(2304, k, 0).R.detach().double().numpy() for k in (1, 2, 4, 8)}
print("init orthonormal check", {k: round(float(np.abs(v.T @ v - np.eye(k)).max()), 6) for k, v in inits.items()})
for e, s in cells:
    Z, J = load(e, s)
    out = {}
    for k in sorted([c for c in Z if c.startswith("das")]):
        kk = int(k[3:]); D = Z[k].astype(np.float64)
        out[k] = (round(capture(D, inits[kk]), 3), round(top(D, inits[kk]), 3), J[k]["meta"].get("train_interchange_acc"))
    print(f"{e:4s} {s[:32]:32s}", out)
print("(capture of DAS in its own init span; 1 = did not move, Haar ~k/2304)")

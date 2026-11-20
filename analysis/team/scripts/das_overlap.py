"""Are the 'feature' part (span of the top-20 OMP atoms) and the 'error' part (pca_pooled 32/64) of a DAS learned subspace the
same directions? For each rotation: capture in atoms20, in pcaR, in their joint span, and the excess of joint over the larger part
(if ~sum, the two are disjoint pieces; if ~max, overlapping). Also: how much of the atoms20 span itself lies in pcaR. Post hoc."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from das_lib import *
import pandas as pd
pd.set_option("display.width", 250)
rows = []
for exp, stem, L, cell, ests in [("E17", "add_chain_L18_L18_16k_l074", 18, "RS18", ["das1", "das1_e10", "das1_s2", "das2", "das2_e30", "das4", "das4_s1", "das8", "das8_s1", "das8_e10"]),
                                 ("E18", "add_chain_L18_L18_16k_l074", 18, "RS18", ["das16"]),
                                 ("E18", "entity_track_L12_L12_16k_l082", 12, "OT12", ["das1", "das4", "das8", "das16"]),
                                 ("E18", "people_cities_L12_L12_16k_l082__fit-pc_en", 12, "PC12", ["das8", "das16"])]:
    Z, J = load(exp, stem); D = decoder(L); R = 64 if Z["pca_pooled"].shape[1] >= 64 else 32
    for k in ests:
        Lr, _, _ = learned(Z[k], k); _, sel = subspace_omp(Lr, D, ms=(20,))
        A = D[sel].T; P = Z["pca_pooled"][:, :R]
        ca, cp, cj = capture(Lr, A), capture(Lr, P), capture(Lr, np.hstack([A, P]))
        rows.append(dict(cell=cell, est=k, pcaR=R, in_atoms20=ca, in_pca=cp, in_joint=cj, sum_parts=ca + cp,
                         atoms20_in_pca=capture(A, P), haar_atoms20_in_pca=R / 2304))
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/overlap.csv", index=False); print(d.round(3).to_string(index=False))

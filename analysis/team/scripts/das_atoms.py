"""Feature mediation geometry of DAS subspaces (E17 RS L18 all fits; E18 RS L18, OT L12, PC L12; PC L16 has no decoder on disk).
For each rotation: (a) energy share of the subspace inside the span of the best m 16k-decoder atoms (greedy subspace OMP), for the
full rotation U and for its learned component (own init projected out); (b) capture in the same run's error subspace
(pca_pooled first r columns, answer_sep); (c) top atoms and the recurrence of the L18 atoms named in geometry.md.
Null: Haar random k-subspaces (n=40 per k) through the same OMP. Post hoc, exploratory.
Outputs: das_out/atoms_rot.csv, atoms_top.csv, atoms_null.csv, atoms_recurrent.csv."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from das_lib import *
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
MS = (1, 2, 5, 10, 20)

cells = [("E17", "add_chain_L18_L18_16k_l074", 18, "RS18"), ("E18", "add_chain_L18_L18_16k_l074", 18, "RS18"),
         ("E18", "entity_track_L12_L12_16k_l082", 12, "OT12"), ("E18", "people_cities_L12_L12_16k_l082__fit-pc_en", 12, "PC12"),
         ("E18", "people_cities_L16_L16_16k_l078__fit-pc_en", 16, "PC16")]
rows, tops, rec = [], [], []
c1 = np.load(f"{OUT}/rs18_das1_consensus.npy")
done = set()
for exp, stem, L, cell in cells:
    Z, J = load(exp, stem)
    D = decoder(L) if L in DEC else None
    for k in [k for k in Z if k.startswith("das")]:
        if (cell, k) in done:   # E17 and E18 share the plain dasK (bit-identical); keep one
            continue
        done.add((cell, k))
        U = orth(Z[k]); Lr, s, cap = learned(Z[k], k); kk = U.shape[1]
        r = dict(exp=exp, cell=cell, est=k, k=kk, train_acc=J[k]["meta"]["train_interchange_acc"], init_capture=cap,
                 learned_dims=Lr.shape[1])
        for nm, B in (("pca", Z["pca_pooled"]), ("as", Z.get("answer_sep"))):
            if B is None:
                continue
            for rr in (kk, 8, 32, 64):
                if rr <= B.shape[1]:
                    r[f"U_in_{nm}{rr if rr != kk else 'k'}"] = capture(U, B[:, :rr])
                    r[f"Lrn_in_{nm}{rr if rr != kk else 'k'}"] = capture(Lr, B[:, :rr])
        if D is not None:
            o, sel = subspace_omp(U, D, ms=MS); ol, sell = subspace_omp(Lr, D, ms=MS)
            for m in MS:
                r[f"U_omp{m}"] = o[m]; r[f"Lrn_omp{m}"] = ol[m]
            sh = atom_share(Lr, D, sell[:5])
            r["top5_learned"] = " ".join(f"{a}:{sh[a]:.2f}" for a in sell[:5])
            r["maxcos_learned_atom"] = float(np.sqrt(((D @ Lr) ** 2).sum(1).max()))
            for rank_, a in enumerate(sell[:10]):
                tops.append(dict(cell=cell, est=k, rank=rank_, atom=a, cos2_learned=float(((Lr.T @ D[a]) ** 2).sum()),
                                 cos2_U=float(((U.T @ D[a]) ** 2).sum())))
            if L == 18:
                allsh = ((D @ Lr) ** 2).sum(1); order = np.argsort(-allsh)
                pos = {int(a): i for i, a in enumerate(order[:2000])}
                for a in RECURRENT_L18:
                    rec.append(dict(est=k, train_acc=r["train_acc"], atom=a, cos2_learned=float(allsh[a]),
                                    rank_among_16k=pos.get(a, ">2000"), cos2_U=float(((U.T @ D[a]) ** 2).sum())))
        if cell == "RS18":
            r["consensus_in_learned"] = float(((Lr.T @ c1) ** 2).sum())
        rows.append(r)
rot = pd.DataFrame(rows); rot.to_csv(f"{OUT}/atoms_rot.csv", index=False)
pd.DataFrame(tops).to_csv(f"{OUT}/atoms_top.csv", index=False)
rec = pd.DataFrame(rec); rec.to_csv(f"{OUT}/atoms_recurrent.csv", index=False)

# consensus direction
D18 = decoder(18); oc, sc = subspace_omp(c1[:, None], D18, ms=MS)
print("RS18 das1 consensus: OMP", {m: round(v, 3) for m, v in oc.items()}, "top atoms", [(a, round(float((D18[a] @ c1) ** 2), 3)) for a in sc[:8]])

# null
nul = []
rng = np.random.default_rng(7)
for L in (12, 18):
    D = decoder(L)
    for k in (1, 2, 4, 8, 16):
        for i in range(40):
            Q = orth(rng.standard_normal((2304, k))); o, _ = subspace_omp(Q, D, ms=MS)
            nul.append(dict(layer=L, k=k, **{f"omp{m}": o[m] for m in MS}, maxcos=float(np.sqrt(((D @ Q) ** 2).sum(1).max()))))
nul = pd.DataFrame(nul); nq = nul.groupby(["layer", "k"]).quantile(.99).reset_index(); nq.to_csv(f"{OUT}/atoms_null.csv", index=False)
print("== Haar null p99 (n=40 per k)"); print(nq.round(3).to_string(index=False))

cols = ["exp", "cell", "est", "k", "train_acc", "init_capture", "U_omp1", "U_omp5", "U_omp20", "Lrn_omp1", "Lrn_omp5", "Lrn_omp20",
        "maxcos_learned_atom", "Lrn_in_pcak", "Lrn_in_pca8", "Lrn_in_pca32", "Lrn_in_pca64", "U_in_pca32", "U_in_pca64", "Lrn_in_as32",
        "consensus_in_learned", "top5_learned"]
print("== per rotation"); print(rot[[c for c in cols if c in rot]].round(3).to_string(index=False))
print("== recurrent L18 atoms: cos^2 with learned span (rank among 16k)")
rec["cell_txt"] = rec.apply(lambda x: f"{x.cos2_learned:.2f}({x.rank_among_16k})", axis=1)
print(rec.pivot(index="est", columns="atom", values="cell_txt").to_string())

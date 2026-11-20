"""E18 high-rank geometry (post hoc, exploratory). Cells: RS L18, OT L12, PC L12, PC L16 (16k SAEs).
1. evr spectrum: only the first 8 values are saved (harness/run.py truncates evr[:8]); higher-rank spectra are not recoverable.
2. DAS-k (k=1..16) with the shared init removed (make_final das_learned): init capture, train acc, principal cosines of the learned
   component with pca_pooled / answer_sep at m = k, 16, 64 (top cos, Haar p99 by MC for (dim, m)); capture of learned DAS in
   error-U m=64 (Haar m/2304); raw DAS for reference. Nesting: learned das_k inside learned das16.
3. Error-subspace overlap OT L12 vs PC L12 (and PC L12 vs L16, RS18 vs OT12 as a cross-task/-layer reference) at r=1..64.
Out: team/geometry_out/e18_das_vs_err.csv, e18_err_overlap.csv, e18_das_nesting.csv"""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from geometry_lib import *
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
cells = {"RS18": "add_chain_L18_L18_16k_l074", "OT12": "entity_track_L12_L12_16k_l082",
         "PC12": "people_cities_L12_L12_16k_l082__fit-pc_en", "PC16": "people_cities_L16_L16_16k_l078__fit-pc_en"}
Z, J = {}, {}
for c, s in cells.items():
    Z[c], J[c] = load("E18", s)
    print(c, "evr pca[:8]", np.round(J[c]["pca_pooled"]["evr"], 3), "cum8", round(sum(J[c]["pca_pooled"]["evr"]), 3))
rows, Lr = [], {}
for c in cells:
    for k in (1, 2, 4, 8, 16):
        est = f"das{k}"; U = Z[c][est]; acc = J[c][est]["meta"]["train_interchange_acc"]
        L, cap, sv = das_learned(U, est); Lr[(c, k)] = L
        row = dict(cell=c, das=est, train_acc=acc, valid=acc >= 0.5, init_capture=cap, learned_dims=L.shape[1])
        for e in ("pca_pooled", "answer_sep"):
            for m in sorted({k, 16, 64}):
                pc = pcos(L, Z[c][e][:, :m])
                row[f"{e[:3]}_top_m{m}"] = float(pc[0]); row[f"{e[:3]}_p99_m{m}"] = haar_top_p99(L.shape[1], m)
            row[f"{e[:3]}_raw_top_m{k}"] = top(U, Z[c][e][:, :k])
            row[f"{e[:3]}_cap_m64"] = capture(L, Z[c][e][:, :64]); row[f"{e[:3]}_cap_m16"] = capture(L, Z[c][e][:, :16])
            row[f"{e[:3]}_all_cos_m{k}"] = " ".join(f"{x:.2f}" for x in pcos(L, Z[c][e][:, :k]))
        rows.append(row)
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/e18_das_vs_err.csv", index=False)
print("capture nulls: m=16", round(16 / D, 4), "m=64", round(64 / D, 4))
print(d.drop(columns=[x for x in d.columns if "all_cos" in x]).round(3).to_string(index=False))
print(d[["cell", "das", "pca_all_cos_m" + "1"]].head(0))
for r in d.itertuples():
    print(r.cell, r.das, "pca principal cos (learned vs pca r=k):", getattr(r, f"pca_all_cos_m{int(r.das[3:])}"), "| ans:", getattr(r, f"ans_all_cos_m{int(r.das[3:])}"))
# nesting: is learned das_k inside learned das16 (and das8)?
nest = []
for c in cells:
    for k in (1, 2, 4, 8):
        for K in (8, 16):
            if K <= k: continue
            nest.append(dict(cell=c, k=k, K=K, capture_k_in_K=capture(Lr[(c, k)], Lr[(c, K)]), top=top(Lr[(c, k)], Lr[(c, K)]),
                             haar_capture=Lr[(c, K)].shape[1] / D, p99=haar_top_p99(Lr[(c, k)].shape[1], Lr[(c, K)].shape[1])))
n = pd.DataFrame(nest); n.to_csv(f"{OUT}/e18_das_nesting.csv", index=False); print(n.round(3).to_string(index=False))
# error-subspace overlap across cells
ov = []
for a, b in [("OT12", "PC12"), ("PC12", "PC16"), ("OT12", "RS18"), ("PC12", "RS18")]:
    for e in ("pca_pooled", "answer_sep"):
        for r in (1, 2, 4, 8, 16, 32, 64):
            pc = pcos(Z[a][e][:, :r], Z[b][e][:, :r])
            ov.append(dict(a=a, b=b, est=e, r=r, top=float(pc[0]), mean_cos2=float(np.mean(pc ** 2)), n_above_p99=int((pc > haar_top_p99(r, r)).sum()),
                           p99=haar_top_p99(r, r), haar_mean_cos2=r / D))
    for k in (8, 16):   # learned DAS OT vs PC
        ov.append(dict(a=a, b=b, est=f"das{k}_learned", r=k, top=top(Lr[(a, k)], Lr[(b, k)]), mean_cos2=float(np.mean(pcos(Lr[(a, k)], Lr[(b, k)]) ** 2)),
                       p99=haar_top_p99(Lr[(a, k)].shape[1], Lr[(b, k)].shape[1]), haar_mean_cos2=k / D))
o = pd.DataFrame(ov); o.to_csv(f"{OUT}/e18_err_overlap.csv", index=False); print(o.round(3).to_string(index=False))

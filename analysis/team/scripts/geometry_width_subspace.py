"""Error-direction geometry across SAE widths (E18 16k [64 cols], E4 32k-524k [32 cols]; E20's U are bit-identical to E4's
for the same SAE and E9's are the first 32 columns of E18's - checked here). Post hoc, exploratory.
Per (task, wide SAE, estimator): top principal cosine with the 16k U at r = 1,2,4,8,16,32 (Haar p99 from geometry_lib MC),
capture of 16k r1/r4/r8 in the wide r32, evr1 and evr sum(1..8) (only 8 evr values are saved by harness/run.py).
Out: team/geometry_out/width_subspace.csv"""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from geometry_lib import *
import glob as g, pandas as pd
# identity checks
for t, L, s in [("add_chain", 18, "65k_l064"), ("entity_track", 12, "65k_l072"), ("entity_track", 12, "262k_l067")]:
    a, _ = load("E20", f"{t}_L{L}_L{L}_{s}"); b, _ = load("E4", f"{t}_L{L}_L{L}_{s}")
    print("E20 vs E4", t, s, {e: float(np.abs(a[e] - b[e]).max()) for e in ("pca_pooled", "answer_sep")})
for t, L, s in [("add_chain", 18, "16k_l074"), ("entity_track", 12, "16k_l082")]:
    a, _ = load("E9", f"{t}_L{L}_L{L}_{s}"); b, _ = load("E18", f"{t}_L{L}_L{L}_{s}")
    print("E9 vs E18[:, :32]", t, {e: float(np.abs(a[e] - b[e][:, :32]).max()) for e in a})
rows = []
for t, L, ref in [("entity_track", 12, "16k_l082"), ("add_chain", 18, "16k_l074"), ("ioi_control", 12, "16k_l082")]:
    A, ja = load("E18", f"{t}_L{L}_L{L}_{ref}")
    if A is None: A, ja = load("E4", f"{t}_L{L}_L{L}_{ref}")
    for f in sorted(g.glob(f"{R}/E4/*/directions/{t}_L{L}_L{L}_*.npz")):
        s = f.split(f"_L{L}_L{L}_")[1][:-4]
        B, jb = load("E4", f"{t}_L{L}_L{L}_{s}")
        for est in ("pca_pooled", "answer_sep"):
            row = dict(task=t, layer=L, sae=s, width_k=int(s.split("k_")[0]), l0=int(s.split("_l0")[1]), est=est,
                       in_E20=(t, s) in {("add_chain", "65k_l064"), ("entity_track", "65k_l072"), ("entity_track", "262k_l067")})
            for r in (1, 2, 4, 8, 16, 32):
                row[f"top_r{r}"] = top(A[est][:, :r], B[est][:, :r]); row[f"mean_cos2_r{r}"] = float(np.mean(pcos(A[est][:, :r], B[est][:, :r]) ** 2))
            for r in (1, 4, 8):
                row[f"cap16k_r{r}_in_w32"] = capture(A[est][:, :r], B[est][:, :32])
            ev = jb[est]["evr"]; row["evr1"] = ev[0]; row["evr_sum8"] = float(np.sum(ev[:8]))
            ev16 = ja[est]["evr"]; row["evr1_16k"] = ev16[0]; row["evr_sum8_16k"] = float(np.sum(ev16[:8]))
            rows.append(row)
d = pd.DataFrame(rows)
nulls = {f"p99_r{r}": haar_top_p99(r, r) for r in (1, 2, 4, 8, 16, 32)}
print("Haar p99 (MC, n=1000):", {k: round(v, 3) for k, v in nulls.items()}, " capture null for 32-dim: ", round(32 / D, 4))
d.to_csv(f"{OUT}/width_subspace.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(d[["task", "sae", "est", "in_E20", "top_r1", "top_r2", "top_r4", "top_r8", "top_r16", "top_r32", "mean_cos2_r8", "mean_cos2_r32",
         "cap16k_r1_in_w32", "cap16k_r8_in_w32", "evr1", "evr_sum8", "evr1_16k", "evr_sum8_16k"]].round(3).to_string(index=False))

"""Summary statistics for cross-task overlap vs within-task cross-template / cross-language / cross-width / adjacent-layer overlap."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
import pandas as pd
ct = pd.read_csv(f"{SP}/deep/geo_crosstask_E8.csv"); ct = ct[ct.L.between(8, 20)]
for est in ["pca_pooled", "answer_sep"]:
    for r in (1, 4):
        x = ct[(ct.est == est) & (ct.r == r)]
        print(f"cross-task {est} r={r}: n={len(x)} median {x.top.median():.3f} max {x.top.max():.3f}  share > Haar p99 ({HAAR[r]}): {(x.top > HAAR[r]).mean():.2f}")
e11 = pd.read_csv(f"{SP}/deep/geo_crosslayer_E11.csv")
print("OT vs PC (E11, L8-16) from geo_decay output: pca r1 0.10-0.24, r4 0.27-0.42; answer_sep r1 0.01-0.13")
null = haar_null(4, n=300); print("Haar r=4 null check: p99 %.3f" % np.percentile(null, 99))

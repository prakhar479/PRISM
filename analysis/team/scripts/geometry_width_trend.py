"""Does persistence of the 16k error subspace in wider SAEs' error decline with width or L0? (OT L12 and IOI L12, E4 SAEs;
Spearman, n = 10 / 5 wide SAEs; uncorrected, 2 tasks x 2 estimators x 3 metrics x 2 predictors = 24 tests). Post hoc."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from geometry_lib import *
import pandas as pd
from scipy.stats import spearmanr
d = pd.read_csv(f"{OUT}/width_subspace.csv"); d = d[d.width_k > 16]
rows = []
for (t, e), x in d.groupby(["task", "est"]):
    if len(x) < 4: continue
    for met in ("top_r1", "mean_cos2_r8", "mean_cos2_r32"):
        for pred in ("width_k", "l0"):
            r = spearmanr(x[pred], x[met]); rows.append(dict(task=t, est=e, metric=met, predictor=pred, n=len(x), rho=r.statistic, p=r.pvalue,
                                                             metric_min=x[met].min(), metric_max=x[met].max()))
o = pd.DataFrame(rows); o.to_csv(f"{OUT}/width_trend.csv", index=False); print(o.round(3).to_string(index=False))
from scipy.stats import false_discovery_control
o["q_bh"] = false_discovery_control(o.p.values); print("BH q:"); print(o[["task", "est", "metric", "predictor", "rho", "p", "q_bh"]].round(3).to_string(index=False))
o.to_csv(f"{OUT}/width_trend.csv", index=False)
# pairwise matrix among all OT L12 SAEs (E4): mean cos^2 at r=8 (pca_pooled); is it a distance-from-16k effect or convergence?
import glob as g
for t in ("entity_track", "ioi_control"):
    Ss = sorted([f.split("_L12_L12_")[1][:-4] for f in g.glob(f"{R}/E4/*/directions/{t}_L12_L12_*.npz")], key=lambda s: (int(s.split("k_")[0]), int(s.split("_l0")[1])))
    U = {s: load("E4", f"{t}_L12_L12_{s}")[0]["pca_pooled"] for s in Ss}
    M = pd.DataFrame([[float(np.mean(pcos(U[a][:, :8], U[b][:, :8]) ** 2)) for b in Ss] for a in Ss], index=Ss, columns=Ss)
    M.to_csv(f"{OUT}/width_pairwise_{t}.csv"); print(t, "pairwise mean cos^2 r8 (pca_pooled)"); print(M.round(2).to_string())
    # log-width distance vs similarity across all pairs
    import itertools, math
    pr = [(abs(math.log2(int(a.split("k_")[0]) / int(b.split("k_")[0]))), abs(math.log2(int(a.split("_l0")[1]) / int(b.split("_l0")[1]))), M.loc[a, b], min(int(a.split("k_")[0]), int(b.split("k_")[0])))
          for a, b in itertools.combinations(Ss, 2)]
    pr = pd.DataFrame(pr, columns=["dlogw", "dlogl0", "sim", "minw"])
    import numpy.linalg as la
    X = np.c_[np.ones(len(pr)), pr.dlogw, pr.dlogl0, np.log2(pr.minw)]; b = la.lstsq(X, pr.sim, rcond=None)[0]
    print(t, "OLS sim ~ 1 + |dlog2 width| + |dlog2 L0| + log2(min width), coef", b.round(3), "n pairs", len(pr), "(pairs not independent: descriptive)")

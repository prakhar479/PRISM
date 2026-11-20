"""Does longer-trained DAS (das8_e10: 10 epochs) steer better than das8 (3 epochs)? Same rank, same init (seed 0), same
train set. Paired per eligible pair (f_pilot): exact McNemar test on top-1 == A at each scale, plus paired-bootstrap CI of the
NLDR difference (1000 resamples). 21 tests (3 tasks x 7 scales), BH-corrected.
Output: steering_out/das_epochs.csv"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from steering_common import OUT, bh

pp = pd.read_parquet(OUT / "pp_E19.parquet")
pp = pp[pp.f_pilot & pp.estimator.isin(["das8", "das8_e10"]) & pp.condition.str.startswith("steer_")]
rng = np.random.default_rng(0)
rows = []
for (task, s), g in pp.groupby(["task", "s"]):
    w = g.pivot_table(index="pair_id", columns="estimator", values=["isA", "D", "D_clean_A", "D_clean_B", "elig"], aggfunc="first")
    el = w[("elig", "das8")].astype(bool).to_numpy()
    a, b = w[("isA", "das8")].astype(bool).to_numpy()[el], w[("isA", "das8_e10")].astype(bool).to_numpy()[el]
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    Da, Db = w[("D", "das8")].to_numpy(), w[("D", "das8_e10")].to_numpy()
    A, Bc = w[("D_clean_A", "das8")].to_numpy(), w[("D_clean_B", "das8")].to_numpy()
    def nl(D, i): return (D[i].mean() - Bc[i].mean()) / (A[i].mean() - Bc[i].mean())
    n = len(Da)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        bs.append(nl(Db, i) - nl(Da, i))
    i0 = np.arange(n)
    rows.append(dict(task=task, s=s, n_elig=int(el.sum()), flip_das8=a.mean(), flip_e10=b.mean(), e10_only=n01, das8_only=n10,
                     p_mcnemar=p, nldr_das8=nl(Da, i0), nldr_e10=nl(Db, i0), dnldr=nl(Db, i0) - nl(Da, i0),
                     dnldr_lo=np.percentile(bs, 2.5), dnldr_hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows)
R["q_bh"] = bh(R.p_mcnemar)
R.to_csv(OUT / "das_epochs.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))

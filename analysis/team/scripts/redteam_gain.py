"""RED TEAM (exploratory, post hoc). Attack on claim B: "the SAE error acts as a (near-uniform) gain on task logit
differences". E8 atlas per-pair (one run, f_pilot, pairs with clean A and clean B correct).

Per cell and condition c in {feat_A_errB, feat_A_meanPos, feat_A_zero}: loss_i = D_A,i - D_c,i, swing_i = D_A,i - D_B,i.
  gain model:     loss_i = g * swing_i          (through the origin)
  additive model: loss_i = a                    (constant logit loss)
  affine:         loss_i = a + b * swing_i
Compare residual SDs (gain vs additive), the affine intercept (gain predicts 0) and slope with bootstrap CIs, and the
spread of r_i = 1 - loss_i/swing_i. A gain that holds equally for zero- and mean-error patches is generic (any loss of
the error), not a property of B's error.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path("/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad")
rng = np.random.default_rng(0)
pp = pd.read_parquet(SP / "ana_runs/E8/5726ecbd0230/per_pair.parquet",
                     columns=["pair_id", "task", "layer", "condition", "estimator", "seed", "f_pilot", "ans_A_id", "ans_B_id",
                              "top1_id", "logit_A", "logit_B"])
pp = pp[(pp.estimator.fillna("") == "") & pp.f_pilot]
pp["D"] = pp.logit_A - pp.logit_B
rows = []
for (task, L), g in pp.groupby(["task", "layer"]):
    g = g.drop_duplicates(["pair_id", "condition"])
    D = g.pivot(index="pair_id", columns="condition", values="D")
    T = g.pivot(index="pair_id", columns="condition", values="top1_id")
    a = g.drop_duplicates("pair_id").set_index("pair_id").reindex(D.index)
    el = D.index[(T.clean_A == a.ans_A_id) & (T.clean_B == a.ans_B_id)]
    sw = (D.clean_A - D.clean_B).loc[el].to_numpy()
    for c in ("feat_A_errB", "feat_A_meanPos", "feat_A_zero"):
        loss = (D.clean_A - D[c]).loc[el].to_numpy()
        g_hat = (loss @ sw) / (sw @ sw)
        sd_gain = np.std(loss - g_hat * sw)
        sd_add = np.std(loss - loss.mean())
        X = np.c_[np.ones(len(sw)), sw]
        b = np.linalg.lstsq(X, loss, rcond=None)[0]
        bs = np.array([np.linalg.lstsq(X[i], loss[i], rcond=None)[0] for i in (rng.integers(0, len(sw), len(sw)) for _ in range(1000))])
        r = 1 - loss / sw
        rows.append({"task": task, "layer": L, "cond": c, "n": len(el), "swing_mean": sw.mean(), "swing_cv": sw.std() / sw.mean(),
                     "loss_mean": loss.mean(), "gain_g": g_hat, "resid_sd_gain": sd_gain, "resid_sd_additive": sd_add,
                     "gain_better": sd_gain < sd_add, "affine_icpt": b[0], "icpt_lo": np.percentile(bs[:, 0], 2.5),
                     "icpt_hi": np.percentile(bs[:, 0], 97.5), "affine_slope": b[1], "slope_lo": np.percentile(bs[:, 1], 2.5),
                     "slope_hi": np.percentile(bs[:, 1], 97.5), "r_q10": np.quantile(r, .1), "r_med": np.median(r),
                     "r_q90": np.quantile(r, .9), "corr_loss_swing": np.corrcoef(loss, sw)[0, 1]})
R = pd.DataFrame(rows)
R.to_csv(SP / "team/scripts/out/redteam_gain.csv", index=False)
pd.set_option("display.width", 250)
print(R[R.cond == "feat_A_errB"].round(3).to_string())
print(R.groupby("cond")[["gain_better"]].mean(), "\nicpt CI excludes 0 (errB):",
      ((R.cond == "feat_A_errB") & ((R.icpt_lo > 0) | (R.icpt_hi < 0))).sum(), "of", (R.cond == "feat_A_errB").sum())
print(R[R.cond != "feat_A_errB"].groupby("cond")[["r_med", "corr_loss_swing"]].describe().round(3).T)

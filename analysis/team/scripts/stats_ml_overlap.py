"""STATS (exploratory, post hoc). Priority 1(a), within-language check on the ONLY region where a native-script set and a
romanized set of the same language overlap in predictability.

For each native set T and romanized/code-mixed reference R in {own Dakshina romanized set, hinglish_phinc (Hindi targets)}:
  overlap interval I = [max(min nllc_T, min nllc_R), min(max nllc_T, max nllc_R)] (clean NLL per token);
  estimators on windows inside I:
    raw   : mean cost_T / mean cost_R
    adj   : regression-adjusted ratio: OLS cost ~ nllc (+ log eps) fitted separately in T and R inside I, both evaluated at the
            pooled covariate mean (Oaxaca-style; linear, so modest extrapolation within I only)
  CIs: 2000 bootstrap resamples of windows within each set (the interval is held fixed).
Output: team/stats_out/mo_overlap.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/team/scripts")
from stats_common import OUT, e12_windows  # noqa: E402

rng = np.random.default_rng(11)
B = 2000
w = e12_windows()
w["le"] = np.log(w.eps)
PAIRS = [("hi_deva_dakshina", "hi_latn_dakshina"), ("hi_deva_web", "hi_latn_dakshina"), ("ta_taml_dakshina", "ta_latn_dakshina"),
         ("ta_web", "ta_latn_dakshina"), ("bn_beng_dakshina", "bn_latn_dakshina"), ("bn_web", "bn_latn_dakshina"),
         ("hi_deva_dakshina", "hinglish_phinc"), ("hi_deva_web", "hinglish_phinc")]


def adj(Xt, yt, Xr, yr, xbar):
    bt = np.linalg.lstsq(np.column_stack([np.ones(len(Xt)), Xt]), yt, rcond=None)[0]
    br = np.linalg.lstsq(np.column_stack([np.ones(len(Xr)), Xr]), yr, rcond=None)[0]
    return (bt[0] + xbar @ bt[1:]) / (br[0] + xbar @ br[1:])


rows = []
for (L, width), g in w.groupby(["layer", "width"]):
    for T, R in PAIRS:
        t, r = g[g.set == T], g[g.set == R]
        if not len(t) or not len(r):
            continue
        lo_i, hi_i = max(t.nllc.min(), r.nllc.min()), min(t.nllc.max(), r.nllc.max())
        ti, ri = t[(t.nllc >= lo_i) & (t.nllc <= hi_i)], r[(r.nllc >= lo_i) & (r.nllc <= hi_i)]
        base = dict(layer=L, width=width, target=T, ref=R, lo_nllc=lo_i, hi_nllc=hi_i, n_T=len(ti), n_R=len(ri),
                    share_T=len(ti) / len(t), share_R=len(ri) / len(r))
        if len(ti) < 20 or len(ri) < 20:
            rows.append({**base, "estimator": "insufficient overlap (<20 windows)"})
            continue
        yt, yr = ti.cost.to_numpy(), ri.cost.to_numpy()
        for est, cols in (("raw", []), ("adj_nllc", ["nllc"]), ("adj_nllc_eps", ["nllc", "le"])):
            Xt, Xr = ti[cols].to_numpy(), ri[cols].to_numpy()

            def stat(it, ir):
                if not cols:
                    return yt[it].mean() / yr[ir].mean()
                xbar = np.concatenate([Xt[it], Xr[ir]]).mean(0)
                return adj(Xt[it], yt[it], Xr[ir], yr[ir], xbar)
            pt = stat(np.arange(len(yt)), np.arange(len(yr)))
            bs = np.array([stat(rng.integers(0, len(yt), len(yt)), rng.integers(0, len(yr), len(yr))) for _ in range(B)])
            rows.append({**base, "estimator": est, "ratio": pt, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                         "cost_T": yt.mean(), "cost_R": yr.mean(), "nllc_T": ti.nllc.mean(), "nllc_R": ri.nllc.mean(),
                         "eps_T": ti.eps.mean(), "eps_R": ri.eps.mean(), "fvu_T": ti.fvu.mean(), "fvu_R": ri.fvu.mean()})
O = pd.DataFrame(rows)
O.to_csv(f"{OUT}/mo_overlap.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
print(O.round(3).to_string(index=False))

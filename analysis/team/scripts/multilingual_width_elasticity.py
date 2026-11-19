"""EXPLORATORY, post hoc (2026-11-19). E21 decomposition of the width gain per set and layer:
  log(ce65/ce16) = elasticity * log(fvu65/fvu16)
i.e. how much of a set's splice-cost change is "less error" (FVU reduction) vs "each unit of error costs more/less" (elasticity
> 1 means cost falls proportionally faster than FVU). Same pooled sums and same paired window bootstrap (2000) as
multilingual_width.py; elasticity difference vs English with independent draws per set. Uncorrected CIs (descriptive).
Output: $SP/team/multilingual_out/E21_elasticity.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path("/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad")
OUT = SP / "team" / "multilingual_out"
B = 2000


def main():
    w = pd.read_parquet(OUT / "windows.parquet")
    rng = np.random.default_rng(22)
    res = {}
    for (s, L), g in w.groupby(["set", "layer"]):
        a = g[g.width == "16k"].sort_values("window")
        b = g[g.width == "65k"].sort_values("window")
        n = len(a)
        Wt = np.vstack([np.bincount(r, minlength=n) for r in rng.integers(0, n, (B, n))]).astype(float)
        Wt = np.vstack([np.ones(n), Wt])

        def agg(d):
            ce = Wt @ (d.nll_splice - d.nll_clean).to_numpy() / (Wt @ d.pred.to_numpy())
            fv = Wt @ d.fvu_num.to_numpy() / (Wt @ d.fvu_den.to_numpy())
            return ce, fv
        c16, f16 = agg(a)
        c65, f65 = agg(b)
        res[(s, L)] = {"dlog_ce": np.log(c65 / c16), "dlog_fvu": np.log(f65 / f16)}
    rows = []
    for (s, L), d in res.items():
        el = d["dlog_ce"] / d["dlog_fvu"]
        e = res[("en_pile", L)]
        el_en = e["dlog_ce"] / e["dlog_fvu"]
        r = {"set": s, "layer": L, "dlog_ce": d["dlog_ce"][0], "dlog_fvu": d["dlog_fvu"][0], "elasticity": el[0],
             "el_lo": np.percentile(el[1:], 2.5), "el_hi": np.percentile(el[1:], 97.5),
             "dlog_fvu_minus_en": d["dlog_fvu"][0] - e["dlog_fvu"][0],
             "dlog_fvu_minus_en_lo": np.percentile(d["dlog_fvu"][1:] - e["dlog_fvu"][1:], 2.5),
             "dlog_fvu_minus_en_hi": np.percentile(d["dlog_fvu"][1:] - e["dlog_fvu"][1:], 97.5)}
        if s != "en_pile":
            r["el_minus_en"] = el[0] - el_en[0]
            r["el_minus_en_lo"], r["el_minus_en_hi"] = np.percentile(el[1:] - el_en[1:], [2.5, 97.5])
        rows.append(r)
    T = pd.DataFrame(rows).sort_values(["layer", "set"])
    T.to_csv(OUT / "E21_elasticity.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""EXPLORATORY, post hoc (2026-11-19). What existing E12/E21 data can say about "direction vs size" in the absence of E23
(norm-matched noise controls were NOT run). Two partial substitutes, both weak:
 (a) whole-residual ablations as a 'generic perturbation' baseline: native/romanized ratio of mean- and zero-ablation cost
     (mean_range, zero_range; nats/token) next to the ratio of splice cost. If native-script contexts were simply fragile to any
     perturbation, the ablation ratios should be about as large as the splice ratio. Caveat: ablations are not norm-matched,
     are huge (2-10 nats), and are bounded by clean predictability, which differs between scripts.
 (b) within-set window covariation: Spearman of splice cost with FVU (size) and with mean-ablation cost (window fragility),
     per set x layer x width.
Bootstrap: 2000 window resamples within set (independent across sets). Uncorrected (descriptive).
Output: $SP/team/multilingual_out/fragility_*.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SP = __import__("layout_shim").root()
OUT = SP / "team" / "multilingual_out"
B = 2000
PAIRS = {"hi": ("hi_deva_dakshina", "hi_latn_dakshina"), "ta": ("ta_taml_dakshina", "ta_latn_dakshina"),
         "bn": ("bn_beng_dakshina", "bn_latn_dakshina")}


def main():
    w = pd.read_parquet(OUT / "windows.parquet")
    rng = np.random.default_rng(23)
    rows = []
    for lang, (a, b) in PAIRS.items():
        for L in (12, 18):
            for wd in ("16k", "65k"):
                out = {"lang": lang, "layer": L, "width": wd}
                bs = {}
                for s in (a, b):
                    g = w[(w.set == s) & (w.layer == L) & (w.width == wd)]
                    n = len(g)
                    Wt = np.vstack([np.ones(n)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
                    p = Wt @ g.pred.to_numpy()
                    c = Wt @ g.nll_clean.to_numpy()
                    bs[s] = {"splice": (Wt @ g.nll_splice.to_numpy() - c) / p, "mean_abl": (Wt @ g.nll_mean.to_numpy() - c) / p,
                             "zero_abl": (Wt @ g.nll_zero.to_numpy() - c) / p, "ce_clean": c / p}
                for k in ("splice", "mean_abl", "zero_abl", "ce_clean"):
                    r = bs[a][k] / bs[b][k]
                    out[f"{k}_nat"], out[f"{k}_rom"] = bs[a][k][0], bs[b][k][0]
                    out[f"{k}_ratio"] = r[0]
                    out[f"{k}_ratio_lo"], out[f"{k}_ratio_hi"] = np.percentile(r[1:], [2.5, 97.5])
                for k in ("mean_abl", "zero_abl"):
                    rr = (bs[a]["splice"] / bs[b]["splice"]) / (bs[a][k] / bs[b][k])
                    out[f"splice_over_{k}_ratio"] = rr[0]
                    out[f"splice_over_{k}_ratio_lo"], out[f"splice_over_{k}_ratio_hi"] = np.percentile(rr[1:], [2.5, 97.5])
                rows.append(out)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "fragility_ablation_ratios.csv", index=False)
    rows = []
    for (s, L, wd), g in w.groupby(["set", "layer", "width"]):
        rows.append({"set": s, "grp": g.grp.iloc[0], "layer": L, "width": wd,
                     "rho_splice_fvu": spearmanr(g.ce_delta, g.fvu).statistic,
                     "rho_splice_epsx": spearmanr(g.ce_delta, g.eps_x).statistic,
                     "rho_splice_meanabl": spearmanr(g.ce_delta, g.mean_range).statistic,
                     "rho_splice_zeroabl": spearmanr(g.ce_delta, g.zero_range).statistic,
                     "rho_splice_ceclean": spearmanr(g.ce_delta, g.ce_clean).statistic})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "fragility_window_rho.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(T[["lang", "layer", "width", "splice_ratio", "splice_ratio_lo", "splice_ratio_hi", "mean_abl_ratio", "mean_abl_ratio_lo",
             "mean_abl_ratio_hi", "zero_abl_ratio", "zero_abl_ratio_lo", "zero_abl_ratio_hi", "ce_clean_ratio",
             "splice_over_mean_abl_ratio", "splice_over_mean_abl_ratio_lo", "splice_over_mean_abl_ratio_hi"]].round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

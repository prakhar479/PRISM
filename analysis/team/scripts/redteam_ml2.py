"""RED TEAM (exploratory, post hoc). Claim A follow-up, from out/redteam_windows.parquet (written by redteam_ml.py).

(a) How much of the native-romanized splice-cost difference would the within-set dependence of splice cost on clean CE
    explain if extrapolated (romanized clean CE is 2-4 nats higher; the distributions do not overlap)? Bootstrap over windows.
(b) Predictability-matched comparisons that ARE possible: native-script web sets and Latin sets vs English, binned by clean
    CE (0.25-nat bins, >= 20 windows in each), ratio of mean splice cost; pooled as a bin-weighted ratio.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
OUT = SP / "team" / "scripts" / "out"
w = pd.read_parquet(OUT / "redteam_windows.parquet")
rng = np.random.default_rng(1)


def slope(x, y):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


rows = []
pairs = {"hi": ("hi_deva_dakshina", "hi_latn_dakshina"), "ta": ("ta_taml_dakshina", "ta_latn_dakshina"),
         "bn": ("bn_beng_dakshina", "bn_latn_dakshina")}
for lang, (nat, rom) in pairs.items():
    for (L, width), g in w.groupby(["layer", "width"]):
        a, b = g[g.set == nat], g[g.set == rom]
        if not len(a) or not len(b):
            continue
        A, B = a[["ce_clean", "ce_delta"]].to_numpy(), b[["ce_clean", "ce_delta"]].to_numpy()

        def stat(A, B):
            dclean = B[:, 0].mean() - A[:, 0].mean()
            ddelta = A[:, 1].mean() - B[:, 1].mean()
            s_pool = (slope(A[:, 0], A[:, 1]) + slope(B[:, 0], B[:, 1])) / 2       # average within-set slope
            return ddelta, dclean, s_pool, -s_pool * dclean / ddelta
        p = stat(A, B)
        bs = np.array([stat(A[rng.integers(0, len(A), len(A))], B[rng.integers(0, len(B), len(B))]) for _ in range(1000)])
        lo, hi = np.percentile(bs[:, 3], [2.5, 97.5])
        rows.append({"lang": lang, "layer": L, "width": width, "delta_nat_minus_rom": p[0], "clean_rom_minus_nat": p[1],
                     "mean_within_slope": p[2], "share_explained_by_extrapolated_slope": p[3], "share_lo": lo, "share_hi": hi})
S = pd.DataFrame(rows)
S.to_csv(OUT / "redteam_ml2_slope_share.csv", index=False)
print(S.round(3).to_string())

# (b) binned comparisons vs English
mrows = []
for tgt in ["hi_deva_web", "hi_deva_dakshina", "ta_web", "bn_web", "ta_taml_dakshina", "bn_beng_dakshina", "fr_web",
            "hinglish_phinc", "hi_latn_dakshina"]:
    for (L, width), g in w.groupby(["layer", "width"]):
        a, e = g[g.set == tgt], g[g.set == "en_pile"]
        if not len(a) or not len(e):
            continue
        bins = np.arange(0, 10.01, 0.25)
        ab, eb = pd.cut(a.ce_clean, bins), pd.cut(e.ce_clean, bins)
        num = den = 0.0
        nb = nwin = 0
        for iv in ab.cat.categories:
            ma, me = (ab == iv).to_numpy(), (eb == iv).to_numpy()
            if ma.sum() >= 20 and me.sum() >= 20:
                wgt = ma.sum()
                num += wgt * a.ce_delta[ma].mean()
                den += wgt * e.ce_delta[me].mean()
                nb += 1
                nwin += ma.sum()
        # FVU-normalised too (sensitivity per unit FVU, window level)
        mrows.append({"set": tgt, "layer": L, "width": width, "n_bins": nb, "n_target_windows_matched": nwin,
                      "frac_target_matched": nwin / len(a), "matched_ratio_vs_en": num / den if den else np.nan,
                      "raw_ratio_vs_en": a.ce_delta.mean() / e.ce_delta.mean(),
                      "fvu_ratio_vs_en": (a.fvu_num.sum() / a.fvu_den.sum()) / (e.fvu_num.sum() / e.fvu_den.sum())})
M = pd.DataFrame(mrows)
M.to_csv(OUT / "redteam_ml2_matched_vs_en.csv", index=False)
print(M.round(3).to_string())

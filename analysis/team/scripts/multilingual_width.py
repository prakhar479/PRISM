"""EXPLORATORY, post hoc (2026-11-19). E21: does a 4x wider SAE (65k, nominal L0 72 @L12 / 64 @L18, l0_rule match16k) change
text fidelity per language, and the native-script excess? Run multilingual_load.py first.

All set-level statistics are pooled ratios of sums over windows (as harness fidelity / ml_script_pairs):
  ce_delta = sum(nll_splice - nll_clean) / sum(pred)   [nats/token, splice cost]
  fvu = sum fvu_num / sum fvu_den; eps_x = mean |eps|/|x| per position; l0 = active latents per position
  sens = ce_delta / fvu (cost per unit FVU)
  nz = ce_delta / zero_range = 1 - recovered_zero; nm = ce_delta / mean_range = 1 - recovered_mean
Bootstrap: 2000 draws; windows resampled within each set, the SAME window indices for both widths (the 16k and 65k runs score
identical windows, checked in multilingual_load.py), independent draws across sets. Two-sided percentile p-values
(floor 1/2000) with Benjamini-Hochberg within each named family of tests.
Outputs: $SP/team/multilingual_out/E21_*.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
OUT = SP / "team" / "multilingual_out"
B = 2000
PAIRS = {"hi": ("hi_deva_dakshina", "hi_latn_dakshina"), "ta": ("ta_taml_dakshina", "ta_latn_dakshina"),
         "bn": ("bn_beng_dakshina", "bn_latn_dakshina")}
COLS = ["nll_clean", "nll_splice", "nll_zero", "nll_mean", "pred", "fvu_num", "fvu_den", "eps_ratio_sum", "l0_sum", "pos"]
STATS = ["ce_delta", "fvu", "eps_x", "l0", "sens", "nz", "nm", "zero_range", "mean_range", "ce_clean"]


def stats(S):
    """S [..., len(COLS)] sums -> dict of stats arrays."""
    c, s, z, m, p, fn, fd, er, l0, pos = (S[..., i] for i in range(len(COLS)))
    d = (s - c) / p
    f = fn / fd
    zr, mr = (z - c) / p, (m - c) / p
    return {"ce_delta": d, "fvu": f, "eps_x": er / pos, "l0": l0 / pos, "sens": d / f, "nz": d / zr, "nm": d / mr,
            "zero_range": zr, "mean_range": mr, "ce_clean": c / p}


def pval(bs, null):
    bs = bs[np.isfinite(bs)]
    return max(2 * min(np.mean(bs <= null), np.mean(bs >= null)), 1 / B)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def ci(bs):
    return np.nanpercentile(bs, [2.5, 97.5])


def main():
    w = pd.read_parquet(OUT / "windows.parquet")
    rng = np.random.default_rng(21)
    point, boot = {}, {}          # (set, layer, width) -> stats dict (point: scalars, boot: [B])
    for (s, L), g in w.groupby(["set", "layer"]):
        n = (g.width == "16k").sum()
        idx = rng.integers(0, n, (B, n))
        Wt = np.vstack([np.bincount(r, minlength=n) for r in idx]).astype(float)
        for wd in ("16k", "65k"):
            A = g[g.width == wd].sort_values("window")[COLS].to_numpy(float)
            point[(s, L, wd)] = stats(A.sum(0))
            boot[(s, L, wd)] = stats(Wt @ A)
    sets = sorted({k[0] for k in point})
    grp = w.drop_duplicates("set").set_index("set").grp.to_dict()

    # 1) per-set level stats at each width + width ratio 65k/16k (paired)
    rows = []
    for s in sets:
        for L in (12, 18):
            r = {"set": s, "grp": grp[s], "layer": L}
            for st in STATS:
                a, b = point[(s, L, "16k")][st], point[(s, L, "65k")][st]
                r[f"{st}_16k"], r[f"{st}_65k"] = a, b
                bs = boot[(s, L, "65k")][st] / boot[(s, L, "16k")][st]
                r[f"{st}_ratio"] = b / a
                r[f"{st}_ratio_lo"], r[f"{st}_ratio_hi"] = ci(bs)
                if st in ("ce_delta", "fvu", "sens"):
                    r[f"{st}_ratio_p"] = pval(bs, 1.0)
            # marginal cost of the error removed by width: (ce16 - ce65) / (fvu16 - fvu65)
            num = boot[(s, L, "16k")]["ce_delta"] - boot[(s, L, "65k")]["ce_delta"]
            den = boot[(s, L, "16k")]["fvu"] - boot[(s, L, "65k")]["fvu"]
            r["marg_cost"] = ((point[(s, L, "16k")]["ce_delta"] - point[(s, L, "65k")]["ce_delta"])
                              / (point[(s, L, "16k")]["fvu"] - point[(s, L, "65k")]["fvu"]))
            r["marg_cost_lo"], r["marg_cost_hi"] = ci(num / den)
            # relative gain minus English's relative gain
            gs = 1 - boot[(s, L, "65k")]["ce_delta"] / boot[(s, L, "16k")]["ce_delta"]
            ge = 1 - boot[("en_pile", L, "65k")]["ce_delta"] / boot[("en_pile", L, "16k")]["ce_delta"]
            r["gain_rel"] = 1 - r["ce_delta_ratio"]
            r["gain_abs"] = r["ce_delta_16k"] - r["ce_delta_65k"]
            if s != "en_pile":
                gp = r["gain_rel"] - (1 - point[("en_pile", L, "65k")]["ce_delta"] / point[("en_pile", L, "16k")]["ce_delta"])
                r["gain_rel_minus_en"] = gp
                r["gain_rel_minus_en_lo"], r["gain_rel_minus_en_hi"] = ci(gs - ge)
                r["gain_rel_minus_en_p"] = pval(gs - ge, 0.0)
            rows.append(r)
    T = pd.DataFrame(rows)
    T["ce_delta_ratio_q"] = bh(T.ce_delta_ratio_p)
    T["fvu_ratio_q"] = bh(T.fvu_ratio_p)
    T["sens_ratio_q"] = bh(T.sens_ratio_p)
    m = T.gain_rel_minus_en_p.notna()
    T.loc[m, "gain_rel_minus_en_q"] = bh(T.loc[m, "gain_rel_minus_en_p"])
    T.to_csv(OUT / "E21_width_by_set.csv", index=False)

    # 2) contrasts (native/romanized same Dakshina pool; every set / English) at each width, and 65k-vs-16k ratio of ratios
    con = [(lang + " nat/rom", a, b) for lang, (a, b) in PAIRS.items()] + [(s + " / en", s, "en_pile") for s in sets if s != "en_pile"]
    rows = []
    for name, a, b in con:
        for L in (12, 18):
            r = {"contrast": name, "num": a, "den": b, "layer": L}
            for st in ("ce_delta", "fvu", "eps_x", "sens", "nz", "nm", "l0", "ce_clean"):
                for wd in ("16k", "65k"):
                    r[f"{st}_{wd}"] = point[(a, L, wd)][st] / point[(b, L, wd)][st]
                    r[f"{st}_{wd}_lo"], r[f"{st}_{wd}_hi"] = ci(boot[(a, L, wd)][st] / boot[(b, L, wd)][st])
                rr = (boot[(a, L, "65k")][st] / boot[(b, L, "65k")][st]) / (boot[(a, L, "16k")][st] / boot[(b, L, "16k")][st])
                r[f"{st}_RoR"] = r[f"{st}_65k"] / r[f"{st}_16k"]
                r[f"{st}_RoR_lo"], r[f"{st}_RoR_hi"] = ci(rr)
                if st in ("ce_delta", "sens", "nz"):
                    r[f"{st}_RoR_p"] = pval(rr, 1.0)
                    r[f"{st}_65k_p"] = pval(boot[(a, L, "65k")][st] / boot[(b, L, "65k")][st], 1.0)
            # absolute excess (nats/token) at each width and its change
            ex = {wd: boot[(a, L, wd)]["ce_delta"] - boot[(b, L, wd)]["ce_delta"] for wd in ("16k", "65k")}
            for wd in ("16k", "65k"):
                r[f"excess_{wd}"] = point[(a, L, wd)]["ce_delta"] - point[(b, L, wd)]["ce_delta"]
            r["excess_change"] = r["excess_65k"] - r["excess_16k"]
            r["excess_change_lo"], r["excess_change_hi"] = ci(ex["65k"] - ex["16k"])
            r["excess_change_p"] = pval(ex["65k"] - ex["16k"], 0.0)
            # marginal-cost ratio (cost per unit FVU removed by width)
            mc = {x: (boot[(x, L, "16k")]["ce_delta"] - boot[(x, L, "65k")]["ce_delta"])
                  / (boot[(x, L, "16k")]["fvu"] - boot[(x, L, "65k")]["fvu"]) for x in (a, b)}
            mcp = {x: (point[(x, L, "16k")]["ce_delta"] - point[(x, L, "65k")]["ce_delta"])
                   / (point[(x, L, "16k")]["fvu"] - point[(x, L, "65k")]["fvu"]) for x in (a, b)}
            r["marg_cost_ratio"] = mcp[a] / mcp[b]
            r["marg_cost_ratio_lo"], r["marg_cost_ratio_hi"] = ci(mc[a] / mc[b])
            rows.append(r)
    C = pd.DataFrame(rows)
    for fam in ("ce_delta_RoR_p", "sens_RoR_p", "nz_RoR_p", "excess_change_p", "ce_delta_65k_p", "sens_65k_p"):
        dk = C.contrast.str.contains("nat/rom")
        for msk, tag in ((dk, "dak"), (~dk, "en")):
            C.loc[msk, fam.replace("_p", "_q")] = bh(C.loc[msk, fam])
    C.to_csv(OUT / "E21_contrasts.csv", index=False)

    # 3) shift-not-tail at each width: share of native windows above English's 90th percentile of ce_delta
    rows = []
    for L in (12, 18):
        for wd in ("16k", "65k"):
            g = w[(w.layer == L) & (w.width == wd)]
            q90 = np.percentile(g[g.set == "en_pile"].ce_delta, 90)
            for s in sets:
                if s == "en_pile":
                    continue
                x = g[g.set == s].ce_delta.to_numpy()
                rows.append({"set": s, "layer": L, "width": wd, "en_q90": q90, "share_above_en_q90": float(np.mean(x > q90)),
                             "median_ratio_vs_en": float(np.median(x) / np.median(g[g.set == "en_pile"].ce_delta))})
    pd.DataFrame(rows).to_csv(OUT / "E21_shift.csv", index=False)

    # 4) pooled OLS (as deep F1): ce_delta ~ covariate + ce_clean + group dummies (English baseline), per layer x width,
    #    200 bootstraps resampling windows within sets
    rows = []
    rng2 = np.random.default_rng(5)
    for L in (12, 18):
        for wd in ("16k", "65k"):
            g = w[(w.layer == L) & (w.width == wd)].reset_index(drop=True)
            grps = ["native", "roman", "hinglish", "fr"]
            for cov in ("eps_x", "fvu"):
                def fit(gg):
                    X = np.column_stack([np.ones(len(gg)), gg[cov], gg.ce_clean] + [(gg.grp == x).astype(float) for x in grps])
                    return np.linalg.lstsq(X, gg.ce_delta.to_numpy(), rcond=None)[0]
                b = fit(g)
                pos = {s: np.where(g.set.to_numpy() == s)[0] for s in g.set.unique()}
                bs = np.array([fit(g.iloc[np.concatenate([rng2.choice(ix, len(ix)) for ix in pos.values()])]) for _ in range(200)])
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                for i, nm in enumerate(["const", cov, "ce_clean"] + grps):
                    rows.append({"layer": L, "width": wd, "covariate": cov, "term": nm, "coef": b[i], "lo": lo[i], "hi": hi[i]})
    pd.DataFrame(rows).to_csv(OUT / "E21_ols.csv", index=False)

    # 5) per-window paired width effect: share of windows where 65k costs less; Spearman of per-window cost reduction with
    #    per-window FVU reduction (within set)
    from scipy.stats import spearmanr
    rows = []
    for (s, L), g in w.groupby(["set", "layer"]):
        a = g[g.width == "16k"].sort_values("window")
        b = g[g.width == "65k"].sort_values("window")
        dce = a.ce_delta.to_numpy() - b.ce_delta.to_numpy()
        dfv = a.fvu.to_numpy() - b.fvu.to_numpy()
        rho = spearmanr(dce, dfv).statistic
        rho16 = spearmanr(a.ce_delta, a.fvu).statistic
        rows.append({"set": s, "layer": L, "share_windows_65k_cheaper": float(np.mean(dce > 0)),
                     "share_windows_65k_lower_fvu": float(np.mean(dfv > 0)), "spearman_dce_dfvu": rho,
                     "spearman_ce_fvu_16k": rho16, "spearman_ce_fvu_65k": spearmanr(b.ce_delta, b.fvu).statistic})
    pd.DataFrame(rows).to_csv(OUT / "E21_window_paired.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print("== per-set width ratios (65k/16k)")
    print(T[["set", "layer", "ce_delta_16k", "ce_delta_65k", "ce_delta_ratio", "ce_delta_ratio_lo", "ce_delta_ratio_hi",
             "fvu_ratio", "sens_ratio", "sens_ratio_lo", "sens_ratio_hi", "l0_16k", "l0_65k", "marg_cost", "marg_cost_lo",
             "marg_cost_hi", "gain_rel_minus_en", "gain_rel_minus_en_lo", "gain_rel_minus_en_hi", "gain_rel_minus_en_q"]
            ].round(3).to_string(index=False))
    print("== contrasts")
    print(C[["contrast", "layer", "ce_delta_16k", "ce_delta_65k", "ce_delta_65k_lo", "ce_delta_65k_hi", "ce_delta_RoR",
             "ce_delta_RoR_lo", "ce_delta_RoR_hi", "ce_delta_RoR_q", "fvu_16k", "fvu_65k", "sens_16k", "sens_65k",
             "sens_RoR", "sens_RoR_lo", "sens_RoR_hi", "excess_16k", "excess_65k", "excess_change_lo", "excess_change_hi",
             "marg_cost_ratio", "marg_cost_ratio_lo", "marg_cost_ratio_hi"]].round(3).to_string(index=False))
    print(C[["contrast", "layer", "nz_16k", "nz_65k", "nz_65k_lo", "nz_65k_hi", "nz_RoR", "nz_RoR_lo", "nz_RoR_hi",
             "nm_16k", "nm_65k", "l0_16k", "l0_65k", "eps_x_65k", "ce_clean_65k"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

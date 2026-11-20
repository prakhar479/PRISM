"""Task 1: multilingual splice cost x SAE width (E12 16k vs E21 65k), per-window models. EXPLORATORY, post hoc.

Design facts used: the 65k (E21, node-D) and 16k (E12, node-C) runs score the SAME windows (nll_clean identical, checked
below), so width is a within-window contrast. Dakshina native/romanized sets are the same shuffled sentence pool but are packed
differently (different tokens per sentence), so windows do not align 1:1; we align them on SENTENCE BLOCKS: the sentence index of
each window is recovered by counting the "\n" separator token (id 108) in the stored token ids (harness/langsets.packed_windows).
Outputs -> team/stats_out/ml_*.csv; printed summary.
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".layout/team/scripts")
from stats_common import DAK, LATIN, NATIVE, OUT, SP, bh, e12_windows, ols, wild_cluster_p  # noqa: E402

NB = 2000
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

w = e12_windows()
w = w[w.layer.isin([12, 18])]
print("windows:", w.groupby(["width", "layer"]).size().to_dict())

# ---------------------------------------------------------------- 0. same windows across widths / machines?
chk = []
for (s, L), g in w.groupby(["set", "layer"]):
    a = g[g.width == "16k"].sort_values("win")
    b = g[g.width == "65k"].sort_values("win")
    if len(a) == 0 or len(b) == 0:
        continue
    chk.append({"set": s, "layer": L, "n16": len(a), "n65": len(b),
                **{f"maxabs_{k}": float(np.abs(a[k].to_numpy() - b[k].to_numpy()).max()) for k in
                   ("nll_clean", "nll_mean", "nll_zero", "fvu_den", "pred", "pos")}})
chk = pd.DataFrame(chk)
print("\n[0] cross-width/machine identity of width-independent quantities (max abs diff):")
print(chk.drop(columns=["layer"]).groupby("set").max().to_string())
chk.to_csv(f"{OUT}/ml_identity_check.csv", index=False)

# ---------------------------------------------------------------- 1. per set x layer: cost16, cost65, ratio (window bootstrap)
wide = w.pivot_table(index=["set", "layer", "win"], columns="width",
                     values=["nll_splice", "nll_clean", "pred", "fvu_num", "fvu_den", "eps", "cost", "fvu"]).dropna()
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()
wide["native"] = wide.set.isin(NATIVE).astype(int)
print("\nfraction of windows with cost <= 0:", {k: float((w[w.width == k].cost <= 0).mean()) for k in ("16k", "65k")})


def set_stats(g, ix=None):
    g = g if ix is None else g.iloc[ix]
    p = g.pred_16k.sum()
    c16 = (g.nll_splice_16k - g.nll_clean_16k).sum() / p
    c65 = (g.nll_splice_65k - g.nll_clean_16k).sum() / p
    f16 = g.fvu_num_16k.sum() / g.fvu_den_16k.sum()
    f65 = g.fvu_num_65k.sum() / g.fvu_den_16k.sum()
    return np.array([c16, c65, c65 / c16, f16, f65, f65 / f16, (c16 / f16), (c65 / f65), (c65 / f65) / (c16 / f16)])


NAMES = ["cost16", "cost65", "cost_ratio", "fvu16", "fvu65", "fvu_ratio", "sens16", "sens65", "sens_ratio"]
rows, boots = [], {}
for (s, L), g in wide.groupby(["set", "layer"]):
    g = g.reset_index(drop=True)
    pt = set_stats(g)
    bs = np.array([set_stats(g, rng.integers(0, len(g), len(g))) for _ in range(NB)])
    # block bootstrap (blocks of 10 consecutive windows) as a dependence check
    nb = len(g) // 10
    blocks = np.arange(nb * 10).reshape(nb, 10)
    bb = np.array([set_stats(g, blocks[rng.integers(0, nb, nb)].ravel()) for _ in range(500)])
    boots[(s, L)] = bs
    r = {"set": s, "layer": L, "native": int(s in NATIVE), "n_windows": len(g)}
    for i, k in enumerate(NAMES):
        r[k] = pt[i]
        r[k + "_lo"], r[k + "_hi"] = np.quantile(bs[:, i], [0.025, 0.975])
    r["cost_ratio_blk_lo"], r["cost_ratio_blk_hi"] = np.quantile(bb[:, 2], [0.025, 0.975])
    rows.append(r)
T1 = pd.DataFrame(rows)
T1.to_csv(f"{OUT}/ml_width_by_set.csv", index=False)
print("\n[1] per set x layer (ratio of means; 95% window bootstrap, 2000; block-of-10 bootstrap 500 for cost_ratio):")
print(T1[["set", "layer", "native", "cost16", "cost65", "cost_ratio", "cost_ratio_lo", "cost_ratio_hi", "cost_ratio_blk_lo",
          "cost_ratio_blk_hi", "fvu_ratio", "sens_ratio", "sens_ratio_lo", "sens_ratio_hi"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 2. script contrast of width benefit across SETS (unit = set)
print("\n[2] set-level test: does log(cost65/cost16) differ native vs Latin? (exact permutation over 12 sets, 924 splits)")
T2 = []
for L in (12, 18):
    t = T1[T1.layer == L].set_index("set")
    for metric in ("cost_ratio", "sens_ratio"):
        x = np.log(t[metric])
        nat = t.native.to_numpy().astype(bool)
        obs = x[nat].mean() - x[~nat].mean()
        perm = []
        for comb in itertools.combinations(range(12), 6):
            m = np.zeros(12, bool)
            m[list(comb)] = True
            perm.append(x[m].mean() - x[~m].mean())
        perm = np.array(perm)
        p = float(np.mean(np.abs(perm) >= abs(obs) - 1e-12))
        T2.append({"layer": L, "metric": metric, "mean_log_native": x[nat].mean(), "mean_log_latin": x[~nat].mean(),
                   "diff_log": obs, "ratio_of_ratios": float(np.exp(obs)), "perm_p": p})
T2 = pd.DataFrame(T2)
print(T2.round(4).to_string(index=False))
T2.to_csv(f"{OUT}/ml_setlevel_perm.csv", index=False)

# ---------------------------------------------------------------- 3. within-window differenced regression with covariates
# d log cost = b0 + g*native + d*dlogFVU + e*dlog(eps) + k*(nllc_pt - mean) per layer; equivalent to window FE + width + width x
# native in levels. Clustered: CR1 by window (HC1) vs CR1 by set (12 clusters) + wild cluster bootstrap (all 4096 sign vectors).
print("\n[3] within-window model: dlog(cost) [65k vs 16k] ~ native + dlogFVU + dlog(|eps|/|x|) + clean NLL/token (centred)")
T3 = []
for L in (12, 18):
    g = wide[(wide.layer == L) & (wide.cost_16k > 0) & (wide.cost_65k > 0)].copy()
    g["nllc"] = g.nll_clean_16k / g.pred_16k
    for yname, y in (("dlog_cost", np.log(g.cost_65k / g.cost_16k)), ("dcost", g.cost_65k - g.cost_16k)):
        if yname == "dlog_cost":
            covs = np.c_[np.log(g.fvu_65k / g.fvu_16k), np.log(g.eps_65k / g.eps_16k), g.nllc - g.nllc.mean()]
        else:
            covs = np.c_[g.fvu_65k - g.fvu_16k, g.eps_65k - g.eps_16k, g.nllc - g.nllc.mean()]
        for spec, X in (("no_cov", np.c_[np.ones(len(g)), g.native]), ("cov", np.c_[np.ones(len(g)), g.native, covs])):
            b, se_w = ols(X, y)
            _, se_s = ols(X, y, g.set)
            p_wild, t_s = wild_cluster_p(X, y.to_numpy(), g.set.to_numpy(), 1)
            T3.append({"layer": L, "y": yname, "spec": spec, "n_windows": len(g), "n_dropped_nonpos": int(
                ((wide.layer == L).sum()) - len(g)), "native_coef": b[1], "se_window": se_w[1], "se_set": se_s[1],
                "p_wild_set": p_wild, "dFVU_coef": b[2] if spec == "cov" else np.nan,
                "dFVU_se_set": se_s[2] if spec == "cov" else np.nan, "intercept_latin": b[0]})
T3 = pd.DataFrame(T3)
print(T3.round(4).to_string(index=False))
T3.to_csv(f"{OUT}/ml_window_reg.csv", index=False)

# ---------------------------------------------------------------- 4. Dakshina sentence-block pairing (same sentences, two scripts)
print("\n[4] Dakshina, paired on sentence blocks (50 sentences); 2000 bootstraps over blocks")
B = 50


def sentence_mid(setname):
    ids = json.load(open(f"{SP}/hw/data/lang/{setname}.json"))["ids"]
    nl = np.array([sum(1 for t in x[1:] if t == 108) for x in ids])
    start = np.r_[0, np.cumsum(nl)[:-1]]
    return start + nl / 2.0, int(nl.sum())


T4, T4b = [], []
for lang, (sn, sl) in DAK.items():
    mid_n, tot_n = sentence_mid(sn)
    mid_l, tot_l = sentence_mid(sl)
    nblk = int(min(tot_n, tot_l) // B)
    for L in (12, 18):
        agg = {}
        for s, mid in ((sn, mid_n), (sl, mid_l)):
            g = wide[(wide.set == s) & (wide.layer == L)].sort_values("win").copy()
            g["blk"] = (mid[g.win.to_numpy()] // B).astype(int)
            g = g[g.blk < nblk]
            g["d16"] = g.nll_splice_16k - g.nll_clean_16k
            g["d65"] = g.nll_splice_65k - g.nll_clean_16k
            a = g.groupby("blk")[["d16", "d65", "pred_16k", "fvu_num_16k", "fvu_num_65k", "fvu_den_16k", "nll_clean_16k"]].sum()
            agg[s] = a.reindex(range(nblk))
        an, al = agg[sn], agg[sl]
        ok = an.notna().all(axis=1) & al.notna().all(axis=1)
        an, al = an[ok].to_numpy(), al[ok].to_numpy()
        nb = len(an)

        def stat(ix):
            n, l_ = an[ix].sum(0), al[ix].sum(0)
            cn16, cn65, cl16, cl65 = n[0] / n[2], n[1] / n[2], l_[0] / l_[2], l_[1] / l_[2]
            fn16, fn65, fl16, fl65 = n[3] / n[5], n[4] / n[5], l_[3] / l_[5], l_[4] / l_[5]
            r16, r65 = cn16 / cl16, cn65 / cl65
            s16, s65 = (cn16 / fn16) / (cl16 / fl16), (cn65 / fn65) / (cl65 / fl65)
            return np.array([r16, r65, np.log(r65 / r16), s16, s65, np.log(s65 / s16), cn16 - cl16, cn65 - cl65,
                             (cn65 - cl65) - (cn16 - cl16), fn16 / fl16, fn65 / fl65])

        pt = stat(np.arange(nb))
        bs = np.array([stat(rng.integers(0, nb, nb)) for _ in range(NB)])
        nm = ["cost_ratio16", "cost_ratio65", "dlog_cost_ratio", "sens_ratio16", "sens_ratio65", "dlog_sens_ratio",
              "excess16", "excess65", "d_excess", "fvu_ratio16", "fvu_ratio65"]
        r = {"lang": lang, "layer": L, "n_blocks": nb}
        for i, k in enumerate(nm):
            r[k] = pt[i]
            r[k + "_lo"], r[k + "_hi"] = np.quantile(bs[:, i], [0.025, 0.975])
        r["p_dlog_sens"] = float(min(1, 2 * min(np.mean(bs[:, 5] <= 0), np.mean(bs[:, 5] >= 0))))
        r["p_dlog_cost"] = float(min(1, 2 * min(np.mean(bs[:, 2] <= 0), np.mean(bs[:, 2] >= 0))))
        r["p_d_excess"] = float(min(1, 2 * min(np.mean(bs[:, 8] <= 0), np.mean(bs[:, 8] >= 0))))
        # compare to unpaired window bootstrap of the 16k cost ratio (independent resampling of the two sets)
        r["unpaired_ratio16_lo"], r["unpaired_ratio16_hi"] = np.quantile(
            boots[(sn, L)][:, 0] / boots[(sl, L)][:, 0], [0.025, 0.975])
        T4.append(r)
T4 = pd.DataFrame(T4)
T4["q_dlog_sens_BH6"] = bh(T4.p_dlog_sens)
T4["q_dlog_cost_BH6"] = bh(T4.p_dlog_cost)
T4.to_csv(f"{OUT}/ml_dakshina_blocks.csv", index=False)
show = ["lang", "layer", "n_blocks", "cost_ratio16", "cost_ratio16_lo", "cost_ratio16_hi", "unpaired_ratio16_lo",
        "unpaired_ratio16_hi", "cost_ratio65", "cost_ratio65_lo", "cost_ratio65_hi", "dlog_cost_ratio", "dlog_cost_ratio_lo",
        "dlog_cost_ratio_hi", "p_dlog_cost", "q_dlog_cost_BH6"]
print(T4[show].round(4).to_string(index=False))
show2 = ["lang", "layer", "fvu_ratio16", "fvu_ratio65", "sens_ratio16", "sens_ratio16_lo", "sens_ratio16_hi", "sens_ratio65",
         "sens_ratio65_lo", "sens_ratio65_hi", "dlog_sens_ratio", "dlog_sens_ratio_lo", "dlog_sens_ratio_hi", "p_dlog_sens",
         "q_dlog_sens_BH6", "excess16", "excess65", "d_excess", "d_excess_lo", "d_excess_hi"]
print(T4[show2].round(4).to_string(index=False))

# ---------------------------------------------------------------- 5. levels model across both widths (cluster by window)
# cost ~ set:layer FE + W65:layer + W65:native:layer + FVU:layer + FVU:native:layer + eps + nllc ; per layer
print("\n[5] levels model per layer: cost ~ set FE + W65 + W65:native + FVU + FVU:native + |eps|/|x| + nllc; CR1 by window")
T5 = []
for L in (12, 18):
    g = w[w.layer == L].copy()
    g["W"] = (g.width == "65k").astype(float)
    sets = sorted(g.set.unique())
    FE = np.c_[[(g.set == s).to_numpy().astype(float) for s in sets]].T
    X = np.c_[FE, g.W, g.W * g.native, g.fvu, g.fvu * g.native, g.eps, g.nllc]
    names = sets + ["W65", "W65:native", "FVU", "FVU:native", "eps", "nllc"]
    cl = g.set + "_" + g.win.astype(str)
    b, se = ols(X, g.cost, cl)
    _, se_set = ols(X, g.cost, g.set)
    for nmx in ("W65", "W65:native", "FVU", "FVU:native", "eps", "nllc"):
        j = names.index(nmx)
        T5.append({"layer": L, "term": nmx, "coef": b[j], "se_window": se[j], "se_set": se_set[j],
                   "z_window": b[j] / se[j]})
T5 = pd.DataFrame(T5)
print(T5.round(5).to_string(index=False))
T5.to_csv(f"{OUT}/ml_levels_reg.csv", index=False)

# ---------------------------------------------------------------- 6. how much do window-level SEs overstate certainty for SCRIPT?
# 16k only, per layer (6, 12, 18, 24): cost ~ native + |eps|/|x| + clean NLL/token (+ English indicator), window HC1 vs set CR1
print("\n[6] 16k script effect: cost ~ native + eps + nllc + is_english; SE by window (HC1) vs by set (CR1, wild p)")
w16 = e12_windows()
w16 = w16[w16.width == "16k"]
T6 = []
for L, g in w16.groupby("layer"):
    X = np.c_[np.ones(len(g)), g.native, g.eps, g.nllc, (g.set == "en_pile").astype(float)]
    b, se_w = ols(X, g.cost)
    _, se_s = ols(X, g.cost, g.set)
    p_wild, _ = wild_cluster_p(X, g.cost.to_numpy(), g.set.to_numpy(), 1)
    T6.append({"layer": L, "n_windows": len(g), "n_sets": g.set.nunique(), "native_coef": b[1], "se_window": se_w[1],
               "se_set": se_s[1], "se_inflation": se_s[1] / se_w[1], "p_wild_set": p_wild})
T6 = pd.DataFrame(T6)
print(T6.round(4).to_string(index=False))
T6.to_csv(f"{OUT}/ml_script_se_inflation.csv", index=False)

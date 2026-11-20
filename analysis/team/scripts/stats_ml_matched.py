"""STATS (exploratory, post hoc). Priority 1(a): native-script vs Latin-script splice cost on COMMON SUPPORT of predictability
(clean NLL per token) and error size (FVU, |eps|/|x|, raw |eps|^2 per position), E12 (16k) + E21 (65k) per-window data.

For every target set T (all 12 sets, so Latin targets act as placebos) and reference R in
  en_pile, fr_web, hinglish_phinc, rom_own (the Dakshina romanized set of the same language; Indic targets only),
  latin_other (all Latin sets pooled, excluding T)
and every matching spec
  P  : clean NLL/token (nllc)
  PF : nllc + log FVU
  PE : nllc + log mean |eps|/|x|
  PR : nllc + log raw |eps|^2 per position (mean-free error size)
we estimate the ATT-type ratio  R = mean cost(T, matched) / mean cost(R, matched to T's covariates)  with
  CEM: coarsened exact matching, bins nllc 0.25 nats, log-FVU 0.10, log-eps 0.05, log-raw 0.10; R windows reweighted to T's
       stratum counts; only strata containing both sets are kept.
  NN : 1-nearest neighbour with replacement, Chebyshev distance after scaling each covariate by a caliper of half the CEM bin;
       T windows with no R neighbour inside the caliper are dropped.
Cost per window = (nll_splice - nll_clean)/pred (nats/token). CIs: percentile bootstrap resampling windows within each set
(stratified by set, matching recomputed each draw), B=1000 (CEM) / 300 (NN). Uncorrected; counts of tests given in stats.md.
Output: team/stats_out/mm_matched.csv
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, ".layout/team/scripts")
from stats_common import LATIN, NATIVE, OUT, e12_windows  # noqa: E402

B_CEM, B_NN = 1000, 300
rng = np.random.default_rng(20261119)
ROM = {"hi_deva_dakshina": "hi_latn_dakshina", "hi_deva_web": "hi_latn_dakshina", "ta_taml_dakshina": "ta_latn_dakshina",
       "ta_web": "ta_latn_dakshina", "bn_beng_dakshina": "bn_latn_dakshina", "bn_web": "bn_latn_dakshina"}
SPECS = {"P": ["nllc"], "PF": ["nllc", "lf"], "PE": ["nllc", "le"], "PR": ["nllc", "lr"]}
BINW = {"nllc": 0.25, "lf": 0.10, "le": 0.05, "lr": 0.10}

w = e12_windows()
w["lf"] = np.log(w.fvu)
w["le"] = np.log(w.eps)
w["lr"] = np.log(w.fvu_num / w.pos)


def cem_codes(X, cols):
    """integer stratum code per row (shared grid across sets)."""
    code = np.zeros(len(X), dtype=np.int64)
    for c in cols:
        b = np.floor(X[c] / BINW[c]).astype(np.int64)
        b = b - b.min() if len(b) else b
        code = code * 100003 + b
    return code


def cem_ratio(ct, yt, cr, yr):
    """ct/cr: dense stratum ids (0..S-1). Returns ratio, matched T share, mean T, weighted mean R."""
    S = int(max(ct.max(), cr.max())) + 1
    nt = np.bincount(ct, minlength=S)
    nr = np.bincount(cr, minlength=S)
    st = np.bincount(ct, yt, minlength=S)
    sr = np.bincount(cr, yr, minlength=S)
    ok = (nt > 0) & (nr > 0)
    if not ok.any():
        return np.nan, 0.0, np.nan, np.nan
    mt = st[ok].sum() / nt[ok].sum()
    mr = (nt[ok] * sr[ok] / nr[ok]).sum() / nt[ok].sum()
    return mt / mr, nt[ok].sum() / len(ct), mt, mr


def nn_ratio(Zt, yt, Zr, yr):
    tree = cKDTree(Zr)
    d, j = tree.query(Zt, k=1, p=np.inf, distance_upper_bound=1.0)
    ok = np.isfinite(d)
    if ok.sum() == 0:
        return np.nan, 0.0, np.nan, np.nan
    mt, mr = yt[ok].mean(), yr[j[ok]].mean()
    return mt / mr, ok.mean(), mt, mr


rows = []
t0 = time.time()
for (L, width), g in w.groupby(["layer", "width"]):
    sets = sorted(g.set.unique())
    for T in sets:
        refs = {"en_pile": ["en_pile"], "fr_web": ["fr_web"], "hinglish_phinc": ["hinglish_phinc"],
                "latin_other": [s for s in LATIN if s in sets and s != T]}
        if T in ROM and ROM[T] in sets:
            refs["rom_own"] = [ROM[T]]
        for rname, rsets in refs.items():
            if T in rsets or not rsets:
                continue
            gt = g[g.set == T].reset_index(drop=True)
            gr = g[g.set.isin(rsets)].reset_index(drop=True)
            rgroups = [np.where(gr.set.to_numpy() == s)[0] for s in rsets]
            yt, yr = gt.cost.to_numpy(), gr.cost.to_numpy()
            for sp, cols in SPECS.items():
                # ---------- CEM
                code = cem_codes(pd.concat([gt[cols], gr[cols]]), cols)
                _, dense = np.unique(code, return_inverse=True)
                ct, cr = dense[:len(gt)], dense[len(gt):]
                pt = cem_ratio(ct, yt, cr, yr)
                # balance of matched samples (CEM weights)
                S = dense.max() + 1
                nt_s, nr_s = np.bincount(ct, minlength=S), np.bincount(cr, minlength=S)
                ok_s = (nt_s > 0) & (nr_s > 0)
                mt_mask, mr_mask = ok_s[ct], ok_s[cr]
                wr = (nt_s / np.maximum(nr_s, 1))[cr] * mr_mask
                bal = {}
                for c in ("nllc", "fvu", "eps"):
                    bal[f"{c}_T"] = gt[c].to_numpy()[mt_mask].mean() if mt_mask.any() else np.nan
                    bal[f"{c}_R"] = (gr[c].to_numpy() * wr).sum() / wr.sum() if wr.sum() > 0 else np.nan
                bs = []
                for _ in range(B_CEM):
                    it = rng.integers(0, len(gt), len(gt))
                    ir = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in rgroups])
                    bs.append(cem_ratio(ct[it], yt[it], cr[ir], yr[ir])[0])
                bs = np.array(bs)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).sum() > 50 else (np.nan, np.nan)
                rows.append(dict(layer=L, width=width, target=T, native=int(T in NATIVE), ref=rname, spec=sp, method="CEM",
                                 ratio=pt[0], lo=lo, hi=hi, share_T_matched=pt[1], n_T=len(gt), n_T_matched=int(mt_mask.sum()),
                                 n_R_used=int(mr_mask.sum()), cost_T=pt[2], cost_R=pt[3], boot_valid=float(np.isfinite(bs).mean()),
                                 **bal))
                # ---------- NN (P and PF only, to bound run time)
                if sp in ("P", "PF"):
                    sc = np.array([BINW[c] / 2 for c in cols])
                    Zt, Zr = gt[cols].to_numpy() / sc, gr[cols].to_numpy() / sc
                    pn = nn_ratio(Zt, yt, Zr, yr)
                    bn = []
                    for _ in range(B_NN):
                        it = rng.integers(0, len(gt), len(gt))
                        ir = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in rgroups])
                        bn.append(nn_ratio(Zt[it], yt[it], Zr[ir], yr[ir])[0])
                    bn = np.array(bn)
                    lo, hi = np.nanpercentile(bn, [2.5, 97.5]) if np.isfinite(bn).sum() > 50 else (np.nan, np.nan)
                    rows.append(dict(layer=L, width=width, target=T, native=int(T in NATIVE), ref=rname, spec=sp,
                                     method="NN", ratio=pn[0], lo=lo, hi=hi, share_T_matched=pn[1], n_T=len(gt),
                                     n_T_matched=int(round(pn[1] * len(gt))), cost_T=pn[2], cost_R=pn[3],
                                     boot_valid=float(np.isfinite(bn).mean())))
        print(f"L{L} {width} {T} done ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/mm_matched.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
show = R[(R.method == "CEM") & (R.spec.isin(["P", "PF"]))]
print(show[["layer", "width", "target", "ref", "spec", "ratio", "lo", "hi", "share_T_matched", "n_T_matched", "n_R_used",
            "nllc_T", "nllc_R", "fvu_T", "fvu_R"]].round(3).to_string())

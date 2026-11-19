"""Task 4: enumerate the inferential tests implied by the E17-E22 standard tables, recompute p-values from per-pair data
(pair bootstrap, B = 2000, seed 0; p from the bootstrap SE, normal approximation; percentile p also kept), and apply
Benjamini-Hochberg (q = 0.05) per experiment. EXPLORATORY. Output: team/stats_out/mult_tests.csv, mult_summary.csv.

Test types (one per table row that carries a decision or a CI someone would read as a claim):
  gap        two-sided, H0 gap(M) = 0 for baseline conditions (feat_A_errB, feat_A_meanPos, ...), M in {flip, NLDR}
  effective  one-sided, H0 flip <= random-U p95 (same cell, filter, condition, rank; >= 20 Haar draws) for full_along_U, repair,
             feat_A_along_U, steer_x<s> rows (the CI half of the addendum's 'effective' rule; the +10-point half is not a test)
  illusion   one-sided, H0 flip(full_along_U) - flip(feat_A_along_U) <= 0, paired
  q2         one-sided, H0 steering flip <= 0.50 (E19)
  transfer   two-sided, H0 flip(English-fit U) - flip(native-fit U) = 0 on the same evaluation pairs (E22), paired
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/team/scripts")
from stats_common import OUT, bh  # noqa: E402
from stats_engine import Cell, boot_W, ci, load, samples  # noqa: E402
from stats_pp import run_dirs  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
FILTERS = ("f_none", "f_A_correct", "f_pilot")
BASE = ("feat_A_errB", "feat_A_meanPos", "feat_A_zero", "feat_B_noise", "feat_A_noise", "feat_B_errA")
B = 2000


def pvals(s, null, side):
    est, bs = s[0], s[1:]
    bs = bs[np.isfinite(bs)]
    se = float(np.std(bs, ddof=1)) if len(bs) > 1 else np.nan
    d = est - null
    if not np.isfinite(est):
        return est, se, np.nan, np.nan
    if se == 0 or not np.isfinite(se):
        p = (0.0 if d > 0 else 1.0) if side == "greater" else (0.0 if d != 0 else 1.0)
    else:
        z = d / se
        p = float(sst.norm.sf(z)) if side == "greater" else float(2 * sst.norm.sf(abs(z)))
    if side == "greater":
        pp = (1 + np.sum(bs <= null)) / (len(bs) + 1)
    else:
        pp = min(1.0, 2 * (1 + min(np.sum(bs <= null), np.sum(bs >= null))) / (len(bs) + 1))
    return est, se, p, float(pp)


def row(exp, cell_id, flt, n, ttype, metric, key, s, null, side, **kw):
    est, se, p, pp = pvals(s, null, side)
    lo, hi = ci(s[1:])
    lo90, hi90 = ci(s[1:], 0.90)
    return {"exp": exp, **cell_id, "filter": flt, "n_pairs": n, "type": ttype, "metric": metric, "condition": key[0],
            "estimator": key[1], "r": key[3], "est": est, "null": null, "se": se, "lo95": lo, "hi95": hi, "lo90": lo90,
            "hi90": hi90, "side": side, "p": p, "p_pct": pp, **kw}


tests = []
for exp in ("E17", "E18", "E19", "E20", "E22"):
    d = run_dirs(exp)[0]
    pp = load(d + "per_pair.parquet")
    pp["estimator"] = pp.estimator.fillna("")
    pp["source"] = pp.source.fillna("")
    groups = list(pp.groupby(["task", "layer", "sae", "fit_families", "family"], sort=False))
    cells = {}
    for gk, g in groups:
        cell_id = dict(zip(["task", "layer", "sae", "fit_families", "family"], gk))
        for flt in FILTERS:
            c = Cell(g, flt)
            if not c.ok or c.n < 2:
                continue
            W = boot_W(c.n, B, 0)
            full = c.arrays(next(k for k in c.keys() if k[0] == "full_A"))
            F = samples(W, *full, c.DA, c.DB, c.elig)
            S, P = {}, {}
            for k in c.keys():
                if k[0] in ("clean_A", "clean_B", "full_A", "full_B"):
                    continue
                a = c.arrays(k)
                if a is None:
                    continue
                if k[1] == "random":
                    P[k] = samples(W[:1], *a, c.DA, c.DB, c.elig)["flip_rate"][0]
                else:
                    S[k] = samples(W, *a, c.DA, c.DB, c.elig)
            cells[(gk, flt)] = (c, W, S)
            null95 = {}
            for k, v in P.items():
                null95.setdefault((k[0], k[3]), []).append(v)
            null95 = {k: (float(np.nanpercentile(v, 95)), len(v)) for k, v in null95.items()}
            for k, s in S.items():
                if k[0] in BASE:
                    for met in ("flip_rate", "nldr"):
                        tests.append(row(exp, cell_id, flt, c.n, "gap", met, k, F[met] - s[met], 0.0, "two-sided"))
                if (k[0] in ("full_along_U", "feat_A_along_U", "repair") or k[0].startswith("steer_")) and int(k[3]) > 0:
                    p95, nr = null95.get((k[0], k[3]), (np.nan, 0))
                    if nr >= 20:
                        tests.append(row(exp, cell_id, flt, c.n, "effective", "flip_rate", k, s["flip_rate"], p95, "greater",
                                         random_n=nr))
                if k[0].startswith("steer_"):
                    tests.append(row(exp, cell_id, flt, c.n, "q2", "flip_rate", k, s["flip_rate"], 0.5, "greater"))
                if k[0] == "full_along_U" and int(k[3]) > 0:
                    k2 = ("feat_A_along_U",) + k[1:]
                    if k2 in S:
                        tests.append(row(exp, cell_id, flt, c.n, "illusion", "flip_rate", k, S[k]["flip_rate"] -
                                         S[k2]["flip_rate"], 0.0, "greater"))
    if exp == "E22":
        # transfer: English-fit vs native-fit U on the same evaluation family and pairs
        for (gk, flt), (c, W, S) in cells.items():
            task, layer, sae, fit, fam = gk
            if fit == fam or not str(fit).endswith("_en"):
                continue
            nat = cells.get(((task, layer, sae, fam, fam), flt))
            if nat is None:
                continue
            cn = nat[0]
            common = cn.pairs.intersection(c.pairs).sort_values()
            ixn = cn.pairs.get_indexer(common)
            Wc = boot_W(len(common), B, 0)
            for k in c.keys():
                if k[0] not in ("full_along_U", "feat_A_along_U") or k[1] == "random" or int(k[3]) <= 0:
                    continue
                a_e = c.arrays(k, common)
                a_n = cn.arrays(k, common)
                if a_e is None or a_n is None:
                    continue
                DA, DB, el = cn.DA[ixn], cn.DB[ixn], cn.elig[ixn]
                se_ = samples(Wc, *a_e, DA, DB, el)["flip_rate"]
                sn_ = samples(Wc, *a_n, DA, DB, el)["flip_rate"]
                tests.append(row(exp, dict(zip(["task", "layer", "sae", "fit_families", "family"], gk)), flt, len(common),
                                 "transfer", "flip_rate", k, se_ - sn_, 0.0, "two-sided", native=float(sn_[0]),
                                 english=float(se_[0])))
    print(exp, "tests so far:", len(tests), flush=True)

T = pd.DataFrame(tests)
T["q_BH_exp_all"] = np.nan
T["q_BH_exp_fAc"] = np.nan
for exp, g in T.groupby("exp"):
    T.loc[g.index, "q_BH_exp_all"] = bh(g.p.to_numpy())
    gf = g[g["filter"] == "f_A_correct"]
    T.loc[gf.index, "q_BH_exp_fAc"] = bh(gf.p.to_numpy())
T["q_BH_exp_all_pct"] = np.nan
for exp, g in T.groupby("exp"):
    T.loc[g.index, "q_BH_exp_all_pct"] = bh(g.p_pct.to_numpy())
T["sig_raw"] = T.p < 0.05
T["sig_BH"] = T.q_BH_exp_all < 0.05
# TOST equivalence within +-10 points for two-sided tests (90% CI inside [-0.10, 0.10])
T["equiv_10pts"] = np.where(T.side == "two-sided", (T.lo90 > -0.10) & (T.hi90 < 0.10), np.nan)
T.to_csv(f"{OUT}/mult_tests.csv", index=False)

S = T.groupby(["exp", "type"]).agg(m=("p", "size"), raw_sig=("sig_raw", "sum"), bh_sig=("sig_BH", "sum"),
                                    bh_sig_pct=("q_BH_exp_all_pct", lambda q: int((q < 0.05).sum()))).reset_index()
tot = T.groupby("exp").agg(m=("p", "size"), raw_sig=("sig_raw", "sum"), bh_sig=("sig_BH", "sum")).reset_index()
print(S.to_string(index=False))
print(tot.to_string(index=False))
S.to_csv(f"{OUT}/mult_summary.csv", index=False)
# tests that are raw-significant but lose significance after BH
lost = T[T.sig_raw & ~T.sig_BH]
print("\nraw p<0.05 but BH q>=0.05:", len(lost))
print(lost[["exp", "task", "layer", "sae", "family", "filter", "type", "metric", "condition", "estimator", "r", "est", "null",
            "lo95", "hi95", "p", "q_BH_exp_all"]].round(4).to_string(index=False))

"""STATS priority 3: CI audit of r5/tables/E18_illusion.csv and E19_steering.csv against an independent recomputation from
per_pair.parquet (no harness imports). 20 random rows per table (seeded). For each row we recompute
  flip_rate, nldr, prob_recovery and their gaps (full_A - cond) with 95% percentile pair-bootstrap CIs
     (a) same generator scheme + seed as the tables (np.random.default_rng(seed).integers(0, n, (B, n)), B = 10000) -> exact test
     (b) independent seed (B = 2000) -> Monte Carlo agreement
  random null (mean / p95 / max / n over the random-U seeds of the same cell, filter, condition, r)
  E18 only: illusion_diff = flip(full_along_U) - flip(feat_A_along_U), paired CI on pairs common to all non-random
     full/feat keys of the cell (as scripts/make_final.e9_paired -> harness/final.paired_diffs).
Output: team/stats_out/ci_audit2.csv, printed summary.
"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

sys.path.insert(0, "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/team/scripts")
from stats_common import OUT, RUNS, SP  # noqa: E402

COLS = ["pair_id", "split", "family", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed", "ans_A_id",
        "ans_B_id", "fit_families", "f_none", "f_A_correct", "f_pilot", "logit_A", "logit_B", "top1_id", "lse"]
KEY = ["condition", "estimator", "source", "r", "seed"]
EPS = None


def load(path, flt):
    t = pds.dataset(path).to_table(columns=COLS, filter=flt).to_pandas()
    fam = t.family.fillna("pilot").astype(str)
    t["family"] = np.where(fam.str.startswith("pilot"), "pilot", fam)
    t["fit_families"] = t.fit_families.fillna("pilot")
    for c in ("estimator", "source"):
        t[c] = t[c].fillna("")
    return t


def W_of(n, B, seed):
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    W = np.zeros((B + 1, n))
    W[0] = 1
    for b in range(B):
        W[b + 1] = np.bincount(idx[b], minlength=n)
    return W


def arrs(f, ansA):
    la, lb, lse = f.logit_A.to_numpy(float), f.logit_B.to_numpy(float), f.lse.to_numpy(float)
    return la - lb, np.exp(la - lse), (f.top1_id.to_numpy() == ansA).astype(float)


def mets(W, P, A, Bc, elig):
    tot = W.sum(1)
    m = lambda x: (W @ x) / tot
    out = {"nldr": (m(P[0]) - m(Bc[0])) / (m(A[0]) - m(Bc[0]) + 1e-8),
           "prob_recovery": (m(P[1]) - m(Bc[1])) / (m(A[1]) - m(Bc[1]) + 1e-8)}
    ne = W @ elig
    with np.errstate(invalid="ignore", divide="ignore"):
        out["flip_rate"] = np.where(ne > 0, (W @ (P[2] * elig)) / np.maximum(ne, 1e-12), np.nan)
    return out


def q(s):
    s = s[np.isfinite(s)]
    return (float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))) if len(s) else (np.nan, np.nan)


rng = np.random.default_rng(31)
rows = []
for exp, tabname in (("E18", "E18_illusion"), ("E19", "E19_steering")):
    tab = pd.read_csv(f"{SP}/r5/tables/{tabname}.csv")
    tab["estimator"] = tab.estimator.fillna("")
    tab["source"] = tab.source.fillna("")
    pick = tab.sample(n=20, random_state=int(rng.integers(1e9)))
    cache = {}
    for _, row in pick.iterrows():
        d = f"{RUNS}/{exp}/{row.run_hash}/"
        seed = json.load(open(d + "manifest.json"))["config"].get("seed", 0)
        ck = (row.task, int(row.layer), row.sae)
        if ck not in cache:
            cache[ck] = load(d + "per_pair.parquet", (pds.field("task") == row.task) & (pds.field("layer") == int(row.layer))
                             & (pds.field("sae") == row.sae))
        g = cache[ck]
        g = g[(g.split == "test") & (g.fit_families == row.fit_families) & (g.family == row.family)]
        sel = g[g[row["filter"]]]
        frames = {tuple(k): f.drop_duplicates("pair_id").set_index("pair_id") for k, f in sel.groupby(KEY, sort=False)}
        base = {k[0]: f for k, f in frames.items() if k[0] in ("clean_A", "clean_B", "full_A")}
        pairs = base["clean_A"].index.intersection(base["clean_B"].index).intersection(base["full_A"].index).sort_values()
        ca, cb = base["clean_A"].loc[pairs], base["clean_B"].loc[pairs]
        ansA = ca.ans_A_id.to_numpy()
        A, Bc = arrs(ca, ansA), arrs(cb, ansA)
        elig = (cb.top1_id.to_numpy() == cb.ans_B_id.to_numpy()).astype(float)
        FULL = arrs(base["full_A"].loc[pairs], ansA)
        key = (row.condition, row.estimator, row.source, int(row.r), int(row.seed))
        P = arrs(frames[key].reindex(pairs), ansA)
        res = {"table": tabname, "task": row.task, "layer": row.layer, "sae": row.sae, "family": row.family,
               "filter": row["filter"], "condition": row.condition, "estimator": row.estimator, "r": row.r,
               "n_pairs_tab": row.n_pairs, "n_pairs_me": len(pairs), "n_elig_tab": row.flip_n_eligible, "n_elig_me": int(elig.sum())}
        for B, s, tag in ((10000, seed, "exact"), (2000, 424242, "indep")):
            W = W_of(len(pairs), B, s)
            S = mets(W, P, A, Bc, elig)
            F = mets(W, FULL, A, Bc, elig)
            for met in ("flip_rate", "nldr", "prob_recovery"):
                res[f"{met}_me"] = S[met][0]
                res[f"gap_{met}_me"] = F[met][0] - S[met][0]
                lo, hi = q(S[met][1:])
                glo, ghi = q(F[met][1:] - S[met][1:])
                for nm, a, b in ((met, lo, hi), (f"gap_{met}", glo, ghi)):
                    res[f"d_{nm}_{tag}"] = max(abs(a - row[f"{nm}_lo"]), abs(b - row[f"{nm}_hi"]))
                    res[f"{nm}_lo_{tag}"], res[f"{nm}_hi_{tag}"] = a, b
            del W
        for met in ("flip_rate", "nldr", "prob_recovery"):
            res[f"d_point_{met}"] = abs(res[f"{met}_me"] - row[met])
            res[f"d_point_gap_{met}"] = abs(res[f"gap_{met}_me"] - row[f"gap_{met}"])
            res[f"{met}_tab"], res[f"{met}_lo_tab"], res[f"{met}_hi_tab"] = row[met], row[f"{met}_lo"], row[f"{met}_hi"]
        # random null: flip over the random-U seeds of the same condition and r
        rk = [k for k in frames if k[0] == row.condition and k[1] == "random" and int(k[3]) == int(row.r)]
        W1 = np.ones((1, len(pairs)))
        rv = np.array([mets(W1, arrs(frames[k].reindex(pairs), ansA), A, Bc, elig)["flip_rate"][0] for k in rk
                       if not frames[k].reindex(pairs).top1_id.isna().any()])
        res.update(random_n_me=len(rv), random_n_tab=row.random_n, random_mean_me=rv.mean() if len(rv) else np.nan,
                   random_mean_tab=row.random_mean, random_p95_me=np.nanpercentile(rv, 95) if len(rv) else np.nan,
                   random_p95_tab=row.random_p95, random_max_me=rv.max() if len(rv) else np.nan, random_max_tab=row.random_max)
        if exp == "E18" and row.condition in ("full_along_U", "feat_A_along_U"):
            # illusion diff, pairs common to clean_A, clean_B and all non-random full/feat keys of the cell
            fk = [k for k in frames if k[0] == "full_along_U" and k[1] != "random"]
            kk = fk + [("feat_A_along_U",) + k[1:] for k in fk if ("feat_A_along_U",) + k[1:] in frames]
            pp = base["clean_A"].index.intersection(base["clean_B"].index)
            for k in kk:
                pp = pp.intersection(frames[k].index)
            pp = pp.sort_values()
            ca2, cb2 = base["clean_A"].loc[pp], base["clean_B"].loc[pp]
            a2 = ca2.ans_A_id.to_numpy()
            A2, B2 = arrs(ca2, a2), arrs(cb2, a2)
            el2 = (cb2.top1_id.to_numpy() == cb2.ans_B_id.to_numpy()).astype(float)
            W = W_of(len(pp), 10000, seed)
            kf = ("full_along_U",) + key[1:]
            ke = ("feat_A_along_U",) + key[1:]
            dd = (mets(W, arrs(frames[kf].loc[pp], a2), A2, B2, el2)["flip_rate"]
                  - mets(W, arrs(frames[ke].loc[pp], a2), A2, B2, el2)["flip_rate"])
            lo, hi = q(dd[1:])
            res.update(illusion_diff_me=dd[0], illusion_diff_tab=row.illusion_diff, illusion_lo_me=lo,
                       illusion_lo_tab=row.illusion_diff_lo, illusion_hi_me=hi, illusion_hi_tab=row.illusion_diff_hi,
                       d_illusion=max(abs(dd[0] - row.illusion_diff), abs(lo - row.illusion_diff_lo), abs(hi - row.illusion_diff_hi)))
            del W
        rows.append(res)
        print(tabname, row.task, row.condition, row.estimator, row.r, row["filter"], "done", flush=True)

A = pd.DataFrame(rows)
A.to_csv(f"{OUT}/ci_audit2.csv", index=False)
pd.set_option("display.width", 250)
print("rows:", len(A), "n_pairs mismatches:", int((A.n_pairs_tab != A.n_pairs_me).sum()),
      "n_elig mismatches:", int((A.n_elig_tab != A.n_elig_me).sum()))
for t, g in A.groupby("table"):
    print(f"\n{t}: {len(g)} rows")
    for c in sorted(c for c in g if c.startswith("d_")):
        print(f"  max {c}: {g[c].max():.2e}")
    for c in ("random_n", "random_mean", "random_p95", "random_max"):
        print(f"  max |{c} me - tab|: {np.nanmax(np.abs(g[c + '_me'] - g[c + '_tab'])):.2e}")
print(A[["table", "task", "condition", "estimator", "r", "filter", "flip_rate_tab", "flip_rate_lo_tab", "flip_rate_hi_tab",
         "flip_rate_lo_exact", "flip_rate_hi_exact", "flip_rate_lo_indep", "flip_rate_hi_indep"]].round(4).to_string(index=False))

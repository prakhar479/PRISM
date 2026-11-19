"""Task 3: power / minimal detectable effects (MDE) for flip-rate contrasts, and whether the E17-E22 nulls are informative.
EXPLORATORY. alpha = 0.05 two-sided, power 0.80 unless stated. Outputs team/stats_out/power_*.csv.

Informative null := the 90% CI of the contrast lies inside +-10 points (TOST equivalence at the preregistered 10-point bar).
Underpowered null := CI includes 0 AND extends beyond +-10 points.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sst

sys.path.insert(0, "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/team/scripts")
from stats_common import OUT, RUNS  # noqa: E402
from stats_engine import Cell, boot_W, ci, load, samples  # noqa: E402
from stats_pp import run_dirs  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
za, zb = sst.norm.ppf(0.975), sst.norm.ppf(0.80)

# ------------------------------------------------------------------ 1. analytic MDE tables
rows = []
for n in (98, 150, 208, 450):
    for p1 in (0.0, 0.02, 0.10, 0.30, 0.50):
        # unpaired two-proportion (arcsine): smallest p2 > p1 with power 0.8
        h = (za + zb) * np.sqrt(2 / n)
        a = np.arcsin(np.sqrt(p1)) + h / 2
        p2 = np.sin(min(a, np.pi / 2)) ** 2
        rows.append({"n_per_arm": n, "design": "unpaired", "p_base": p1, "psi_discordant": np.nan, "MDE": p2 - p1})
    for psi in (0.05, 0.10, 0.20, 0.30, 0.50):
        f = lambda d: (za * np.sqrt(psi) + zb * np.sqrt(max(psi - d * d, 1e-12))) ** 2 / d ** 2 - n
        d = optimize.brentq(f, 1e-4, np.sqrt(psi) - 1e-6) if f(np.sqrt(psi) - 1e-6) < 0 else np.nan
        rows.append({"n_per_arm": n, "design": "paired (McNemar)", "p_base": np.nan, "psi_discordant": psi, "MDE": d})
M = pd.DataFrame(rows)
M.to_csv(f"{OUT}/power_mde_analytic.csv", index=False)
print("[1] analytic MDE (alpha .05 two-sided, power .80)")
print(M.pivot_table(index=["design", "p_base", "psi_discordant"], columns="n_per_arm", values="MDE", dropna=False).round(3)
      .dropna(how="all").to_string())


def paired_mde(psi, n):
    if psi <= 0:
        return np.nan
    f = lambda d: (za * np.sqrt(psi) + zb * np.sqrt(max(psi - d * d, 1e-12))) ** 2 / d ** 2 - n
    try:
        return optimize.brentq(f, 1e-4, np.sqrt(psi) - 1e-6)
    except ValueError:
        return np.nan


def contrast(name, s1, s2, h1, h2, elig, extra=None):
    d = s1 - s2
    lo, hi = ci(d[1:])
    lo90, hi90 = ci(d[1:], 0.90)
    se = np.std(d[1:], ddof=1)
    p = float(2 * sst.norm.sf(abs(d[0]) / se)) if se > 0 else (0.0 if d[0] != 0 else 1.0)
    e = elig > 0
    psi = float(np.mean(h1[e] != h2[e])) if h1 is not None else np.nan
    n = int(e.sum())
    r = {"contrast": name, "diff": d[0], "lo95": lo, "hi95": hi, "lo90": lo90, "hi90": hi90, "p": p, "n_elig": n,
         "psi_discordant": psi, "MDE80_paired": paired_mde(psi, n), "MDE_from_SE": (za + zb) * se,
         "equiv_10pts": bool(lo90 > -0.10 and hi90 < 0.10), "ci_excludes_0": bool(lo > 0 or hi < 0)}
    r.update(extra or {})
    return r


# ------------------------------------------------------------------ 2. E17 within-run contrasts (training length, lr, seed)
print("\n[2] E17 paired contrasts, RS L18, f_A_correct (B = 2000 pair bootstrap)")
d17 = run_dirs("E17")[0]
g = load(d17 + "per_pair.parquet")
g["estimator"] = g.estimator.fillna("")
g["source"] = g.source.fillna("")
C = Cell(g[g.family == "pilot"], "f_A_correct")
W = boot_W(C.n, 2000, 0)


def get(est, cond):
    k = [x for x in C.keys() if x[0] == cond and x[1] == est]
    a = C.arrays(k[0])
    return samples(W, *a, C.DA, C.DB, C.elig)["flip_rate"], a[1]


pairs = [("das1_e10", "das1", "epochs 10 vs 3"), ("das1_e30", "das1", "epochs 30 vs 3"), ("das2_e10", "das2", "epochs 10 vs 3"),
         ("das2_e30", "das2", "epochs 30 vs 3"), ("das4_e10", "das4", "epochs 10 vs 3"), ("das8_e10", "das8", "epochs 10 vs 3"),
         ("das1_e10_lr3e-2", "das1_e10", "lr 3e-2 vs 1e-2"), ("das2_e10_lr3e-2", "das2_e10", "lr 3e-2 vs 1e-2"),
         ("das1_s1", "das1", "seed 1 vs 0"), ("das1_s2", "das1", "seed 2 vs 0"), ("das2_s1", "das2", "seed 1 vs 0"),
         ("das2_s2", "das2", "seed 2 vs 0"), ("das4_s1", "das4", "seed 1 vs 0"), ("das8_s1", "das8", "seed 1 vs 0"),
         ("das1_e10_s1", "das1_e10", "seed 1 vs 0"), ("das1_e10_s2", "das1_e10", "seed 2 vs 0"),
         ("das2_e10_s1", "das2_e10", "seed 1 vs 0"), ("das2_e10_s2", "das2_e10", "seed 2 vs 0")]
R = []
for a, b, lab in pairs:
    for cond in ("full_along_U", "feat_A_along_U"):
        sa, ha = get(a, cond)
        sb, hb = get(b, cond)
        R.append(contrast(f"{a} - {b}", sa, sb, ha, hb, C.elig, {"exp": "E17", "kind": lab, "condition": cond}))
    # illusion diff-in-diff
    fa, hfa = get(a, "full_along_U")
    ea, _ = get(a, "feat_A_along_U")
    fb, _ = get(b, "full_along_U")
    eb, _ = get(b, "feat_A_along_U")
    R.append(contrast(f"{a} - {b}", fa - ea, fb - eb, None, None, C.elig, {"exp": "E17", "kind": lab, "condition": "illusion_diff"}))

# ------------------------------------------------------------------ 3. E20 (65k/262k) vs 16k on the same pairs (E9 run)
print("[3] E20 width contrasts vs E9 16k, same pairs, f_A_correct")
d9 = run_dirs("E9")[0]
d20 = run_dirs("E20")[0]
for task, L, s16, wides in (("add_chain", 18, "L18_16k_l074", ["L18_65k_l064"]),
                            ("entity_track", 12, "L12_16k_l082", ["L12_65k_l072", "L12_262k_l067"])):
    import pyarrow.dataset as pds
    g9 = load(d9 + "per_pair.parquet", (pds.field("task") == task) & (pds.field("sae") == s16))
    g9["estimator"], g9["source"] = g9.estimator.fillna(""), g9.source.fillna("")
    c9 = Cell(g9[g9.family == "pilot"], "f_A_correct")
    for sw in wides:
        g20 = load(d20 + "per_pair.parquet", (pds.field("task") == task) & (pds.field("sae") == sw))
        g20["estimator"], g20["source"] = g20.estimator.fillna(""), g20.source.fillna("")
        c20 = Cell(g20[g20.family == "pilot"], "f_A_correct")
        common = c9.pairs.intersection(c20.pairs).sort_values()
        i9, i20 = c9.pairs.get_indexer(common), c20.pairs.get_indexer(common)
        Wc = boot_W(len(common), 2000, 0)
        same_base = bool(np.allclose(c9.DA[i9], c20.DA[i20]) and np.allclose(c9.DB[i9], c20.DB[i20]))

        def S(c, ix, k):
            kk = [x for x in c.keys() if x[0] == k[0] and x[1] == k[1] and int(x[3]) == k[2]]
            if not kk:
                return None, None
            a = c.arrays(kk[0], common)
            full = c.arrays([x for x in c.keys() if x[0] == "full_A"][0], common)
            s = samples(Wc, *a, c.DA[ix], c.DB[ix], c.elig[ix])
            f = samples(Wc, *full, c.DA[ix], c.DB[ix], c.elig[ix])
            return {m: f[m] - s[m] for m in s}, a[1]

        for cond in ("feat_A_errB", "feat_A_meanPos"):
            g_w, hw = S(c20, i20, (cond, "", -1))
            g_n, hn = S(c9, i9, (cond, "", -1))
            for met in ("flip_rate", "nldr"):
                R.append(contrast(f"gap {cond} {sw} - 16k", g_w[met], g_n[met], hw if met == "flip_rate" else None, hn,
                                  c9.elig[i9], {"exp": "E20", "kind": f"width {task}", "condition": f"gap_{met}",
                                                "same_baselines": same_base}))
        for est in ("das4", "das8", "pca_pooled", "answer_sep"):
            for r in ((4,) if est == "das4" else (8,) if est == "das8" else (1, 2, 4, 8)):
                fw, _ = S(c20, i20, ("full_along_U", est, r))
                ew, _ = S(c20, i20, ("feat_A_along_U", est, r))
                fn, _ = S(c9, i9, ("full_along_U", est, r))
                en, _ = S(c9, i9, ("feat_A_along_U", est, r))
                if fw is None or fn is None or ew is None or en is None:
                    continue
                # gap(M) = full_A - M, so illusion diff full - feat = gap(feat) - gap(full)
                R.append(contrast(f"illusion {est} r{r} {sw} - 16k", ew["flip_rate"] - fw["flip_rate"],
                                  en["flip_rate"] - fn["flip_rate"], None, None, c9.elig[i9],
                                  {"exp": "E20", "kind": f"width {task}", "condition": "illusion_diff"}))
P = pd.DataFrame(R)
P.to_csv(f"{OUT}/power_contrasts.csv", index=False)
print(P[["exp", "kind", "contrast", "condition", "diff", "lo95", "hi95", "lo90", "hi90", "p", "n_elig", "psi_discordant",
         "MDE80_paired", "MDE_from_SE", "equiv_10pts", "ci_excludes_0"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ 4. DAS train interchange accuracy: binomial CIs (n_train)
print("\n[4] DAS fit validity: train interchange accuracy with Wilson 95% CI (n = n_train); rule: valid iff acc >= 0.5")
A = []
for e in ("E9", "E17", "E18", "E19", "E20", "E22"):
    for d in run_dirs(e):
        for f in glob.glob(d + "directions/*.json"):
            j = json.load(open(f))
            for est, v in j.items():
                meta = (v or {}).get("meta") or {}
                if "train_interchange_acc" not in meta:
                    continue
                n = int(meta.get("n_train", v.get("n_fit_rows", 0)))
                acc = float(meta["train_interchange_acc"])
                k = round(acc * n)
                lo, hi = (sst.binomtest(k, n).proportion_ci(method="wilson") if n else (np.nan, np.nan))
                A.append({"exp": e, "file": f.split("/")[-1][:-5], "estimator": est, "n_train": n, "acc": acc,
                          "lo": lo, "hi": hi, "valid": acc >= 0.5, "ci_straddles_0.5": lo < 0.5 <= hi or lo <= 0.5 < hi})
A = pd.DataFrame(A).drop_duplicates(["file", "estimator", "acc"])
A.to_csv(f"{OUT}/power_das_acc.csv", index=False)
print(A.round(3).to_string(index=False))

# ------------------------------------------------------------------ 5. nulls in the E19/E22 tables (from stats_multiplicity output)
try:
    T = pd.read_csv(f"{OUT}/mult_tests.csv")
except FileNotFoundError:
    T = None
if T is not None:
    print("\n[5a] E22 transfer (English-fit minus native-fit flip), f_A_correct: informative vs underpowered")
    t = T[(T.type == "transfer") & (T["filter"] == "f_A_correct")].copy()
    t["verdict"] = np.where(t.lo95 > 0, "English better", np.where(t.hi95 < 0, "native better",
                            np.where((t.lo90 > -0.10) & (t.hi90 < 0.10), "null, equivalent within 10 pts", "null, underpowered")))
    print(t[["task", "family", "condition", "estimator", "r", "n_pairs", "native", "english", "est", "lo95", "hi95", "lo90",
             "hi90", "p", "q_BH_exp_all", "verdict"]].round(3).to_string(index=False))
    t.to_csv(f"{OUT}/power_e22_transfer.csv", index=False)
    print("\n[5b] E19: best dose per (task, estimator, rank), f_A_correct: does the upper 95% CI reach 0.50 (Q2 bar)?")
    q = T[(T.type == "q2") & (T["filter"] == "f_A_correct")].copy()
    best = q.loc[q.groupby(["task", "estimator", "r"]).est.idxmax()]
    print(best[["task", "estimator", "r", "condition", "n_pairs", "est", "lo95", "hi95"]].round(3).to_string(index=False))
    best.to_csv(f"{OUT}/power_e19_best.csv", index=False)

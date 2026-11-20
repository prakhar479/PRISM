"""Task 5: audit the bootstrap CIs of the standard tables against an independent recomputation from per_pair.parquet.
For a random sample of rows per table: recompute point estimates (flip, NLDR, gaps) and 95% pair-bootstrap CIs with
(a) the same generator scheme and seed (exact-match test, B = 2000 and 10000) and (b) an independent seed (Monte Carlo check).
Output: team/stats_out/ci_audit.csv."""
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".layout/team/scripts")
from stats_common import OUT, RUNS, SP  # noqa: E402
from stats_engine import Cell, boot_W, ci, load, samples  # noqa: E402

pd.set_option("display.width", 250)
TABLES = [(f"{SP}/hw/tables/final/E9_conditions.csv", "E9"), (f"{SP}/hw/tables/final/E2r_conditions.csv", "E2r"),
          (f"{SP}/r5/tables/E17_conditions.csv", "E17"), (f"{SP}/r5/tables/E18_conditions.csv", "E18"),
          (f"{SP}/r5/tables/E20_conditions.csv", "E20"), (f"{SP}/r5/tables/E22_conditions.csv", "E22"),
          (f"{SP}/r5/tables/E19_conditions.csv", "E19")]
rng = np.random.default_rng(1)
rows = []
for path, exp in TABLES:
    tab = pd.read_csv(path)
    for c in ("estimator", "source"):
        tab[c] = tab[c].fillna("")
    tab = tab[~tab.condition.isin(["clean_B", "full_A", "clean_A"])]
    pick = tab.sample(n=min(8, len(tab)), random_state=int(rng.integers(1e9)))
    for _, row in pick.iterrows():
        d = glob.glob(f"{RUNS}/{exp}/{row.run_hash}/")[0]
        seed = json.load(open(d + "manifest.json"))["config"].get("seed", 0)
        import pyarrow.dataset as pds
        f = (pds.field("task") == row.task) & (pds.field("layer") == int(row.layer)) & (pds.field("sae") == row.sae)
        g = load(d + "per_pair.parquet", f)
        g = g[(g.fit_families == row.fit_families) & (g.family == row.family)]
        g["estimator"] = g.estimator.fillna("")
        g["source"] = g.source.fillna("")
        cell = Cell(g, row["filter"])
        key = (row.condition, row.estimator, row.source, int(row.r), int(row.seed))
        arr = cell.arrays(key)
        full = cell.arrays(("full_A", "", "", -1, -1))
        if arr is None or full is None:
            # keys in per_pair may hold NaN for estimator/source: try loose match
            cand = [k for k in cell.keys() if k[0] == row.condition and str(k[1]) == row.estimator and int(k[3]) == int(row.r)
                    and int(k[4]) == int(row.seed)]
            full_c = [k for k in cell.keys() if k[0] == "full_A"]
            arr = cell.arrays(cand[0]) if cand else None
            full = cell.arrays(full_c[0]) if full_c else None
        if arr is None:
            rows.append({"table": exp, "condition": row.condition, "estimator": row.estimator, "status": "key not found"})
            continue
        res = {"table": exp, "run": row.run_hash, "task": row.task, "layer": row.layer, "sae": row.sae, "family": row.family,
               "filter": row["filter"], "condition": row.condition, "estimator": row.estimator, "r": row.r, "seed": row.seed,
               "n_pairs_tab": row.n_pairs, "n_pairs_me": cell.n, "n_elig_tab": row.flip_n_eligible, "n_elig_me": int(cell.elig.sum())}
        for B, s, tag in ((2000, seed, "B2k"), (10000, seed, "B10k"), (2000, 987654, "indep")):
            W = boot_W(cell.n, B, s)
            S = samples(W, *arr, cell.DA, cell.DB, cell.elig)
            F = samples(W, *full, cell.DA, cell.DB, cell.elig)
            for met in ("flip_rate", "nldr"):
                lo, hi = ci(S[met][1:])
                glo, ghi = ci(F[met][1:] - S[met][1:])
                res[f"{met}_me"] = S[met][0]
                res[f"gap_{met}_me"] = F[met][0] - S[met][0]
                res[f"{met}_lo_{tag}"], res[f"{met}_hi_{tag}"] = lo, hi
                res[f"gap_{met}_lo_{tag}"], res[f"gap_{met}_hi_{tag}"] = glo, ghi
        for met in ("flip_rate", "nldr"):
            res[f"{met}_tab"], res[f"{met}_lo_tab"], res[f"{met}_hi_tab"] = row[met], row[f"{met}_lo"], row[f"{met}_hi"]
            res[f"gap_{met}_tab"], res[f"gap_{met}_lo_tab"], res[f"gap_{met}_hi_tab"] = row[f"gap_{met}"], row[f"gap_{met}_lo"], row[f"gap_{met}_hi"]
        res["status"] = "ok"
        rows.append(res)
A = pd.DataFrame(rows)
ok = A[A.status == "ok"].copy()
for met in ("flip_rate", "nldr", "gap_flip_rate", "gap_nldr"):
    ok[f"d_point_{met}"] = (ok[f"{met}_me"] - ok[f"{met}_tab"]).abs()
    for tag in ("B2k", "B10k", "indep"):
        ok[f"d_ci_{met}_{tag}"] = np.maximum((ok[f"{met}_lo_{tag}"] - ok[f"{met}_lo_tab"]).abs(),
                                             (ok[f"{met}_hi_{tag}"] - ok[f"{met}_hi_tab"]).abs())
A = A.merge(ok[[c for c in ok if c.startswith("d_")] + ["table", "condition", "estimator", "r", "seed", "filter", "run"]],
            on=["table", "condition", "estimator", "r", "seed", "filter", "run"], how="left")
A.to_csv(f"{OUT}/ci_audit.csv", index=False)
print("rows audited:", len(ok), "not found:", int((A.status != "ok").sum()))
print("n_pairs / n_eligible mismatches:", int((ok.n_pairs_tab != ok.n_pairs_me).sum()), int((ok.n_elig_tab != ok.n_elig_me).sum()))
summ = {c: float(ok[c].max()) for c in ok if c.startswith("d_")}
for k, v in summ.items():
    print(f"max {k}: {v:.6f}")
print("\nper table, max CI-bound discrepancy (exact-seed B=2000 / B=10000 / independent seed), over flip, nldr and their gaps:")
for t, g in ok.groupby("table"):
    print(t, len(g), {tag: float(g[[f"d_ci_{m}_{tag}" for m in ("flip_rate", "nldr", "gap_flip_rate", "gap_nldr")]].max().max())
                      for tag in ("B2k", "B10k", "indep")},
          "max point diff", float(g[[f"d_point_{m}" for m in ("flip_rate", "nldr", "gap_flip_rate", "gap_nldr")]].max().max()))
print(ok[["table", "condition", "estimator", "r", "filter", "flip_rate_tab", "flip_rate_lo_tab", "flip_rate_hi_tab",
          "flip_rate_lo_B10k", "flip_rate_hi_B10k", "flip_rate_lo_indep", "flip_rate_hi_indep", "gap_nldr_tab", "gap_nldr_lo_tab",
          "gap_nldr_lo_B10k", "gap_nldr_hi_tab", "gap_nldr_hi_B10k"]].round(4).to_string(index=False))

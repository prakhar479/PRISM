#!/usr/bin/env python
"""DEV: E4 width persistence (PREREGISTRATION.md) from ingested E4 runs. No model.

    python scripts/make_e4.py [--n-boot 10000] [--out tables]

Preregistered rule: "evaluated at matched realized L0 (within 25%) between 16k and the widest available SAE; reported as a ratio
with CI; the pilot's 60% threshold is applied to the CI, and a CI straddling 60% is reported as inconclusive."
Here, per (task, layer, filter, metric, baseline) and per wider SAE:
    gap_w   = M(full_A) - M(baseline) with SAE w (baseline feat_A_errB = the pilot's features-only gap; meanPos / zero as context)
    ratio_w = gap_w / gap_16k, with a 95% percentile CI from ONE paired bootstrap over the test pairs both cells share
    realized L0 = mean active latents on the patched test positions (cell checks `mean_l0_test`), also on generic text
    l0_matched  = |L0_w / L0_16k - 1| <= 0.25 (test positions)
The preregistered row per (task, layer, filter, metric) is the widest width with l0_matched = True and baseline feat_A_errB
(`prereg_row`); `rule_outcome` applies the 60% rule to its CI: '>= 60%', '< 60%', or 'inconclusive (CI straddles 60%)'; when the 16k gap's own
CI includes 0 the ratio has no meaning and the outcome is 'undefined (16k gap CI includes 0)'.
Metrics: flip_rate (primary), nldr (the pilot's metric, which the 60% threshold came from), prob_recovery. Same metric code as E8.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from harness import e8 as E
from harness import stats as St
from harness.profile import REPO_ROOT, load_profile

THRESH = 0.60
L0_TOL = 0.25
BASES = ("feat_A_errB", "feat_A_meanPos", "feat_A_zero")


def width_num(sae: str) -> int:
    w = sae.split("_")[1]
    return int(w[:-1]) * (1024 if w.endswith("k") else 1024 ** 2)


def cell_l0(summary: dict) -> dict:
    out = {}
    for c in summary.get("cells", []):
        sae = c["cell"].split("_L", 1)[1]
        sae = "L" + sae
        t = c.get("mean_l0_test")
        out[(c["cell"].rsplit("_L", 1)[0], sae)] = {"l0_test": float(np.mean(t)) if t else np.nan,
                                                    "l0_generic": (c.get("generic") or {}).get("mean_l0", np.nan)}
    return out


def analyse(pp: pd.DataFrame, l0: dict, flt: str, n_boot: int, seed: int):
    rows = []
    for (task, layer), g in pp.groupby(["task", "layer"]):
        g = g[g[flt]]
        saes = sorted(g.sae.unique(), key=width_num)
        ref = [s for s in saes if s.startswith(f"L{layer}_16k")]
        if not ref or len(saes) < 2:
            continue
        ref = ref[0]
        frames = {(s, c): f.set_index("pair_id") for (s, c), f in g[g.condition.isin(("clean_A", "clean_B", "full_A") + BASES)].groupby(["sae", "condition"])}
        for s in saes:
            if s == ref:
                continue
            need = [(x, c) for x in (ref, s) for c in ("clean_A", "clean_B", "full_A", "feat_A_errB")]
            if any(k not in frames for k in need):
                continue
            pairs = None
            for (x, c), f in frames.items():
                if x in (ref, s):
                    pairs = f.index if pairs is None else pairs.intersection(f.index)
            pairs = pairs.sort_values()
            n = len(pairs)
            W = E.weights([np.arange(n)] + list(St.boot_indices(n, n_boot, seed)), n)
            res = {}
            for x in (ref, s):
                base_A = frames[(x, "clean_A")].loc[pairs]
                ansA, ansB = base_A.ans_A_id.to_numpy(), base_A.ans_B_id.to_numpy()
                A, B = E.Arr(base_A, ansA), E.Arr(frames[(x, "clean_B")].loc[pairs], ansA)
                elig = (frames[(x, "clean_B")].loc[pairs].top1_id.to_numpy() == ansB).astype(float)
                full = E.metric_samples(E.Arr(frames[(x, "full_A")].loc[pairs], ansA), A, B, elig, W)
                for b in BASES:
                    if (x, b) in frames:
                        v = E.metric_samples(E.Arr(frames[(x, b)].loc[pairs], ansA), A, B, elig, W)
                        res[(x, b)] = {m: full[m] - v[m] for m in E.METRICS}
                res[(x, "n_elig")] = int(elig.sum())
            lr, ls = l0.get((task, ref), {}), l0.get((task, s), {})
            for b in BASES:
                if (ref, b) not in res or (s, b) not in res:
                    continue
                for m in E.METRICS:
                    g0, gw = res[(ref, b)][m], res[(s, b)][m]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = np.where(np.abs(g0) > 1e-12, gw / g0, np.nan)
                    lo, hi = E._ci(r[1:])
                    g0lo, g0hi = E._ci(g0[1:])
                    rows.append({"task": task, "layer": int(layer), "filter": flt, "metric": m, "baseline": b,
                                 "ref_sae": ref, "sae": s, "width": s.split("_")[1], "n_pairs": n, "flip_n_eligible": res[(ref, "n_elig")],
                                 "gap_16k": float(g0[0]), "gap_16k_lo": g0lo, "gap_16k_hi": g0hi, "gap_w": float(gw[0]), "ratio": float(r[0]), "ratio_lo": lo, "ratio_hi": hi,
                                 "l0_test_16k": lr.get("l0_test"), "l0_test_w": ls.get("l0_test"),
                                 "l0_generic_16k": lr.get("l0_generic"), "l0_generic_w": ls.get("l0_generic")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["l0_ratio"] = df.l0_test_w / df.l0_test_16k
    df["l0_matched"] = (df.l0_ratio - 1).abs() <= L0_TOL
    df["width_n"] = df.sae.map(width_num)
    df["prereg_row"] = False
    for _, g in df[(df.baseline == "feat_A_errB") & df.l0_matched].groupby(["task", "layer", "filter", "metric"]):
        widest = g[g.width_n == g.width_n.max()]
        df.loc[widest.index, "prereg_row"] = True

    def outcome(r):
        if not r.prereg_row:
            return ""
        if not (np.isfinite(r.gap_16k_lo) and (r.gap_16k_lo > 0 or r.gap_16k_hi < 0)):
            return "undefined (16k gap CI includes 0)"
        if not np.isfinite(r.ratio_lo):
            return "no CI"
        return ">= 60%" if r.ratio_lo >= THRESH else ("< 60%" if r.ratio_hi < THRESH else "inconclusive (CI straddles 60%)")
    df["rule_outcome"] = df.apply(outcome, axis=1)
    return df.drop(columns="width_n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "tables"))
    a = ap.parse_args()
    root = load_profile(a.profile).root("runs_root")
    out = []
    for man in sorted(root.glob("E4/*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("smoke") or m.get("tiny") or m.get("status") != "complete":
            continue
        pp = pd.read_parquet(man.parent / "per_pair.parquet")
        l0 = cell_l0(json.loads((man.parent / "summary.json").read_text()))
        for flt in m["config"]["filters"]:
            d = analyse(pp, l0, flt, a.n_boot, m["config"].get("seed", 0))
            if not d.empty:
                d.insert(0, "run_hash", m["run_hash"])
                out.append(d)
    if not out:
        print("no complete E4 runs")
        return 1
    df = pd.concat(out, ignore_index=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "E4_width_persistence.csv", index=False)
    print(f"{len(df)} rows -> {Path(a.out) / 'E4_width_persistence.csv'}; preregistered rows: {int(df.prereg_row.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""DEV: tables and verdicts of the batch-2 repair experiments (branch batch2-repair; docs/NEXT_BATCH_PLAN.md H1, H2, H5, H6).

    python scripts/make_repair.py --runs results_server/pilot-x2-server/runs --out analysis/batch2_repair
    python scripts/make_repair.py --runs tests/_tiny_runs --out /tmp/repair --boot 20          # plumbing check

Reads P23-P27 (per_pair.parquet) and the PF fidelity runs. A cell's dictionary is `<base>__<add-on>` (harness.dicts.parse_addon).
  gap      NLDR(full_A) - NLDR(feat_A_errB): the task state lost when only the dictionary's reconstruction is copied from the clean run
  closure  (gap_base - gap_addon) / gap_base: the share of that gap the add-on removes (pair bootstrap CI); NaN when gap_base < GAP_MIN
           (a ratio with a tiny denominator says nothing: reported, never counted)
  capture  NLDR(feat_A_along_U) / NLDR(full_along_U), mean over the independent DAS seeds; dcapture = capture - capture of the base
  dCE      ce_delta(add-on) - ce_delta(base) from the PF job (held-out generic text; < 0 = the add-on lowers the splice cost)
Outputs (<out>/): repair_cells.csv, repair_curves.csv (closure by family, oracle flag and k), REPAIR.md (verdicts).
Decision rules (written before the run, docs/NEXT_BATCH_PLAN.md section 5):
  H1  per task with a stable gap, at k in {2, 8, 32}: the best of sens / fis (predicted) closes >= 2x what var closes (floor 0.05) and exceeds
      the upper CI of rnd; the task holds if that is true at >= 2 of the 3 k; go if >= 2 tasks hold and dCE of the winning add-on <= +0.01.
  H2  decoder problem if cmlp closes >= 0.5 of what mlp closes and clin >= 0.5 of cmlp; information problem if mlp closes >= 2x cmlp.
  H6  stacks if klft + sens32 closes >= max(klft, sens32) + 0.05 (vs the 16k Gemma Scope base).
Every difference below the protocol noise (~0.1 in capture, catalogue S10) is a tie."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import make_pilot as mp
from harness import dicts as Dct
from harness import metrics as M

GAP_MIN = 0.05
CAP = "capture_feat_A_along_U"
EXPS = ("P23", "P27", "P25", "P26", "P24")


def split_key(sae: str):
    base, _, addon = sae.partition("__")
    return base, addon


def closure_boot(cb: "mp.Cell", ca: "mp.Cell", n_boot: int, seed: int = 0):
    """(point, lo, hi) of (gap_base - gap_addon) / gap_base over the pairs both cells have; gap = NLDR(full_A) - NLDR(feat_A_errB)."""
    rb, ra, rf = cb.rows("feat_A_errB"), ca.rows("feat_A_errB"), cb.rows("full_A")
    common = cb.pairs.intersection(rb.index).intersection(ra.index).intersection(rf.index)
    if len(common) < 5:
        return (np.nan, np.nan, np.nan)
    pos = cb.pairs.get_indexer(common)
    Pb, Pa, Pf = mp.outs(rb.loc[common]), mp.outs(ra.loc[common]), mp.outs(rf.loc[common])
    A, B, aA, aB = cb.A.take(pos), cb.B.take(pos), cb.ansA[pos], cb.ansB[pos]

    def f(idx):
        n = lambda P: M.per_condition(P.take(idx), A.take(idx), B.take(idx), aA[idx], aB[idx])["nldr"]
        full, base, add = n(Pf), n(Pb), n(Pa)
        gap0 = full - base
        return (add - base) / gap0 if abs(gap0) >= GAP_MIN else np.nan
    pt = f(np.arange(len(common)))
    rng = np.random.default_rng(seed)
    bs = np.array([f(rng.integers(0, len(common), len(common))) for _ in range(n_boot)]) if n_boot else np.array([])
    bs = bs[np.isfinite(bs)]
    return (pt, *(np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)))


def cell_rows(rec, n_boot, exp):
    """One row per cell (task, layer, base dictionary, add-on) with gap, closure, capture."""
    cells = {(m["task"], m["layer"], m["sae"]): c for m, c in mp.cells(rec)}
    cap = mp.capture_table(rec, sorted(e for e in set(rec["pp"][rec["pp"].condition == "feat_A_along_U"].estimator) if e.startswith("das")),
                           ("feat_A_along_U",), 20)
    capm = cap.groupby(["task", "layer", "sae"])[CAP].mean().to_dict() if len(cap) else {}
    capsd = cap.groupby(["task", "layer", "sae"])[CAP].std().fillna(0.0).to_dict() if len(cap) else {}
    refs = {}
    for (task, layer, sae) in cells:
        b, a = split_key(sae)
        if not a and "klft" not in b and (task, layer) not in refs:
            refs[(task, layer)] = sae
    rows = []
    for (task, layer, sae), c in cells.items():
        b, a = split_key(sae)
        full, errb = c.value("full_A"), c.value("feat_A_errB")
        row = {"exp": exp, "task": task, "layer": layer, "sae": sae, "base": b, "addon": a, "n_pairs": len(c.pairs),
               "gap": full - errb, "gap_flip": c.value("full_A", key="flip_rate") - c.value("feat_A_errB", key="flip_rate"),
               "errB_only_nldr": c.value("feat_B_errA"), "capture": capm.get((task, layer, sae), np.nan),
               "capture_sd_seed": capsd.get((task, layer, sae), np.nan)}
        if a:
            p = Dct.parse_addon(a)
            row.update(kind=p["kind"], k=p["k"], oracle=bool(p["oracle"]))
        else:
            row.update(kind="base", k=0, oracle=False)
        own = cells.get((task, layer, f"{b}"))
        ref = cells.get((task, layer, refs.get((task, layer), "")))
        row["closure_own"] = row["closure_own_lo"] = row["closure_own_hi"] = np.nan
        row["closure_ref"] = row["closure_ref_lo"] = row["closure_ref_hi"] = np.nan
        if a and own is not None:
            row["closure_own"], row["closure_own_lo"], row["closure_own_hi"] = closure_boot(own, c, n_boot)
        if a and ref is not None:
            row["closure_ref"], row["closure_ref_lo"], row["closure_ref_hi"] = closure_boot(ref, c, n_boot)
        rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        bc = t[t.addon == ""].set_index(["task", "layer", "sae"]).capture
        t["dcapture"] = [r.capture - bc.get((r.task, r.layer, r.base), np.nan) if r.addon else np.nan for r in t.itertuples()]
    return t


def join_fidelity(t: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    if pf is None or not len(pf) or not len(t):
        return t
    cols = [c for c in ("ce_delta", "fvu", "eps_over_x", "l0", "splice_over_noise", "recovered_mean") if c in pf]
    f = pf.rename(columns={"dict": "sae"})[["sae"] + cols].drop_duplicates("sae")
    t = t.merge(f, on="sae", how="left")
    b = f.rename(columns={c: f"{c}_base" for c in cols}).rename(columns={"sae": "base"})
    t = t.merge(b, on="base", how="left")
    for c in ("ce_delta", "fvu", "splice_over_noise"):
        if c in cols:
            t[f"d_{c}"] = np.where(t.addon != "", t[c] - t[f"{c}_base"], np.nan)
    return t


def h1_rule(t: pd.DataFrame, ks, need_k):
    """Per (task, layer) with a stable gap: does a predicted sens / fis beat var and rnd at >= need_k of the k values? -> (rows, verdict)."""
    rows, holds = [], 0
    stable = 0
    for (task, layer), g in t[(t.addon != "") & ~t.oracle].groupby(["task", "layer"]):
        base_gap = t[(t.task == task) & (t.layer == layer) & (t.addon == "")].gap
        if not len(base_gap) or base_gap.min() < GAP_MIN:
            rows.append({"task": task, "layer": layer, "stable_gap": False, "holds": False, "passes": 0})
            continue
        stable += 1
        passes, dce = 0, []
        for k in ks:
            gk = g[g.k == k]
            c = lambda kind, col="closure_ref": gk[gk.kind == kind][col].max() if (gk.kind == kind).any() else np.nan
            best = max([x for x in (c("sens"), c("fis")) if np.isfinite(x)] or [np.nan])
            var, rnd_hi = c("var"), c("rnd", "closure_ref_hi")
            ok = np.isfinite(best) and best >= 2 * max(var if np.isfinite(var) else 0.0, 0.05) and (not np.isfinite(rnd_hi) or best > rnd_hi)
            passes += int(ok)
            if ok and "d_ce_delta" in gk:
                dce.append(float(gk[gk.kind.isin(["sens", "fis"])].d_ce_delta.min()))
        h = passes >= need_k and (not dce or min(dce) <= 0.01)
        holds += int(h)
        rows.append({"task": task, "layer": layer, "stable_gap": True, "passes": passes, "holds": bool(h)})
    verdict = "UNCLEAR (fewer than 2 (task, layer) cells have a stable gap)" if stable < 2 else \
        ("GO" if holds >= 2 else "NO-GO")
    return pd.DataFrame(rows), f"{verdict} ({holds} of {stable} stable cells hold)"


def h2_verdict(t: pd.DataFrame):
    out = []
    for (task, layer), g in t[(t.exp == "P27") & (t.addon != "") & ~t.sae.str.contains("klft")].groupby(["task", "layer"]):
        v = {a: float(g[g.addon == a].closure_ref.iloc[0]) for a in ("clin", "mlp1024", "cmlp1024", "lin", "sens32") if (g.addon == a).any()}
        if not {"clin", "mlp1024", "cmlp1024"} <= set(v) or any(not np.isfinite(v[x]) for x in ("clin", "mlp1024", "cmlp1024")):
            out.append((task, v, "gap too small or missing"))
            continue
        m, cm, cl = v["mlp1024"], v["cmlp1024"], v["clin"]
        kind = "decoder problem (the codes hold it, a refit decoder recovers it)" if cm >= 0.5 * m and cl >= 0.5 * cm and m > 0.1 else \
            "information problem (x holds more than the codes)" if m >= 2 * max(cm, 0.05) and m > 0.1 else "mixed / none recovered"
        out.append((task, v, kind))
    return out


def h6_verdict(t: pd.DataFrame):
    """Per (task, layer) of P27: (task, klft closure, sens32 closure, klft + sens32 closure, stacks?); closures vs the Gemma Scope 16k base."""
    out = []
    for (task, layer), g in t[t.exp == "P27"].groupby(["task", "layer"]):
        kl = g.sae.str.contains("klft")
        base, klb = g[(g.addon == "") & ~kl], g[(g.addon == "") & kl]
        sens, klsens = g[(g.addon == "sens32") & ~kl], g[(g.addon == "sens32") & kl]
        if min(len(base), len(klb), len(sens), len(klsens)) == 0 or base.gap.iloc[0] < GAP_MIN:
            continue
        kl_closure = float((base.gap.iloc[0] - klb.gap.iloc[0]) / base.gap.iloc[0])
        s_, ks_ = float(sens.closure_ref.iloc[0]), float(klsens.closure_ref.iloc[0])
        out.append((task, kl_closure, s_, ks_, bool(ks_ >= max(kl_closure, s_) + 0.05)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="analysis/batch2_repair")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    runs = mp.load_runs(Path(a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pf = {}
    for rec in runs.get("PF", []):
        f = rec["dir"] / "fidelity.csv"
        if f.exists():
            cid = rec["manifest"]["config"].get("identity", {}).get("config")
            pf[cid] = pd.concat([pf[cid], pd.read_csv(f)]) if cid in pf else pd.read_csv(f)
    report = ["# Batch-2 repair (exploratory, post hoc; scripts/make_repair.py)", "",
              f"closure = share of the feature-only gap removed (NaN when the base gap < {GAP_MIN}); oracle = the true error projected on the subspace; "
              "sens / fis / var / rnd = the generic-text Fisher-weighted / Fisher / variance / random subspace with a linear predictor.", ""]
    tables = {}
    for exp in EXPS:
        if exp not in runs:
            report += [f"## {exp}: no complete run", ""]
            continue
        try:
            t = join_fidelity(cell_rows(runs[exp], a.boot, exp), pf.get(exp))
        except Exception as ex:  # noqa: BLE001 - one experiment's failure must not hide the others
            report += [f"## {exp}: analysis failed ({type(ex).__name__}: {ex})", ""]
            continue
        tables[exp] = t
        t.to_csv(out / f"repair_cells_{exp}.csv", index=False)
        cu = t[t.addon != ""].groupby(["exp", "task", "layer", "base", "kind", "oracle", "k"]).agg(
            closure=("closure_ref", "mean"), lo=("closure_ref_lo", "mean"), hi=("closure_ref_hi", "mean"),
            capture=("capture", "mean"), dcapture=("dcapture", "mean")).reset_index()
        if "d_ce_delta" in t:
            cu = cu.merge(t[t.addon != ""][["task", "layer", "base", "kind", "oracle", "k", "d_ce_delta"]].groupby(
                ["task", "layer", "base", "kind", "oracle", "k"]).d_ce_delta.mean().reset_index(), on=["task", "layer", "base", "kind", "oracle", "k"], how="left")
        cu.to_csv(out / f"repair_curves_{exp}.csv", index=False)
        report += [f"## {exp}", f"{len(t)} cells; base gaps: " + ", ".join(
            f"{r.task} L{r.layer} {r.gap:.3f}" for r in t[t.addon == ""].sort_values(["task", "layer"]).itertuples()), ""]
        if exp in ("P23", "P24", "P25", "P26"):
            ks = (2, 8, 32) if exp == "P23" else (8, 32) if exp in ("P24", "P26") else (32,)
            rows, verdict = h1_rule(t, ks, 2 if exp == "P23" else 1)
            report += [f"- **H1 rule on {exp}** (k = {', '.join(map(str, ks))}): {verdict}"]
            for r in rows.itertuples():
                report.append(f"  - {r.task} L{r.layer}: " + ("gap too small" if not r.stable_gap else f"passes at {r.passes} k -> {'holds' if r.holds else 'does not hold'}"))
            sub = cu[~cu.oracle & cu.k.isin(ks)] if len(cu) else cu
            if len(sub):
                piv = sub.pivot_table(index=["task", "layer", "k"], columns="kind", values="closure").round(2)
                report += ["", "  closure (predicted) by kind:", "  ```", *("  " + ln for ln in piv.to_string().splitlines()), "  ```"]
            orc = cu[cu.oracle & cu.k.isin(ks)] if len(cu) else cu
            if len(orc):
                piv = orc.pivot_table(index=["task", "layer", "k"], columns="kind", values="closure").round(2)
                report += ["  closure (oracle projection of the true error) by kind:", "  ```", *("  " + ln for ln in piv.to_string().splitlines()), "  ```"]
            report.append("")
        if exp == "P27":
            report.append("- **H2 (decoder vs information)**")
            for task, v, kind in h2_verdict(t):
                report.append(f"  - {task}: " + ", ".join(f"{k} {x:.2f}" for k, x in v.items()) + f" -> {kind}")
            report.append("- **H6 (stacking with KL+MSE fine-tuning)**")
            for task, kl, s, kls, ok in h6_verdict(t):
                report.append(f"  - {task}: klft closes {kl:.2f}, sens32 {s:.2f}, klft + sens32 {kls:.2f} -> {'stacks' if ok else 'klft absorbs it / no gain'}")
            report.append("")
    (out / "REPAIR.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

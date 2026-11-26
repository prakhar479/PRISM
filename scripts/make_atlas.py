#!/usr/bin/env python
"""DEV: tables and verdicts of the batch-2 atlas (branch batch2-atlas; docs/NEXT_BATCH_PLAN.md H3, H4, H5) from its finished runs.

    python scripts/make_atlas.py --runs results_server/pilot-x1-server/runs --out analysis/batch2_atlas
    python scripts/make_atlas.py --runs tests/_tiny_runs --out /tmp/atlas --boot 20          # plumbing check

Reads P20 / P21 / P22 / P28 (per_pair.parquet) and the PF fidelity runs. Definitions (scripts/make_pilot.py): capture =
NLDR(feat_A_along_U) / NLDR(full_along_U) with U the held-out-validated DAS subspace (one value per independent DAS seed, rank 8;
refusal: rank 1 and 4 and the difference-of-means direction); gap = NLDR(full_A) - NLDR(feat_A_errB).
Outputs (<out>/): atlas_capture.csv (one row per task x layer x dictionary x seed), atlas_cells.csv (seed mean / sd / min / max and
the gap), atlas_curves.csv (mean capture by layer), atlas_metrics.csv (Spearman of capture with each fidelity metric),
atlas_replicates.csv (independently trained dictionaries at the same layer), ATLAS.md (verdicts).
Decision rules (written before the run, docs/NEXT_BATCH_PLAN.md section 5):
  H3  the capture minimum of a task falls in the middle third of depth (layers 9-16 of 26) and the dip (median of the outer layers minus the
      minimum) exceeds the largest seed sd at the minimum; go if that holds for >= 2 of 3 tasks in a model.
  H4  go if a metric other than FVU has |Spearman| >= 0.7 with capture and beats FVU by >= 0.05; no-go if FVU already has |rho| >= 0.7 and is
      not beaten; otherwise unclear. Computed pooled and after removing each (task, layer) mean (within-layer).
Every difference below the protocol noise (~0.1, catalogue S10) is a tie."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import make_pilot as mp

N_LAYERS = 26
EFFECT_MIN = 0.15        # a subspace whose own write-back moves the output by less than this (NLDR) has no capture to report: the ratio is noise
METRICS = ["fvu", "eps_over_x", "ce_delta", "recovered_mean", "l0", "splice_over_noise"]
SEED_RE = re.compile(r"_s(\d+)$")
CAP = "capture_feat_A_along_U"


def depth(layer, n=N_LAYERS):
    return layer / (n - 1)


def capture_rows(rec, n_boot, ests=None):
    """One row per (task, layer, sae, estimator): the capture table plus the seed parsed from the estimator name."""
    pp = rec["pp"]
    have = sorted(set(pp[pp.condition == "feat_A_along_U"].estimator) - {"random", ""})
    ests = ests or [e for e in have if e.startswith("das") or e == "diffmean"]
    t = mp.capture_table(rec, ests, ("feat_A_along_U",), n_boot)
    if not len(t):
        return t
    t["seed"] = t.estimator.map(lambda e: int(SEED_RE.search(e).group(1)) if SEED_RE.search(e) else 0)
    return mask_low_effect(t)


def mask_low_effect(t: pd.DataFrame) -> pd.DataFrame:
    """capture = NLDR(features along U) / NLDR(full along U) is a ratio: where U itself barely moves the output (full NLDR below EFFECT_MIN,
    or not above the random-subspace p95) it is noise (P22: the difference-of-means direction has no effect below layer 6 and 'captures'
    -1.5 to 7.9 there). Such rows keep their numbers but their capture becomes NaN and `no_effect` is set; every aggregate skips them."""
    t = t.copy()
    t["no_effect"] = (t.full_nldr < EFFECT_MIN) | (t.full_nldr <= t.random_p95_full_nldr + 0.05)
    t.loc[t.no_effect, CAP] = np.nan
    return t


def cell_table(cap: pd.DataFrame, gap: pd.DataFrame, group=("task", "layer", "sae"), extra=()) -> pd.DataFrame:
    """Seed mean / sd / min / max of capture per cell (rank-8 DAS or the given estimators) joined with the decision-level gap."""
    keys = list(group)
    g = cap.groupby(keys + list(extra)).agg(capture=(CAP, "mean"), capture_sd=(CAP, "std"), capture_min=(CAP, "min"),
                                             capture_max=(CAP, "max"), n_seeds=(CAP, "count"), full_nldr=("full_nldr", "mean"),
                                             random_p95=("random_p95_full_nldr", "mean"), no_effect=("no_effect", "mean")).reset_index()
    g["capture_sd"] = g.capture_sd.fillna(0.0)
    if len(gap):
        gcols = [c for c in ("full_A_nldr", "feat_A_errB_nldr", "gap_feat_A_errB_nldr", "gap_feat_A_errB_flip", "feat_B_errA_nldr", "n_pairs")
                 if c in gap]
        g = g.merge(gap[keys + gcols].drop_duplicates(keys), on=keys, how="left")
    return g


def curves(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.groupby(["task", "layer"]).agg(capture=("capture", "mean"), capture_sd_seed=("capture_sd", "max"),
                                                  n_dicts=("sae", "nunique")).reset_index().sort_values(["task", "layer"])


def h3_rows(cv: pd.DataFrame):
    """Per task: argmin layer, its normalised depth, the dip against the outer layers and the verdict inputs."""
    out = []
    for task, g in cv.groupby("task"):
        g = g.dropna(subset=["capture"]).sort_values("layer")
        if len(g) < 5:
            continue
        i = int(g.capture.to_numpy().argmin())
        lmin = int(g.layer.iloc[i])
        mid = (g.layer.map(depth) >= 1 / 3) & (g.layer.map(depth) <= 2 / 3)
        outer = g[~mid].capture
        dip = float(outer.median() - g.capture.iloc[i]) if len(outer) else np.nan
        out.append({"task": task, "n_layers": len(g), "min_layer": lmin, "min_depth": round(depth(lmin), 3), "capture_min": float(g.capture.iloc[i]),
                    "capture_outer_median": float(outer.median()) if len(outer) else np.nan, "dip": dip,
                    "seed_sd_at_min": float(g.capture_sd_seed.iloc[i]), "in_mid_third": bool(mid.iloc[i]),
                    "holds": bool(mid.iloc[i] and dip > g.capture_sd_seed.iloc[i])})
    return pd.DataFrame(out)


def rank_demeaned(df: pd.DataFrame, x: str, y: str, by) -> float:
    """Spearman after removing each group's mean rank (within-group association; groups with < 3 rows are dropped)."""
    d = df[[x, y] + list(by)].dropna()
    d = d.groupby(list(by)).filter(lambda g: len(g) >= 3)
    if len(d) < 6:
        return np.nan
    rx, ry = d[x].rank(), d[y].rank()
    rx = rx - rx.groupby([d[b] for b in by]).transform("mean")
    ry = ry - ry.groupby([d[b] for b in by]).transform("mean")
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def metric_table(cells: pd.DataFrame, pf: pd.DataFrame, label: str) -> pd.DataFrame:
    """Spearman of capture with each fidelity metric, pooled and within (task, layer); one row per (label, task, metric)."""
    if pf is None or not len(pf):
        return pd.DataFrame()
    f = pf.copy()
    f["sae"] = f["dict"]
    m = cells.merge(f[["sae"] + [c for c in METRICS if c in f]], on="sae", how="inner")
    rows = []
    for task, g in list(m.groupby("task")) + [("ALL", m)]:
        for met in [c for c in METRICS if c in g]:
            d = g[["capture", met]].dropna()
            if len(d) < 6:
                continue
            rows.append({"set": label, "task": task, "metric": met, "n": len(d), "spearman_pooled": float(d.corr(method="spearman").iloc[0, 1]),
                         "spearman_within_layer": rank_demeaned(g, "capture", met, ["task", "layer"])})
    return pd.DataFrame(rows)


def dictionary_level(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per dictionary: capture (and gap) averaged over the tasks and seeds the dictionary was scored on."""
    return cells.groupby(["layer", "sae"]).agg(capture=("capture", "mean"), gap=("gap_feat_A_errB_nldr", "mean")).reset_index().assign(task="all tasks")


def layer_metrics(cells: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Per layer: mean capture and gap next to the mean fidelity metrics of the dictionaries scored there (the figure that shows whether
    a metric follows the depth pattern)."""
    if pf is None or not len(pf):
        return pd.DataFrame()
    f = pf.rename(columns={"dict": "sae"})
    d = dictionary_level(cells).merge(f[["sae"] + [c for c in METRICS if c in f]], on="sae", how="inner")
    return d.groupby("layer").mean(numeric_only=True).reset_index()


def grid_effects(cells: pd.DataFrame) -> pd.DataFrame:
    """Width and sparsity effects inside each (task, layer): capture minus the (task, layer) mean, averaged by width and by L0 variant."""
    c = cells.copy()
    m = c.sae.str.extract(r"_(\d+k)_(small|medium|big)$")
    c["width"], c["l0v"] = m[0], m[1]
    c = c[c.width.notna() & c.capture.notna()]
    if not len(c):
        return pd.DataFrame()
    c["dev"] = c.capture - c.groupby(["task", "layer"]).capture.transform("mean")
    rows = []
    for col in ("width", "l0v"):
        for v, g in c.groupby(col):
            rows.append({"factor": col, "level": v, "n": len(g), "capture_dev": float(g.dev.mean()),
                         "capture_dev_by_layer": ", ".join(f"L{L} {x:+.3f}" for L, x in g.groupby("layer").dev.mean().items())})
    return pd.DataFrame(rows)


def h4_verdict(mt: pd.DataFrame):
    if not len(mt):
        return "NOT RUN (no fidelity runs)", []
    a = mt[mt.task == "ALL"]
    lines = []
    best = None
    for col in ("spearman_pooled", "spearman_within_layer"):
        r = a.set_index("metric")[col].dropna()
        if len(r) == 0 or "fvu" not in r:
            continue
        other = r.drop("fvu").abs()
        top = other.idxmax() if len(other) else None
        lines.append(f"- {col}: " + ", ".join(f"{k} {v:+.2f}" for k, v in r.sort_values(key=abs, ascending=False).items()))
        if top is not None and other[top] >= 0.7 and other[top] >= abs(r["fvu"]) + 0.05:
            best = top
        elif abs(r["fvu"]) >= 0.7 and (len(other) == 0 or other.max() < abs(r["fvu"]) + 0.05):
            best = best or "fvu"
    if best == "fvu":
        return "NO-GO (FVU already predicts capture)", lines
    if best:
        return f"GO ({best} predicts capture better than FVU)", lines
    return "UNCLEAR (no metric reaches |rho| 0.7)", lines


def replicates(cells: pd.DataFrame) -> pd.DataFrame:
    """P20's all-layers dictionaries and P28's subset dictionaries are independently trained at layers 7 / 13 / 17 / 22 (same width and L0):
    the capture difference is a dictionary-training noise floor."""
    c = cells.copy()
    c["variant"] = c.sae.str.extract(r"_(16k_(?:small|big))$")[0]
    c["origin"] = np.where(c.sae.str.contains("_all_"), "all", np.where(c.sae.str.contains(r"gs2_1b_L"), "subset", ""))
    c = c[c.variant.notna() & (c.origin != "")]
    a = c[c.origin == "all"].set_index(["task", "layer", "variant"]).capture
    b = c[c.origin == "subset"].set_index(["task", "layer", "variant"]).capture
    idx = a.index.intersection(b.index)
    return pd.DataFrame({"capture_all": a.loc[idx], "capture_subset": b.loc[idx], "abs_diff": (a.loc[idx] - b.loc[idx]).abs()}).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="analysis/batch2_atlas")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    runs = mp.load_runs(Path(a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pf = {}
    for rec in runs.get("PF", []):
        f = rec["dir"] / "fidelity.csv"
        if f.exists():
            cfgid = rec["manifest"]["config"].get("identity", {}).get("config")
            pf[cfgid] = pd.concat([pf[cfgid], pd.read_csv(f)]) if cfgid in pf else pd.read_csv(f)
    report = ["# Batch-2 atlas (exploratory, post hoc; scripts/make_atlas.py)", "",
              "Capture = NLDR(feat_A_along_U) / NLDR(full_along_U), DAS rank 8, mean over independent seeds; differences < ~0.1 are ties (S10).", ""]
    allcap, allcells, allmet = [], [], []
    for exp in ("P20", "P28", "P21", "P22"):
        if exp not in runs:
            report += [f"## {exp}: no complete run", ""]
            continue
        rec = runs[exp]
        try:
            cap = capture_rows(rec, a.boot)
            gap = mp.gap_table(rec, ("feat_A_errB", "feat_A_meanPos", "feat_B_errA"))
        except Exception as ex:  # noqa: BLE001 - one experiment's failure must not hide the others
            report += [f"## {exp}: analysis failed ({type(ex).__name__}: {ex})", ""]
            continue
        if not len(cap):
            report += [f"## {exp}: no capture rows", ""]
            continue
        cap.insert(0, "exp", exp)
        allcap.append(cap)
        if exp == "P22":                                    # refusal: report each estimator separately (diffmean, DAS rank 1 / 4)
            cap["est_kind"] = cap.estimator.str.replace(r"_s\d+$", "", regex=True)
            cells = cell_table(cap, gap, extra=("est_kind",))
        else:
            cells = cell_table(cap[cap.estimator.str.startswith("das8")], gap)
        cells.insert(0, "exp", exp)
        allcells.append(cells)
        kinds = [None] if exp != "P22" else sorted(cells.est_kind.unique())
        h3_all = []
        for kind in kinds:
            sub = cells if kind is None else cells[cells.est_kind == kind]
            h3k = h3_rows(curves(sub))
            if len(h3k):
                h3_all.append(h3k.assign(task=h3k.task if kind is None else h3k.task + "/" + kind))
        cv = curves(cells if exp != "P22" else cells[cells.est_kind == "das1v2_kl"])
        h3 = pd.concat(h3_all) if h3_all else pd.DataFrame()
        report += [f"## {exp}", f"{len(cells)} cells, {cap.seed.nunique()} DAS seeds; capture range {cells.capture.min():.2f}-{cells.capture.max():.2f}; "
                   f"largest seed sd {cells.capture_sd.max():.3f}; mean n_seeds {cells.n_seeds.mean():.1f}", ""]
        if len(h3):
            for r in h3.itertuples():
                report.append(f"- {r.task}: capture minimum {r.capture_min:.3f} at layer {r.min_layer} (depth {r.min_depth:.2f}); outer-layer median "
                              f"{r.capture_outer_median:.3f}; dip {r.dip:+.3f} vs seed sd {r.seed_sd_at_min:.3f}; "
                              f"{'in' if r.in_mid_third else 'outside'} the middle third -> {'dip holds' if r.holds else 'no clear dip'}")
            tasks_ok = int(h3.holds.sum())
            report.append(f"- H3 ({exp}): {tasks_ok} of {len(h3)} tasks have a mid-depth dip -> " + ("GO" if tasks_ok >= 2 else "NO-GO / unclear"))
            h3.insert(0, "exp", exp)
            h3.to_csv(out / f"atlas_h3_{exp}.csv", index=False)
        cv.insert(0, "exp", exp)
        cv.to_csv(out / f"atlas_curves_{exp}.csv", index=False)
        report.append("")
        key = {"P20": "P20", "P28": "P28", "P21": "P21", "P22": "P22"}[exp]
        base_cells = cells if exp != "P22" else cells[cells.est_kind == "das1v2_kl"]
        mt = metric_table(base_cells, pf.get(key), exp)
        if len(mt):
            allmet.append(mt)
        dl = dictionary_level(base_cells)
        mt2 = metric_table(dl, pf.get(key), f"{exp} per dictionary")
        if len(mt2):
            allmet.append(mt2)
        lm = layer_metrics(base_cells, pf.get(key))
        if len(lm):
            lm.insert(0, "exp", exp)
            lm.to_csv(out / f"atlas_layer_metrics_{exp}.csv", index=False)
        ge = grid_effects(base_cells)
        if len(ge):
            ge.insert(0, "exp", exp)
            ge.to_csv(out / f"atlas_grid_effects_{exp}.csv", index=False)
            report += [f"Width / sparsity effects inside each (task, layer) ({exp}): " + "; ".join(
                f"{r.factor} {r.level} {r.capture_dev:+.3f}" for r in ge.itertuples()), ""]
    # ---- pooled H4 over the Gemma 3 base sets (P20 + P28 share the model; P21 is Gemma 2) and per set
    mt = pd.concat(allmet) if allmet else pd.DataFrame()
    if len(mt):
        mt.to_csv(out / "atlas_metrics.csv", index=False)
        report += ["## H4: which cheap metric predicts capture?", ""]
        for label, g in mt.groupby("set"):
            v, lines = h4_verdict(g.assign(task=np.where(g.task == "ALL", "ALL", g.task)))
            report += [f"- **{label}**: {v}"] + [f"  {ln}" for ln in lines]
        report.append("")
    if allcap:
        pd.concat(allcap).to_csv(out / "atlas_capture.csv", index=False)
    if allcells:
        cells_all = pd.concat(allcells)
        cells_all.to_csv(out / "atlas_cells.csv", index=False)
        both = cells_all[cells_all.exp.isin(["P20", "P28"]) & (cells_all.get("est_kind", pd.Series("das8", index=cells_all.index)).fillna("das8") != "diffmean")]
        rep = replicates(both) if len(both) else pd.DataFrame()
        if len(rep):
            rep.to_csv(out / "atlas_replicates.csv", index=False)
            report += ["## Independently trained dictionaries (P20 all-layers vs P28 subset, same layer / width / L0)", "",
                       f"- {len(rep)} pairs: median |capture difference| {rep.abs_diff.median():.3f}, max {rep.abs_diff.max():.3f} "
                       "(the dictionary-training noise floor; compare any dictionary difference with it)", ""]
    (out / "ATLAS.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Calibration for the E9 / M3 thresholds from data that already exists (release 2) and from random geometry. Uses no E9/E14 output.
(a) bootstrap CI half-width of a flip-rate difference at the cell sizes E9/E14 will have (RS L18 f_pilot, E3 pilot-family eval)
(b) the random-U null: flip rate of repair / repair_mean with Haar-random U per rank (E3, RS L18, fit fresh, eval fresh, f_pilot)
(c) principal cosines between independent Haar-random k-dim subspaces of R^2304 (what 'unrelated' looks like)
(d) cosines between the two error estimators already fitted (pca_pooled vs answer_sep), release-1/2 directions/*.json
"""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from harness import final as FN
S = sys.argv[1]
out = []
pp = pd.read_parquet(glob.glob(f"{S}/ana_runs/E3/*/per_pair.parquet")[0])
m = json.load(open(glob.glob(f"{S}/ana_runs/E3/*/manifest.json")[0]))
pp = FN.eval_group(pp)
for task, layer in (("add_chain", 18), ("entity_track", 12)):
    g = pp[(pp.task == task) & (pp.fit_families == "fresh") & (pp.family != "pilot")]
    g = g.assign(family="fresh")
    t = pd.DataFrame(FN.condition_table(g, "f_pilot", 1000, 0))
    fa = t[t.condition == "feat_A_meanPos"].iloc[0]
    hw = (fa.gap_flip_rate_hi - fa.gap_flip_rate_lo) / 2
    out.append(f"(a) {task} L{layer} f_pilot fresh: n={int(fa.n_pairs)} eligible={int(fa.flip_n_eligible)}; CI half-width of a flip-rate gap ~ {hw*100:.1f} pts")
    rnd = t[(t.estimator == "random") & t.condition.isin(["repair"])]
    for r, gr in rnd.groupby("r"):
        same = t[(t.estimator == "pca_pooled") & (t.condition == "repair") & (t.r == r)].flip_rate
        out.append(f"(b) {task} repair r={r}: random flip mean {gr.flip_rate.mean():.3f}, p95 {np.percentile(gr.flip_rate,95):.3f}, max {gr.flip_rate.max():.3f} "
                   f"(n={len(gr)}); errB baseline {t[t.condition=='feat_A_errB'].flip_rate.iloc[0]:.3f}; pca_pooled {same.iloc[0] if len(same) else float('nan'):.3f}")
rng = np.random.default_rng(0)
d = 2304
for k in (1, 2, 4, 8):
    cs = []
    for _ in range(2000):
        U = np.linalg.qr(rng.standard_normal((d, k)))[0]; V = np.linalg.qr(rng.standard_normal((d, k)))[0]
        cs.append(np.linalg.svd(U.T @ V, compute_uv=False)[0])
    out.append(f"(c) Haar random k={k}: top principal cosine p50 {np.median(cs):.3f}, p99 {np.percentile(cs,99):.3f}, max {max(cs):.3f}")
for f in sorted(glob.glob(f"{S}/ana_runs/E*/*/directions/*.json"))[:40]:
    j = json.load(open(f)); c = j.get("_cosines", {})
    for key in ("pca_pooled|answer_sep|r1", "pca_pooled|answer_sep|r2"):
        if key in c:
            out.append(f"(d) {f.split('ana_runs/')[1][:60]} {key}: {[round(x,3) for x in c[key]]}")
print("\n".join(out))

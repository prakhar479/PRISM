"""E17 standard readout: DAS robustness on RS L18 16k (E9 rules), and E9 vs E17 cross-machine reproducibility.
Inputs: r5/tables/E17_illusion.csv, E17_das_stability.csv, hw/tables/final/E9_illusion.csv, per_pair.parquet + directions of
E9 (958ffcdee3d4) and E17 (5a4a5fedea5f). Outputs: team/standard_out/E17_*.csv and a printed markdown readout."""
import json
import re

from standard_common import *

t = pd.read_csv(R5 / "E17_illusion.csv")
mism, nchk = recheck_rules(t)
print(f"rule recheck (effective): {mism} disagreements of {nchk} evaluable rows")


def parse(est):
    m = re.match(r"das(\d+)(?:_e(\d+))?(?:_lr([0-9.e-]+))?(?:_s(\d+))?$", est)
    if not m:
        return dict(k=np.nan, epochs=np.nan, lr="", seed=np.nan)
    return dict(k=int(m[1]), epochs=int(m[2] or 3), lr=m[3] or "1e-2", seed=int(m[4] or 0))


t = add_p_eff(t)
pp = load_pp("E17", ["clean_B", "full_along_U", "feat_A_along_U"])
mc = mcnemar_rows(pp)
mc.to_csv(OUT / "E17_mcnemar.csv", index=False)
x = t[t.condition.isin(["full_along_U", "feat_A_along_U"]) & (t.r > 0)].copy()
# cross-check my flip rates against the shipped table
chk = x.merge(mc, on=["task", "layer", "sae", "fit_families", "family", "filter", "estimator", "r"])
dd = np.where(chk.condition == "full_along_U", chk.flip_full_mine, chk.flip_feat_mine) - chk.flip_rate
print("max |my flip - table flip|:", np.nanmax(np.abs(dd)), "n rows", len(chk))

# multiplicity: every non-random estimator row x {full, feat} x 3 filters (p_eff) + every (estimator, r, filter) McNemar
tests = pd.concat([
    x.assign(test="eff:" + x.condition)[["filter", "estimator", "r", "test", "p_eff"]].rename(columns={"p_eff": "p"}),
    mc.assign(test="illusion_mcnemar")[["filter", "estimator", "r", "test", "p_ill"]].rename(columns={"p_ill": "p"})])
tests["q_bh"] = bh(tests.p)
tests.to_csv(OUT / "E17_tests.csv", index=False)
print(f"E17 tests: {len(tests)} (BH over all); rejections at q<.05: {(tests.q_bh < .05).sum()}")
qmap = tests.set_index(["filter", "estimator", "r", "test"]).q_bh

rows = []
for flt in FILTERS:
    for est in sorted(x.estimator.unique(), key=lambda e: (parse(e)["k"] if e.startswith("das") else 0, e)):
        for r in sorted(x[(x.estimator == est)].r.unique()):
            fu = x[(x["filter"] == flt) & (x.estimator == est) & (x.r == r) & (x.condition == "full_along_U")].iloc[0]
            fe = x[(x["filter"] == flt) & (x.estimator == est) & (x.r == r) & (x.condition == "feat_A_along_U")].iloc[0]
            m = mc[(mc["filter"] == flt) & (mc.estimator == est) & (mc.r == r)].iloc[0]
            pr = parse(est) if est.startswith("das") else dict(k=r, epochs=np.nan, lr="", seed=np.nan)
            acc = fu.das_train_acc
            # what the effective rule would say if the validity gate were ignored (reported, never used as evidence)
            raw_eff = lambda s: bool((s.flip_rate >= s.random_p95 + EFFECT_PTS) and (s.flip_rate_lo > s.random_p95))
            rows.append({"filter": flt, "estimator": est, "r": int(r), "epochs": pr["epochs"], "lr": pr["lr"], "seed": pr["seed"],
                         "n_elig": int(fu.flip_n_eligible), "train_acc": acc,
                         "fit_valid": flag(fu.das_fit_valid) if est.startswith("das") else "",
                         "full": fu.flip_rate, "full_lo": fu.flip_rate_lo, "full_hi": fu.flip_rate_hi, "p95_full": fu.random_p95,
                         "feat": fe.flip_rate, "feat_lo": fe.flip_rate_lo, "feat_hi": fe.flip_rate_hi, "p95_feat": fe.random_p95,
                         "diff": fu.illusion_diff, "diff_lo": fu.illusion_diff_lo, "diff_hi": fu.illusion_diff_hi,
                         "eff_full": flag(fu.full_along_U_effective), "feat_med": flag(fu.feature_mediated),
                         "illusion": flag(fu.illusion),
                         "ungated_eff_full": raw_eff(fu), "ungated_eff_feat": raw_eff(fe),
                         "q_eff_full": qmap[(flt, est, r, "eff:full_along_U")], "q_eff_feat": qmap[(flt, est, r, "eff:feat_A_along_U")],
                         "n10": m.n10, "n01": m.n01, "q_ill": qmap[(flt, est, r, "illusion_mcnemar")]})
R = pd.DataFrame(rows)
R.to_csv(OUT / "E17_readout.csv", index=False)
print(R[R["filter"] == "f_pilot"].round(3).to_string())

# filter agreement on the three rule outcomes
piv = R.pivot_table(index=["estimator", "r"], columns="filter", values=["eff_full", "feat_med", "illusion"], aggfunc="first")
dis = piv[[any(len(set(piv.loc[i, c].values)) > 1 for c in ("eff_full", "feat_med", "illusion")) for i in piv.index]]
print("\nfilter disagreements (estimator, r):\n", dis.to_string() if len(dis) else "none")

# does longer training make rank 1/2 valid?
print("\nrank 1/2 train acc by variant:")
print(R[(R["filter"] == "f_pilot") & (R.r <= 2) & R.estimator.str.startswith("das")][["estimator", "epochs", "lr", "seed", "train_acc", "full", "feat"]].to_string())

# ------------------------------------------------------------------ reproducibility: E9 vs E17 plain das1/2/4/8
e9 = pd.read_csv(FINAL / "E9_illusion.csv")
e9 = e9[(e9.task == "add_chain") & e9.estimator.isin(["das1", "das2", "das4", "das8", "pca_pooled"]) & e9.condition.isin(["full_along_U", "feat_A_along_U"])]
k = ["filter", "condition", "estimator", "r"]
cmp = e9.merge(t, on=k, suffixes=("_E9", "_E17"))
cols = ["flip_rate", "flip_rate_lo", "flip_rate_hi", "nldr", "random_p95", "das_train_acc", "illusion_diff", "illusion_diff_lo"]
cmp_out = cmp[k + [f"{c}_E9" for c in cols] + [f"{c}_E17" for c in cols]]
cmp_out.to_csv(OUT / "E17_vs_E9.csv", index=False)
for c in cols:
    a, b = cmp[f"{c}_E9"], cmp[f"{c}_E17"]
    print(f"E9 vs E17 {c}: max |diff| = {np.nanmax(np.abs(a - b)):.3g} over {a.notna().sum()} rows")
for c in ["effective", "illusion", "feature_mediated", "das_fit_valid"]:
    a, b = cmp[f"{c}_E9"].map(lambda v: "NA" if pd.isna(v) else str(bool(v) if not isinstance(v, str) else v == "True")), cmp[f"{c}_E17"].map(lambda v: "NA" if pd.isna(v) else str(bool(v) if not isinstance(v, str) else v == "True"))
    print(f"  flag {c}: {(a != b).sum()} disagreements / {len(a)}")

d9 = np.load(next((RUNS / "E9").glob("*/directions/add_chain_L18_L18_16k_l074.npz")))
d17 = np.load(next((RUNS / "E17").glob("*/directions/add_chain_L18_L18_16k_l074.npz")))
for kk in ["das1", "das2", "das4", "das8", "pca_pooled"]:
    print(f"  U {kk}: bitwise equal = {np.array_equal(d9[kk], d17[kk])}, max|diff| = {np.abs(d9[kk] - d17[kk]).max():.3g}")
cols = ["pair_id", "condition", "estimator", "r", "seed", "logit_A", "logit_B", "top1_id", "task"]
a = pd.read_parquet(next((RUNS / "E9").glob("*/per_pair.parquet")), columns=cols)
a = a[a.task == "add_chain"]
b = pd.read_parquet(next((RUNS / "E17").glob("*/per_pair.parquet")), columns=cols)
mm = a.merge(b, on=["pair_id", "condition", "estimator", "r", "seed"], suffixes=("_9", "_17"))
print(f"  per-pair rows shared: {len(mm)}; max|logit_A diff| = {(mm.logit_A_9 - mm.logit_A_17).abs().max():.3g}; "
      f"max|logit_B diff| = {(mm.logit_B_9 - mm.logit_B_17).abs().max():.3g}; top1 agreement = {(mm.top1_id_9 == mm.top1_id_17).mean():.4f}")
for e in ("E9", "E17"):
    m = json.loads(next((RUNS / e).glob("*/manifest.json")).read_text())
    print(f"  {e} manifest: host {m['machine']['hostname']}, device {m['machine']['device']}, torch {m['machine']['torch']}, commit {m['git']['commit'][:7]}")

# ------------------------------------------------------------------ seed stability summary (rule: aligned >= 0.5 top cosine)
s = pd.read_csv(R5 / "E17_das_stability.csv")
s["seed_a"] = [parse(v)["seed"] for v in s.das_a]
s["seed_b"] = [parse(v)["seed"] for v in s.das_b]
s["kind"] = np.where(s.seed_a != s.seed_b, "different seed", "same seed (shared init)")
s["aligned_top"] = s.cos_top >= ALIGNED
print("\nDAS rotation stability (top / mean principal cosine), by pair kind:")
print(s.groupby(["r", "kind"]).agg(n=("cos_top", "size"), cos_top_min=("cos_top", "min"), cos_top_med=("cos_top", "median"),
                                    cos_top_max=("cos_top", "max"), cos_mean_med=("cos_mean", "median"),
                                    aligned_share=("aligned_top", "mean"), learned_top_med=("cos_top_learned", "median")).round(3).to_string())
print(s[s.r >= 4][["r", "das_a", "das_b", "acc_a", "acc_b", "cos_top", "cos_mean", "cos_top_learned"]].round(3).to_string())
s.to_csv(OUT / "E17_stability_annot.csv", index=False)

# ------------------------------------------------------------------ alignment of pca_pooled rank-k U with EVERY DAS variant of rank k
# (the shipped cos_top_das column keeps only one DAS variant per (estimator, r) because of drop_duplicates; recomputed here)
from harness.directions import principal_cosines
from scipy.stats import spearmanr
HAAR = {1: .052, 2: .072, 4: .097, 8: .131}
al = []
for v in [k for k in d17.files if k.startswith("das")]:
    kk = parse(v)["k"]
    c = principal_cosines(d17["pca_pooled"][:, :kk], d17[v])
    lab = "aligned" if c[0] >= ALIGNED else ("unrelated" if c[0] <= HAAR[kk] else "partial")
    al.append({"das": v, "k": kk, "cos_top_pca": float(c[0]), "alignment": lab})
AL = pd.DataFrame(al).sort_values(["k", "das"])
AL.to_csv(OUT / "E17_alignment.csv", index=False)
print("\npca_pooled vs DAS alignment per variant:\n", AL.round(3).to_string())
print("shipped cos_top_das (pca rows, f_pilot):", t[(t.estimator == "pca_pooled") & (t["filter"] == "f_pilot") & (t.condition == "full_along_U")][["r", "cos_top_das"]].round(3).values.tolist())
dv = R[(R["filter"] == "f_pilot") & R.estimator.str.startswith("das")]
rho = spearmanr(dv.train_acc, dv.full)
print(f"Spearman(train acc, full_along_U flip) over {len(dv)} DAS variants (f_pilot): rho = {rho.statistic:.3f}, p = {rho.pvalue:.2g}")
rho2 = spearmanr(dv.train_acc, dv.feat)
print(f"Spearman(train acc, feat_A_along_U flip): rho = {rho2.statistic:.3f}, p = {rho2.pvalue:.2g}")

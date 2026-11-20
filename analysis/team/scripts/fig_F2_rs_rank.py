"""F2 Running-sum causal dimensionality (RS L18, 16k, f_pilot, n = 150 eligible pairs).
Flip rate of full_along_U and feat_A_along_U vs subspace rank r for
  * DAS seed 0, 3 epochs, lr 1e-2 (E18: das1..das16), line;
  * E17 DAS variants (seeds 1/2, 10/30 epochs, lr 3e-2) as points; filled = valid fit (train interchange acc >= 0.5),
    hollow = invalid;
  * error-derived U (pca_pooled, answer_sep; E18, r = 1..64) with 95% pair-bootstrap CI bands;
  * random-U null: band 0..p95 over 20 Haar draws per rank, and the preregistered 'effective' bar p95 + 0.10.
Sources: hw/tables/final/E18_illusion.csv, hw/tables/final/E17_illusion.csv. Exploratory/post hoc."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figures_common as F

F.setup()
cols = ["exp_id", "condition", "estimator", "r", "flip_rate", "flip_rate_lo", "flip_rate_hi", "random_mean",
        "random_p95", "random_n", "das_train_acc", "das_fit_valid", "n_pairs", "flip_n_eligible", "run_hash"]
e18 = pd.read_csv(F.FINAL / "E18_illusion.csv")
e18 = e18[(e18.task == "add_chain") & (e18.layer == 18) & (e18["filter"] == "f_pilot") & (e18.r > 0)
          & e18.condition.isin(["full_along_U", "feat_A_along_U"])][cols]
e17 = pd.read_csv(F.FINAL / "E17_illusion.csv")
e17 = e17[(e17.task == "add_chain") & (e17["filter"] == "f_pilot") & (e17.r > 0)
          & e17.condition.isin(["full_along_U", "feat_A_along_U"])][cols]
BASE = ["das1", "das2", "das4", "das8"]
# the E17 base fits are the same fits as E18's (identical settings, same node): check, then keep E18's copy
chk = e17[e17.estimator.isin(BASE)].merge(e18[e18.estimator.isin(BASE)], on=["condition", "estimator"], suffixes=("_17", "_18"))
assert len(chk) == 8 and np.allclose(chk.flip_rate_17, chk.flip_rate_18), chk
var = e17[e17.estimator.str.startswith("das") & ~e17.estimator.isin(BASE)].copy()
var["kind"] = "das_variant"
e18 = e18.copy()
e18["kind"] = np.where(e18.estimator.str.startswith("das"), "das_seed0_3ep", "error_U")
D = pd.concat([e18, var], ignore_index=True)
D["valid"] = D.das_train_acc >= 0.5
assert (D[D.kind != "error_U"].valid == D[D.kind != "error_U"].das_fit_valid.astype(bool)).all()

# random null per (condition, r): identical across estimators of the same rank within a run; take the max p95 over
# all rows of that rank (conservative), from E18 (E17 values agree to <= 0.0004)
rnd = D[D.random_p95.notna()].groupby(["condition", "r"]).agg(p95=("random_p95", "max"), mean=("random_mean", "max")).reset_index()

# deterministic jitter for variant points (multiplicative, in log2 space)
var_order = {}
for r, g in D[D.kind == "das_variant"].groupby("r"):
    names = sorted(g.estimator.unique())
    for i, n in enumerate(names):
        var_order[n] = (i - (len(names) - 1) / 2) * (0.07 if len(names) > 3 else 0.12)
D["x"] = D.r * 2 ** D.estimator.map(var_order).fillna(0.0)

TITLES = {"full_along_U": "(a) Full-residual patch along U  (x_B + P_U(x_A − x_B))",
          "feat_A_along_U": "(b) Feature-only patch along U  (x̂_B + P_U(x̂_A − x̂_B) + ε_B)"}
ER = {"pca_pooled": dict(c=F.ORANGE, m="s", ls="--", lab="Error-U: pca_pooled"),
      "answer_sep": dict(c=F.AQUA, m="^", ls="-.", lab="Error-U: answer_sep")}
fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE, 2.85), sharey=True)
for ax, cond in zip(axes, ["full_along_U", "feat_A_along_U"]):
    rn = rnd[rnd.condition == cond].sort_values("r")
    ax.fill_between(rn.r, 0, rn.p95, color=F.RAND, alpha=0.6, lw=0, step=None, zorder=1)
    ax.plot(rn.r, rn.p95 + 0.10, color=F.RAND_LINE, lw=0.8, ls=":", zorder=1)
    for est, s in ER.items():
        x = D[(D.condition == cond) & (D.estimator == est)].sort_values("r")
        ax.fill_between(x.r, x.flip_rate_lo, x.flip_rate_hi, color=s["c"], alpha=0.14, lw=0, zorder=2)
        ax.plot(x.r, x.flip_rate, color=s["c"], ls=s["ls"], marker=s["m"], ms=3.6, mfc="white", mec=s["c"], mew=1.0,
                lw=1.2, zorder=3)
    das = D[(D.condition == cond) & (D.kind == "das_seed0_3ep")].copy()
    das["k"] = das.estimator.str.extract(r"das(\d+)").astype(int)
    das = das.sort_values("r")
    ax.plot(das.r, das.flip_rate, color=F.BLUE, lw=1.5, zorder=4)
    ax.vlines(das.r, das.flip_rate_lo, das.flip_rate_hi, color=F.BLUE, lw=0.9, zorder=4)
    for _, r in das.iterrows():
        ax.plot(r.r, r.flip_rate, "o", ms=5, mec=F.BLUE, mew=1.1, mfc=F.BLUE if r.valid else "white", zorder=5)
    v = D[(D.condition == cond) & (D.kind == "das_variant")]
    for _, r in v.iterrows():
        ax.plot(r.x, r.flip_rate, "D", ms=3.6, mec=F.VIOLET, mew=0.9, mfc=F.VIOLET if r.valid else "white",
                zorder=4, alpha=0.95)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Subspace rank r")
    ax.set_title(TITLES[cond], fontsize=7.3)
axes[0].set_ylabel("Flip rate (test, n = 150)")
axes[1].text(64, 0.10 + 0.02, "random p95 + 0.10\n(effective bar)", ha="right", va="bottom", fontsize=6, color=F.INK2)
handles = [
    Line2D([], [], color=F.BLUE, marker="o", mfc=F.BLUE, mec=F.BLUE, lw=1.5, label="DAS seed 0, 3 ep (valid fit)"),
    Line2D([], [], color=F.BLUE, marker="o", mfc="white", mec=F.BLUE, lw=0, label="DAS seed 0, 3 ep (invalid fit)"),
    Line2D([], [], color=F.VIOLET, marker="D", mfc=F.VIOLET, mec=F.VIOLET, lw=0, ms=3.6, label="DAS variant, valid"),
    Line2D([], [], color=F.VIOLET, marker="D", mfc="white", mec=F.VIOLET, lw=0, ms=3.6,
           label="DAS variant, invalid (seed/epochs/lr)"),
    Line2D([], [], color=F.ORANGE, marker="s", mfc="white", ls="--", label=ER["pca_pooled"]["lab"]),
    Line2D([], [], color=F.AQUA, marker="^", mfc="white", ls="-.", label=ER["answer_sep"]["lab"]),
    Patch(color=F.RAND, alpha=0.6, label="Random U, 0 to p95 (20 draws)"),
]
fig.tight_layout(w_pad=1.0, rect=(0, 0.10, 1, 1))
fig.legend(handles=handles, loc="lower center", fontsize=6.2, ncol=4, handlelength=2.0, bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.2)

out = D.drop(columns=["x"]).merge(rnd.rename(columns={"p95": "null_p95_used", "mean": "null_mean_used"}),
                                  on=["condition", "r"], how="left")
out["effective_bar"] = out.null_p95_used + 0.10
out["x_plotted"] = D.x
out = out.sort_values(["condition", "kind", "estimator", "r"])
F.save(fig, "F2_rs_rank", out)

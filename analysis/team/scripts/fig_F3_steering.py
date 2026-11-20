"""F3 Steering dose-response (E19): flip rate vs steering scale s for x_B + s|m| P_U m/|P_U m|.
Panels: RS L18 all eligible pairs (with the single-vector ceiling max_k P(A-B = k) = 0.353), RS pairs with A > B,
RS pairs with A < B (one-sided flips), OT L12 all eligible pairs. Filter f_pilot (eligible = clean-B top-1 is B's answer).
Random envelope: 20 Haar U per rank; mean and p95 (np.percentile, linear) over draws, computed within each pair subset.
Sources: ana_runs/E19/<hash>/per_pair.parquet (per-pair top-1), hw/data/prompt_sets/add_chain_pilot.jsonl.gz (state_clean /
state_corrupt -> A - B only); totals cross-checked against hw/tables/final/E19_steering.csv. Exploratory/post hoc."""
import glob
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figures_common as F

F.setup()
run = [Path(m).parent for m in glob.glob(str(F.SP / "ana_runs/E19/*/manifest.json"))
       if json.load(open(m)).get("status") == "complete"]
assert len(run) == 1
cols = ["pair_id", "task", "layer", "condition", "estimator", "source", "r", "seed", "ans_A_id", "ans_B_id", "f_pilot",
        "top1_id"]
pp = pd.read_parquet(run[0] / "per_pair.parquet", columns=cols,
                     filters=[("task", "in", ["add_chain", "entity_track"]), ("f_pilot", "=", True)])
cb = pp[pp.condition == "clean_B"].drop_duplicates(["task", "layer", "pair_id"])
cb = cb.assign(elig=cb.top1_id == cb.ans_B_id)[["task", "layer", "pair_id", "elig"]]
st = pp[pp.condition.str.startswith("steer_x")].merge(cb, on=["task", "layer", "pair_id"])
st = st[st.elig].copy()
st["s"] = st.condition.str.replace("steer_x", "").astype(float)
st["isA"] = (st.top1_id == st.ans_A_id).astype(float)

P = pd.DataFrame([json.loads(l) for l in gzip.open(F.SP / "hw/data/prompt_sets/add_chain_pilot.jsonl.gz", "rt")])
P = P.set_index("pair_id")
delta = (P.state_clean - P.state_corrupt)
st["delta"] = np.where(st.task == "add_chain", st.pair_id.map(delta), np.nan)

# single-vector ceiling over eligible RS pairs: max_k P(delta = k)
rs_pairs = cb[(cb.task == "add_chain") & cb.elig].pair_id
dd = rs_pairs.map(delta)
ceil_all = dd.value_counts(normalize=True).max()
ceil_pos = dd[dd > 0].value_counts(normalize=True).max()
ceil_neg = dd[dd < 0].value_counts(normalize=True).max()
assert abs(ceil_all - 0.35333) < 1e-4, ceil_all

SUBSETS = [("add_chain", "all", "(a) RS L18, all eligible pairs", ceil_all),
           ("add_chain", "pos", "(b) RS L18, pairs with A > B", ceil_pos),
           ("add_chain", "neg", "(c) RS L18, pairs with A < B", ceil_neg),
           ("entity_track", "all", "(d) OT L12, all eligible pairs", None)]
EST = [("pca_pooled", 4, dict(c=F.ORANGE, m="s", ls="--", lab="pca_pooled, r = 4")),
       ("answer_sep", 4, dict(c=F.AQUA, m="^", ls="-.", lab="answer_sep, r = 4")),
       ("das4", 4, dict(c=F.BLUE, m="o", ls="-", lab="DAS-4")),
       ("das8", 8, dict(c=F.VIOLET, m="D", ls=(0, (1, 1)), lab="DAS-8"))]


def subset(g, which):
    return g if which == "all" else g[g.delta > 0] if which == "pos" else g[g.delta < 0]


rows = []
for task, which, _, ceil in SUBSETS:
    g = subset(st[st.task == task], which)
    for est, r, _ in EST:
        x = g[(g.estimator == est) & (g.r == r)]
        for s, h in x.groupby("s"):
            rows.append(dict(task=task, subset=which, series=est, r=r, s=s, flip=h.isA.mean(), n=len(h),
                             n_flipped=int(h.isA.sum())))
    for r in [4, 8]:
        x = g[(g.estimator == "random") & (g.r == r)]
        per = x.groupby(["s", "seed"]).isA.mean().reset_index()
        for s, h in per.groupby("s"):
            rows.append(dict(task=task, subset=which, series=f"random_mean", r=r, s=s, flip=h.isA.mean(),
                             n=int(x[(x.s == s)].groupby("seed").size().iloc[0]), n_draws=h.seed.nunique()))
            rows.append(dict(task=task, subset=which, series=f"random_p95", r=r, s=s, flip=np.percentile(h.isA, 95),
                             n=int(x[(x.s == s)].groupby("seed").size().iloc[0]), n_draws=h.seed.nunique()))
    if ceil is not None:
        rows.append(dict(task=task, subset=which, series="ceiling_max_k_P(delta=k)", r=np.nan, s=np.nan, flip=ceil,
                         n=int(len(dd) if which == "all" else (dd > 0).sum() if which == "pos" else (dd < 0).sum())))
R = pd.DataFrame(rows)

# cross-check the 'all' subsets against the frozen table
T = pd.read_csv(F.FINAL / "E19_steering.csv")
T = T[(T["filter"] == "f_pilot") & T.task.isin(["add_chain", "entity_track"])].copy()
T["s"] = T.condition.str.replace("steer_x", "").astype(float)
for (task, est, r), g in T.groupby(["task", "estimator", "r"]):
    mine = R[(R.task == task) & (R.subset == "all") & (R.series == est) & (R.r == r)]
    if len(mine):
        m = mine.merge(g, on="s")
        assert np.allclose(m.flip, m.flip_rate), (task, est, r)
for r in [4, 8]:
    for task in ["add_chain", "entity_track"]:
        g = T[(T.task == task) & (T.r == r)].drop_duplicates("s")
        mm = R[(R.task == task) & (R.subset == "all") & (R.series == "random_mean") & (R.r == r)].merge(g, on="s")
        mp = R[(R.task == task) & (R.subset == "all") & (R.series == "random_p95") & (R.r == r)].merge(g, on="s")
        assert np.allclose(mm.flip, mm.random_mean) and np.allclose(mp.flip, mp.random_p95), (task, r)
print("cross-check vs E19_steering.csv: OK")

fig, axes = plt.subplots(1, 4, figsize=(F.DOUBLE, 2.45), sharey=True)
for ax, (task, which, title, ceil) in zip(axes, SUBSETS):
    q = R[(R.task == task) & (R.subset == which)]
    rm = q[(q.series == "random_mean") & (q.r == 4)].sort_values("s")
    rp = q[(q.series == "random_p95") & (q.r == 4)].sort_values("s")
    rp8 = q[(q.series == "random_p95") & (q.r == 8)].sort_values("s")
    ax.fill_between(rm.s, rm.flip, rp.flip, color=F.RAND, alpha=0.55, lw=0, zorder=1)
    ax.plot(rm.s, rm.flip, color=F.RAND_LINE, lw=0.8, zorder=1)
    ax.plot(rp8.s, rp8.flip, color=F.RAND_LINE, lw=0.8, ls=":", zorder=1)
    for est, r, sty in EST:
        x = q[(q.series == est) & (q.r == r)].sort_values("s")
        ax.plot(x.s, x.flip, color=sty["c"], ls=sty["ls"], marker=sty["m"], ms=3.6, lw=1.3,
                mfc="white" if est in ("pca_pooled", "answer_sep") else sty["c"], mec=sty["c"], mew=1.0, zorder=3)
    if ceil is not None:
        ax.axhline(ceil, color=F.INK, lw=0.8, ls="--", zorder=2)
        ax.text(0.27, ceil + 0.015, f"one-vector ceiling {ceil:.3f}", fontsize=6, color=F.INK, va="bottom")
    ax.axhline(0.5, color=F.MUTED, lw=0.6, zorder=0)
    n = int(q[q.series == "das4"].n.iloc[0])
    ax.set_title(title + f"\n(n = {n})", fontsize=7.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 1, 4, 16])
    ax.set_xticklabels(["0.25", "1", "4", "16"])
    ax.set_xlabel("Steering scale s")
    ax.set_ylim(-0.02, 1.02)
axes[0].set_ylabel("Flip rate")
axes[0].text(16, 0.51, "Q2 bar 0.5", ha="right", fontsize=6, color=F.INK2, va="bottom")
handles = [Line2D([], [], color=s["c"], ls=s["ls"], marker=s["m"], ms=3.6,
                  mfc="white" if e in ("pca_pooled", "answer_sep") else s["c"], mec=s["c"], label=s["lab"])
           for e, _, s in EST]
handles += [Patch(color=F.RAND, alpha=0.55, label="Random U (r = 4): mean to p95"),
            Line2D([], [], color=F.RAND_LINE, ls=":", lw=0.8, label="Random U (r = 8): p95")]
fig.tight_layout(w_pad=0.5, rect=(0, 0.075, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.2, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0,
           handlelength=2.2)
F.save(fig, "F3_steering", R.sort_values(["task", "subset", "series", "r", "s"]))

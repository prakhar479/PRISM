"""F5 People-cities (E18, PC L12 and L16, 16k, English family): (a) DAS train interchange accuracy vs rank (validity bar
0.5); (b) flip rate vs rank for DAS full_along_U / feat_A_along_U and the best error-U (max over pca_pooled, answer_sep)
up to r = 64, filter f_pilot (n = 150 eligible); (c) per-city mean per-pair NLDR of full_along_U by A's city (DAS-4 at
L12 and L16, DAS-1 at L12), pairs with clean A and clean B both correct (as patterns_pc_cities.py), 95% percentile bootstrap
over pairs within city (2000 resamples, seed 0).
Sources: hw/tables/final/E18_illusion.csv (a, b); ana_runs/E18/<hash>/per_pair.parquet + hw/data/prompt_sets/
people_cities_en.jsonl.gz (A's city) for (c), cross-checked against team/patterns_out/pc_city_means.csv. Exploratory/post hoc."""
import glob
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figures_common as F

F.setup()
rows = []
T = pd.read_csv(F.FINAL / "E18_illusion.csv")
T = T[(T.task == "people_cities") & (T["filter"] == "f_pilot") & (T.r > 0)
      & T.condition.isin(["full_along_U", "feat_A_along_U"])]  # repair rows excluded
LC = {12: F.BLUE, 16: F.ORANGE}

fig = plt.figure(figsize=(F.DOUBLE, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.0, 1.55])
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

# (a) validity
for L, c in LC.items():
    x = T[(T.layer == L) & T.estimator.str.match(r"^das\d+$") & (T.condition == "full_along_U")].sort_values("r")
    axA.plot(x.r, x.das_train_acc, color=c, lw=1.3, zorder=2)
    for _, r in x.iterrows():
        axA.plot(r.r, r.das_train_acc, "o" if L == 12 else "s", ms=4.3, mec=c, mew=1.0,
                 mfc=c if r.das_train_acc >= 0.5 else "white", zorder=3)
        rows.append(dict(panel="a", layer=L, estimator=r.estimator, r=r.r, quantity="das_train_acc",
                         value=r.das_train_acc, valid=bool(r.das_train_acc >= 0.5)))
axA.axhline(0.5, color=F.INK, lw=0.8, ls="--")
axA.text(1, 0.52, "validity bar 0.5", fontsize=6, color=F.INK, va="bottom")
axA.set_xscale("log", base=2); axA.set_xticks([1, 2, 4, 8, 16]); axA.set_xticklabels(["1", "2", "4", "8", "16"])
axA.set_ylim(-0.02, 1.0); axA.set_xlabel("DAS rank k"); axA.set_ylabel("DAS train interchange accuracy")
axA.set_title("(a) DAS fit validity", fontsize=7.3)

# (b) flips vs rank
for L, c in LC.items():
    for cond, ls in [("full_along_U", "-"), ("feat_A_along_U", "--")]:
        x = T[(T.layer == L) & T.estimator.str.match(r"^das\d+$") & (T.condition == cond)].sort_values("r")
        axB.plot(x.r, x.flip_rate, color=c, ls=ls, lw=1.3, zorder=2)
        axB.vlines(x.r, x.flip_rate_lo, x.flip_rate_hi, color=c, lw=0.8, zorder=2)
        for _, r in x.iterrows():
            axB.plot(r.r, r.flip_rate, "o" if L == 12 else "s", ms=4.0, mec=c, mew=1.0,
                     mfc=(c if cond == "full_along_U" else "white"), zorder=3)
            rows.append(dict(panel="b", layer=L, estimator=r.estimator, condition=cond, r=r.r, quantity="flip_rate",
                             value=r.flip_rate, lo=r.flip_rate_lo, hi=r.flip_rate_hi, valid=bool(r.das_fit_valid),
                             random_p95=r.random_p95, n=r.flip_n_eligible))
    e = T[(T.layer == L) & T.estimator.isin(["pca_pooled", "answer_sep"])]
    e = e.groupby("r").flip_rate.max().reset_index()
    axB.plot(e.r, e.flip_rate, color=c, ls=":", lw=1.1, marker="x", ms=3.5, zorder=2)
    rows += [dict(panel="b", layer=L, estimator="max(pca_pooled, answer_sep)", condition="full_and_feat_along_U",
                  r=r.r, quantity="flip_rate", value=r.flip_rate) for _, r in e.iterrows()]
rp = T.random_p95.max()
axB.set_xscale("log", base=2); axB.set_xticks([1, 2, 4, 8, 16, 32, 64])
axB.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
axB.set_ylim(-0.02, 1.0); axB.set_xlabel("Subspace rank r"); axB.set_ylabel("Flip rate (n = 150)")
axB.set_title("(b) Patching along U", fontsize=7.3)
axB.legend(handles=[Line2D([], [], color=F.BLUE, marker="o", label="L12"), Line2D([], [], color=F.ORANGE, marker="s", label="L16"),
                    Line2D([], [], color=F.INK2, ls="-", marker="o", mfc=F.INK2, label="DAS full_along_U"),
                    Line2D([], [], color=F.INK2, ls="--", marker="o", mfc="white", label="DAS feat_A_along_U"),
                    Line2D([], [], color=F.INK2, ls=":", marker="x", label="error-U, both conditions (max)")],
           loc="upper left", fontsize=6.0, handlelength=2.3)
axB.text(64, 0.5, f"random-U p95\n≤ {rp:.3f}, all r", ha="right", fontsize=6, color=F.INK2)

# (c) per-city NLDR
run = [Path(m).parent for m in glob.glob(str(F.SP / "ana_runs/E18/*/manifest.json"))
       if json.load(open(m)).get("status") == "complete"]
assert len(run) == 1
pp = pd.read_parquet(run[0] / "per_pair.parquet",
                     columns=["pair_id", "task", "layer", "sae", "condition", "estimator", "r", "ans_A_id", "ans_B_id",
                              "logit_A", "logit_B", "top1_id"],
                     filters=[("task", "=", "people_cities")])
pp["D"] = pp.logit_A - pp.logit_B
a = pp[pp.condition == "clean_A"].drop_duplicates(["layer", "pair_id"])
b = pp[pp.condition == "clean_B"].drop_duplicates(["layer", "pair_id"])
ref = a[["layer", "pair_id", "D"]].rename(columns={"D": "D_A"}).assign(A_ok=(a.top1_id == a.ans_A_id).values).merge(
    b[["layer", "pair_id", "D"]].rename(columns={"D": "D_B"}).assign(B_ok=(b.top1_id == b.ans_B_id).values),
    on=["layer", "pair_id"])
P = {json.loads(l)["pair_id"]: json.loads(l) for l in gzip.open(F.SP / "hw/data/prompt_sets/people_cities_en.jsonl.gz", "rt")}
SER = [(12, "das4", 4, F.BLUE, "o", "DAS-4, L12"), (16, "das4", 4, F.ORANGE, "s", "DAS-4, L16"),
       (12, "das1", 1, F.VIOLET, "D", "DAS-1, L12")]
rng = np.random.default_rng(0)
city_stats = []
for L, est, r, c, m, lab in SER:
    x = pp[(pp.layer == L) & (pp.condition == "full_along_U") & (pp.estimator == est) & (pp.r == r)]
    x = x.merge(ref, on=["layer", "pair_id"])
    x = x[x.A_ok & x.B_ok].copy()
    x["nldr"] = (x.D - x.D_B) / (x.D_A - x.D_B)
    x["city_idx"] = x.pair_id.map(lambda p: P[p]["state_clean"])
    x["city"] = x.pair_id.map(lambda p: P[p]["clean_target"].strip())
    for (ci, city), g in x.groupby(["city_idx", "city"]):
        v = g.nldr.to_numpy()
        bs = rng.choice(v, size=(2000, len(v)), replace=True).mean(1)
        city_stats.append(dict(layer=L, estimator=est, r=r, city_idx=ci, city=city, mean=v.mean(), n=len(v),
                               lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), label=lab))
C = pd.DataFrame(city_stats)
chk = pd.read_csv(F.TEAM / "patterns_out" / "pc_city_means.csv")
for L, est, r, *_ in SER:
    k = chk[(chk.cell == f"E18:L{L}:pc_en") & (chk.key == f"full_along_U|{est}|r{r}")]
    mm = C[(C.layer == L) & (C.estimator == est)].merge(k, left_on="city_idx", right_on="cityA_idx")
    assert len(mm) == 16 and np.allclose(mm["mean"], mm.m) and (mm.n_x == mm.n_y).all(), (L, est)
print("per-city means match patterns_out/pc_city_means.csv")
order = C[(C.layer == 12) & (C.estimator == "das4")].sort_values("mean", ascending=False).city.tolist()
ypos = {cty: len(order) - 1 - i for i, cty in enumerate(order)}
off = {0: 0.25, 1: 0.0, 2: -0.25}
for j, (L, est, r, c, m, lab) in enumerate(SER):
    q = C[(C.layer == L) & (C.estimator == est)]
    y = q.city.map(ypos) + off[j]
    axC.hlines(y, q.lo, q.hi, color=c, lw=0.9)
    axC.plot(q["mean"], y, m, ms=3.8, color=c, mec=c, mfc=c if est == "das4" else "white", mew=1.0, ls="none", label=lab)
    rows += [dict(panel="c", layer=r_.layer, estimator=r_.estimator, r=r_.r, quantity="per_city_nldr_full_along_U",
                  city=r_.city, value=r_["mean"], lo=r_.lo, hi=r_.hi, n=r_.n) for _, r_ in q.iterrows()]
axC.set_yticks(range(len(order))); axC.set_yticklabels(order[::-1], fontsize=6.3)
axC.axvline(0, color=F.INK2, lw=0.7)
axC.grid(axis="y", visible=False)
axC.set_xlabel("Mean per-pair NLDR, full_along_U (95% CI)")
axC.set_title("(c) Restoration by A's city", fontsize=7.3)
axC.legend(loc="lower right", fontsize=6.0, handlelength=1.2)
fig.tight_layout(w_pad=0.8)
F.save(fig, "F5_people_cities", pd.DataFrame(rows))

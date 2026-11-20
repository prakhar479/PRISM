"""F4 Width (E9 16k vs E20 65k / 262k at matched nominal L0): (a) feature capture of the FIXED DAS subspace,
capture = NLDR(feat_A_along_U) / NLDR(full_along_U) (DAS rotations bit-identical across widths, so full_along_U is
width-independent); (b) the feat_A_errB NLDR gap; (c) paired differences wide - 16k (same pairs, same 2000 bootstrap
resamples). RS L18 and OT L12, filter f_pilot, n = 150 test pairs (OT flip-eligible 89).
Source: team/geometry_out/width_capture.csv (geometry_width_capture.py; E9 + E20 per_pair). Exploratory/post hoc."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figures_common as F

F.setup()
d = pd.read_csv(F.TEAM / "geometry_out" / "width_capture.csv")
d = d[(d["filter"] == "f_pilot")].copy()
d["width_k"] = d.sae.str.extract(r"_(\d+)k_").astype(int)
TASK = {"add_chain": ("RS L18", F.BLUE), "entity_track": ("OT L12", F.ORANGE)}
DAS = {"das4_r4": dict(m="o", ls="-", lab="DAS-4"), "das8_r8": dict(m="s", ls="--", lab="DAS-8")}

cap = d[(d.metric == "nldr") & (d.kind == "capture_feat/full") & d.est.isin(DAS)]
gap = d[(d.kind == "gap_feat_A_errB")]
fig, axes = plt.subplots(1, 3, figsize=(F.DOUBLE, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
rows = []
ax = axes[0]
for task, (tl, c) in TASK.items():
    for est, s in DAS.items():
        x = cap[(cap.task == task) & (cap.est == est)].sort_values("width_k")
        ax.errorbar(x.width_k, x.value, yerr=[x.value - x.lo, x.hi - x.value], color=c, ls=s["ls"], marker=s["m"],
                    ms=4, mfc=c if est == "das4_r4" else "white", mec=c, mew=1.0, lw=1.3, elinewidth=0.9)
        rows += [dict(panel="a", task=task, quantity="capture_nldr", est=est, width_k=r.width_k, sae=r.sae, value=r.value,
                      lo=r.lo, hi=r.hi, n=r.n) for _, r in x.iterrows()]
ax.set_xscale("log", base=2)
ax.set_xticks([16, 65, 262]); ax.set_xticklabels(["16k", "65k", "262k"]); ax.set_xlim(12, 340)
ax.set_ylim(0.45, 0.9)
ax.set_xlabel("SAE width")
ax.set_ylabel("Capture = NLDR(feat) / NLDR(full)")
ax.set_title("(a) Feature capture of fixed DAS U", fontsize=7.3)
ax.legend(handles=[Line2D([], [], color=c, lw=1.3, label=tl) for tl, c in TASK.values()]
          + [Line2D([], [], color=F.INK2, ls=s["ls"], marker=s["m"], mfc=F.INK2 if e == "das4_r4" else "white",
                    mec=F.INK2, label=s["lab"]) for e, s in DAS.items()], loc="center", ncol=2, fontsize=6.1)

ax = axes[1]
for task, (tl, c) in TASK.items():
    x = gap[(gap.task == task) & (gap.metric == "nldr")].sort_values("width_k")
    ax.errorbar(x.width_k, x.value, yerr=[x.value - x.lo, x.hi - x.value], color=c, marker="^", ms=4.5, mfc=c, mec=c,
                lw=1.3, elinewidth=0.9, label=tl)
    rows += [dict(panel="b", task=task, quantity="gap_feat_A_errB_nldr", est="", width_k=r.width_k, sae=r.sae,
                  value=r.value, lo=r.lo, hi=r.hi, n=r.n) for _, r in x.iterrows()]
ax.set_xscale("log", base=2)
ax.set_xticks([16, 65, 262]); ax.set_xticklabels(["16k", "65k", "262k"]); ax.set_xlim(12, 340)
ax.set_ylim(0, 0.3)
ax.set_xlabel("SAE width")
ax.set_ylabel("NLDR gap, full_A − feat_A_errB")
ax.set_title("(b) Task-state gap with B's error", fontsize=7.3)
ax.legend(loc="lower left", fontsize=6.1)

# (c) paired differences wide - 16k
ax = axes[2]
items = []
for task in TASK:
    for est in DAS:
        for w in sorted(cap[(cap.task == task) & (cap.width_k > 16)].width_k.unique()):
            r = cap[(cap.task == task) & (cap.est == est) & (cap.width_k == w)].iloc[0]
            items.append((task, f"{TASK[task][0]} capture, {DAS[est]['lab']}, {w}k", r, DAS[est]["m"], "capture_nldr", est))
for task in TASK:
    for w in sorted(gap[(gap.task == task) & (gap.width_k > 16)].width_k.unique()):
        r = gap[(gap.task == task) & (gap.metric == "nldr") & (gap.width_k == w)].iloc[0]
        items.append((task, f"{TASK[task][0]} gap (NLDR), {w}k", r, "^", "gap_feat_A_errB_nldr", ""))
r = gap[(gap.task == "add_chain") & (gap.metric == "flip_rate") & (gap.width_k == 65)].iloc[0]
items.append(("add_chain", "RS L18 gap (flip), 65k", r, "v", "gap_feat_A_errB_flip", ""))
ylab = []
for i, (task, lab, r, m, q, est) in enumerate(items):
    y = len(items) - 1 - i
    c = TASK[task][1]
    ax.plot([r.d_lo, r.d_hi], [y, y], color=c, lw=1.2)
    ax.plot(r.d_vs16k, y, m, color=c, ms=4.5, mfc=c if (est != "das8_r8") else "white", mec=c, mew=1.0)
    ylab.append((y, lab))
    rows.append(dict(panel="c", task=task, quantity=q + "_diff_vs_16k", est=est, width_k=r.width_k, sae=r.sae,
                     value=r.d_vs16k, lo=r.d_lo, hi=r.d_hi, n=r.n))
ax.axvline(0, color=F.INK2, lw=0.7)
ax.set_yticks([y for y, _ in ylab]); ax.set_yticklabels([l for _, l in ylab], fontsize=6.2)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Wide − 16k (paired 95% CI)")
ax.set_title("(c) Paired change with width", fontsize=7.3)
ncap = sum(1 for it in items if it[4] == "capture_nldr")
ax.axhline(len(items) - 1 - (ncap - 0.5), color=F.AXIS, lw=0.6)
fig.tight_layout(w_pad=0.8)
F.save(fig, "F4_width", pd.DataFrame(rows))

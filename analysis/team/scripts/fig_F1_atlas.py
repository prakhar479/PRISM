"""F1 Atlas (E8): gap of feat_A_errB (x_hat_A + eps_B) vs full_A in NLDR and in flip rate, by task x layer.
Source: hw/tables/E8_gaps.csv (exp_id E8, filter f_A_correct, 16k SAEs; 95% paired pair-bootstrap CIs from the table).
Shows the decision-vs-logit dissociation (IOI L20, OT L12). Exploratory/post hoc figure of E8 (36 cells)."""
import matplotlib.pyplot as plt
import pandas as pd

import figures_common as F

F.setup()
d = pd.read_csv(F.TABLES / "E8_gaps.csv")
d = d[(d.exp_id == "E8") & (d["filter"] == "f_A_correct") & (d.condition == "feat_A_errB")
      & d.metric.isin(["nldr", "flip_rate"])]
assert d.groupby(["task", "layer", "metric"]).size().max() == 1
assert d.sae.str.contains("_16k_").all()
TASKS = [("add_chain", "Running sum (RS)"), ("entity_track", "Object tracking (OT)"),
         ("ioi_control", "Names (IOI)"), ("var_binding", "Variable binding (VB)")]
STY = {"nldr": dict(color=F.BLUE, marker="o", ls="-", label="NLDR gap (logit level)", mfc=F.BLUE),
       "flip_rate": dict(color=F.ORANGE, marker="s", ls="--", label="Flip gap (decision level)", mfc="white")}

fig, axes = plt.subplots(1, 4, figsize=(F.DOUBLE, 2.0), sharey=True)
rows = []
for ax, (task, title) in zip(axes, TASKS):
    for m in ["nldr", "flip_rate"]:
        x = d[(d.task == task) & (d.metric == m)].sort_values("layer")
        s = STY[m]
        ax.fill_between(x.layer, x.gap_lo, x.gap_hi, color=s["color"], alpha=0.13, lw=0)
        ax.plot(x.layer, x.gap, color=s["color"], ls=s["ls"], marker=s["marker"], mfc=s["mfc"], mec=s["color"],
                mew=1.0, ms=3.8, lw=1.3, label=s["label"])
        for _, r in x.iterrows():
            rows.append(dict(task=task, layer=int(r.layer), sae=r.sae, role=r.role, filter="f_A_correct",
                             condition="feat_A_errB", metric=m, gap=r.gap, gap_lo=r.gap_lo, gap_hi=r.gap_hi,
                             value_condition=r.value, full_A=r.full_A, n_pairs=int(r.n_pairs),
                             flip_n_eligible=int(r.flip_n_eligible), run_hash=r.run_hash))
    ax.axhline(0, color=F.AXIS, lw=0.8, zorder=1)
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.set_xlabel("Layer")
    ax.set_xticks([4, 8, 12, 16, 20])
    ax.set_xlim(3, 21)
    conf = d[(d.task == task) & (d.role == "confirmatory")]
    for L in sorted(conf.layer.unique()):
        ax.axvline(L, color=F.MUTED, lw=0.6, ls=":", zorder=0)

# direct annotations on the two dissociation cells
def note(ax, task, L, txt, dx, dy):
    nl = d[(d.task == task) & (d.layer == L) & (d.metric == "nldr")].gap.iloc[0]
    fl = d[(d.task == task) & (d.layer == L) & (d.metric == "flip_rate")].gap.iloc[0]
    ax.annotate(txt.format(nl=nl, fl=fl), xy=(L, nl), xytext=(L + dx, nl + dy), fontsize=6.3, color=F.INK,
                ha="right", va="center", arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.6))

note(axes[1], "entity_track", 12, "L12: NLDR {nl:.2f}\nflip {fl:.2f}", -1.0, 0.06)
note(axes[2], "ioi_control", 20, "L20: NLDR {nl:.2f}\nflip {fl:.3f}", -2.5, 0.0)
axes[0].set_ylabel("Gap = M(full_A) − M(feat_A_errB)")
axes[0].set_ylim(-0.15, 0.42)
axes[0].legend(loc="upper left", ncol=1)
fig.text(0.995, -0.02, "dotted line: preregistered (confirmatory) layer; bands: 95% pair-bootstrap CI; filter f_A_correct, 16k SAEs",
         ha="right", va="top", fontsize=6, color=F.INK2)
fig.tight_layout(w_pad=0.6)
F.save(fig, "F1_atlas", pd.DataFrame(rows).sort_values(["task", "metric", "layer"]))

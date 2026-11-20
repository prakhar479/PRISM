"""F7 (lead claim) Feature capture of the causal (DAS) subspace: capture = NLDR(feat_A_along_U) / NLDR(full_along_U),
one point per VALID DAS fit (train interchange acc >= 0.5), by task / layer / SAE width. Test split, f_pilot.
Fits that appear in several experiments with identical settings (E9 = E17 = E18 seed-0 3-epoch fits, bit-identical) are
plotted once. E22 RS rows (fresh template families rs_en / rs_fr / rs_hi / rs_hinglish, fitted per family) are a
separate column. Lines join the same fixed DAS U across SAE widths (rotation is width-independent; full_along_U identical).
Source: team/lead_capture.csv; RS/OT/PC 16k rows cross-checked against hw/tables/final/E18_illusion.csv (f_pilot).
No CIs: lead_capture.csv has point estimates only (paired width CIs are in F4). Exploratory/post hoc."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figures_common as F

F.setup()
d = pd.read_csv(F.TEAM / "lead_capture.csv")
d["valid"] = d.das_train_acc >= 0.5
assert d.valid.all(), d[~d.valid]
assert np.allclose(d.capture_nldr, d.nldr_feat_A_along_U / d.nldr_full_along_U)
key = ["task", "layer", "sae", "family", "estimator"]
dup = d.groupby(key).capture_nldr.agg(["min", "max", "size"])
assert (dup["max"] - dup["min"]).abs().max() < 1e-12, dup  # duplicates across E9/E17/E18 are identical
pref = {"E18": 0, "E17": 1, "E20": 2, "E22": 3, "E9": 4}
d["pref"] = d.exp.map(pref)
d = d.sort_values("pref").drop_duplicates(key).copy()
d["exps"] = d.apply(lambda r: "+".join(sorted(pd.read_csv(F.TEAM / "lead_capture.csv").query(
    "task == @r.task and layer == @r.layer and sae == @r.sae and family == @r.family and estimator == @r.estimator").exp)), axis=1)

# cross-check vs frozen E18 table (f_pilot)
T = pd.read_csv(F.FINAL / "E18_illusion.csv")
T = T[(T["filter"] == "f_pilot") & T.condition.isin(["full_along_U", "feat_A_along_U"])]
T = T.pivot_table(index=["task", "layer", "estimator"], columns="condition", values="nldr").reset_index()
m = d[d.exp == "E18"].merge(T, on=["task", "layer", "estimator"])
assert len(m) == (d.exp == "E18").sum() and np.allclose(m.nldr_full_along_U, m.full_along_U) \
    and np.allclose(m.nldr_feat_A_along_U, m.feat_A_along_U)
print("E18 rows match E18_illusion.csv f_pilot")

d["width"] = d.sae.str.extract(r"_(\d+k)_")
d["group"] = np.where(d.exp == "E22", "RS L18\n16k\nE22 fam.",
                      d.task.map({"add_chain": "RS", "entity_track": "OT", "people_cities": "PC"}) + " L"
                      + d.layer.astype(str) + "\n" + d.width)
GROUPS = ["RS L18\n16k", "RS L18\n65k", "OT L12\n16k", "OT L12\n65k", "OT L12\n262k", "PC L12\n16k", "PC L16\n16k",
          "RS L18\n16k\nE22 fam."]
assert set(d.group) <= set(GROUPS), set(d.group)
COL = {"add_chain": F.BLUE, "entity_track": F.ORANGE, "people_cities": F.AQUA}
MRK = {1: "^", 2: "v", 4: "o", 8: "s", 16: "D"}
d["k"] = d.estimator.str.extract(r"das(\d+)").astype(int)
d["variant"] = d.estimator.str.contains("_") | (d.exp == "E22")
xpos = {g: i for i, g in enumerate(GROUPS)}
d["x"] = d.group.map(xpos) + d.k.map({1: -0.24, 2: -0.12, 4: 0.0, 8: 0.12, 16: 0.24})
# spread exact overlaps (same group and rank): variants offset slightly
d["x"] = d["x"] + d.groupby(["group", "k"]).cumcount() * 0.07

fig, ax = plt.subplots(figsize=(F.SINGLE, 2.7))
for est in ["das4", "das8"]:
    for task, gs in [("add_chain", ["RS L18\n16k", "RS L18\n65k"]), ("entity_track", ["OT L12\n16k", "OT L12\n65k", "OT L12\n262k"])]:
        q = d[(d.task == task) & (d.estimator == est) & d.group.isin(gs)].sort_values("x")
        ax.plot(q.x, q.capture_nldr, color=COL[task], lw=0.7, alpha=0.6, zorder=1)
for _, r in d.iterrows():
    c = COL[r.task]
    ax.plot(r.x, r.capture_nldr, MRK[r.k], ms=4.3, mec=c, mew=1.0, mfc="white" if r.variant else c, zorder=3)
for i in range(len(GROUPS) - 1):
    if GROUPS[i][:2] != GROUPS[i + 1][:2] or i == len(GROUPS) - 2:
        ax.axvline(i + 0.5, color=F.GRID, lw=0.8, zorder=0)
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, fontsize=5.8)
ax.grid(axis="x", visible=False)
ax.set_xlim(-0.5, len(GROUPS) - 0.5)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Capture = NLDR(feat) / NLDR(full)")
ax.axhline(1.0, color=F.INK2, lw=0.7)
handles = [Line2D([], [], color=F.INK2, marker=MRK[k], ls="none", mfc=F.INK2, label=f"DAS-{k}") for k in [1, 2, 4, 8, 16]]
handles.append(Line2D([], [], color=F.INK2, marker="o", ls="none", mfc="white", label="seed/epoch variant\nor E22 family"))
ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=5.6, handlelength=1.0, columnspacing=0.8,
          bbox_to_anchor=(0.0, 1.0))
fig.tight_layout()
out = d.drop(columns=["pref"]).sort_values(["group", "k", "estimator"])
out["group"] = out.group.str.replace("\n", " ")
F.save(fig, "F7_capture", out)

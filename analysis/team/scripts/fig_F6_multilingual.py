"""F6 Multilingual text fidelity (E12 16k + E21 65k, per-window splice cost = (nll_splice - nll_clean)/pred, nats/token).
(a, b) native-script / English (en_pile) splice-cost ratio for the 6 native-script sets at L12 and L18, at three levels:
   raw (all windows; ratio of mean costs; 95% percentile bootstrap over windows within set, 1000 resamples, seed 0),
   matched on clean NLL (CEM, 0.25-nat bins) and matched on clean NLL + log |eps|/|x| (CEM), both from
   team/stats_out/mm_matched.csv (1000 bootstraps with matching redone). Hollow = unsupported cell (< 50% of native
   windows matched or < 100 windows), as defined in stats.md.
(c) 65k / 16k splice-cost ratio per set (all 12 sets) from team/multilingual_out/E21_width_by_set.csv (paired windows).
Sources: team/multilingual_out/windows.parquet, team/stats_out/mm_matched.csv, team/multilingual_out/E21_width_by_set.csv.
Exploratory/post hoc; uncorrected CIs."""
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figures_common as F

F.setup()
NATIVE = ["hi_deva_dakshina", "hi_deva_web", "ta_taml_dakshina", "ta_web", "bn_beng_dakshina", "bn_web"]
LATIN = ["en_pile", "fr_web", "hinglish_phinc", "hi_latn_dakshina", "ta_latn_dakshina", "bn_latn_dakshina"]
w = pd.read_parquet(F.TEAM / "multilingual_out" / "windows.parquet",
                    columns=["set", "layer", "width", "nll_splice", "nll_clean", "pred"])
w["cost"] = (w.nll_splice - w.nll_clean) / w.pred
mm = pd.read_csv(F.TEAM / "stats_out" / "mm_matched.csv")
mm = mm[(mm.method == "CEM") & (mm.ref == "en_pile") & mm.layer.isin([12, 18]) & mm.target.isin(NATIVE)
        & mm.spec.isin(["P", "PE"])].copy()
mm["supported"] = (mm.share_T_matched >= 0.5) & (mm.n_T_matched >= 100)

rng = np.random.default_rng(0)
rows = []
for (L, wd), g in w[w.layer.isin([12, 18])].groupby(["layer", "width"]):
    en = g[g.set == "en_pile"].cost.to_numpy()
    ben = rng.choice(en, size=(1000, len(en))).mean(1)
    for s in NATIVE:
        t = g[g.set == s].cost.to_numpy()
        bt = rng.choice(t, size=(1000, len(t))).mean(1)
        rat = bt / ben
        rows.append(dict(layer=L, width=wd, set=s, level="raw", ratio=t.mean() / en.mean(), lo=np.percentile(rat, 2.5),
                         hi=np.percentile(rat, 97.5), n_native=len(t), n_en=len(en), supported=True,
                         cost_native=t.mean(), cost_en=en.mean()))
        # consistency: a fully matched P cell has cost_T equal to the raw native mean
        p = mm[(mm.layer == L) & (mm.width == wd) & (mm.target == s) & (mm.spec == "P")]
        if len(p) and p.n_T_matched.iloc[0] == p.n_T.iloc[0]:
            assert np.isclose(p.cost_T.iloc[0], t.mean(), rtol=1e-6), (L, wd, s)
for _, r in mm.iterrows():
    rows.append(dict(layer=r.layer, width=r.width, set=r.target, level={"P": "matched_nllc", "PE": "matched_nllc_eps"}[r.spec],
                     ratio=r.ratio, lo=r.lo, hi=r.hi, n_native=r.n_T, n_native_matched=r.n_T_matched,
                     share_matched=r.share_T_matched, supported=bool(r.supported), cost_native=r.cost_T, cost_en=r.cost_R))
R = pd.DataFrame(rows)
LEV = [("raw", "Raw", F.VIOLET, "o"), ("matched_nllc", "Matched on clean NLL", F.ORANGE, "s"),
       ("matched_nllc_eps", "Matched on clean NLL + |ε|/|x|", F.BLUE, "D")]
LABEL = {"hi_deva_dakshina": "Hindi Dakshina", "hi_deva_web": "Hindi web", "ta_taml_dakshina": "Tamil Dakshina",
         "ta_web": "Tamil web", "bn_beng_dakshina": "Bengali Dakshina", "bn_web": "Bengali web",
         "en_pile": "English (Pile)", "fr_web": "French web", "hinglish_phinc": "Hinglish", "hi_latn_dakshina": "Hindi romanized",
         "ta_latn_dakshina": "Tamil romanized", "bn_latn_dakshina": "Bengali romanized"}

fig = plt.figure(figsize=(F.DOUBLE, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
ax16, ax65, axW = (fig.add_subplot(gs[0, i]) for i in range(3))
cells = [(L, s) for L in [12, 18] for s in NATIVE]
ypos = {c: len(cells) - 1 - i - (0.6 if c[0] == 18 else 0) for i, c in enumerate(cells)}
for ax, wd in [(ax16, "16k"), (ax65, "65k")]:
    for (L, s), y in ypos.items():
        q = R[(R.layer == L) & (R.width == wd) & (R.set == s)].set_index("level")
        xs = [q.loc[l, "ratio"] for l, *_ in LEV if l in q.index]
        ax.plot(xs, [y] * len(xs), color=F.AXIS, lw=0.8, zorder=1)
        for j, (l, _, c, m) in enumerate(LEV):
            if l not in q.index:
                continue
            r = q.loc[l]
            yy = y + (j - 1) * 0.22
            ax.hlines(yy, r.lo, r.hi, color=c, lw=0.9, zorder=2)
            ax.plot(r.ratio, yy, m, ms=3.8, color=c, mec=c, mfc=c if r.supported else "white", mew=1.0, zorder=3)
    ax.axvline(1, color=F.INK2, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.set_xlim(0.45, 6)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"L{L} {LABEL[s]}" for (L, s) in ypos], fontsize=6.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Splice cost, native / English")
    ax.set_title(f"({'a' if wd == '16k' else 'b'}) {wd} SAEs", fontsize=7.3)
ax65.set_yticklabels([])
handles = [Line2D([], [], color=c, marker=m, ls="none", mfc=c, label=lab) for _, lab, c, m in LEV]
handles.append(Line2D([], [], color=F.INK2, marker="D", ls="none", mfc="white", label="unsupported match"))

# (c) width
W = pd.read_csv(F.TEAM / "multilingual_out" / "E21_width_by_set.csv")
sets = NATIVE + LATIN
yw = {s: len(sets) - 1 - i - (0.6 if s in LATIN else 0) for i, s in enumerate(sets)}
for L, c, m, dy in [(12, F.BLUE, "o", 0.15), (18, F.ORANGE, "s", -0.15)]:
    q = W[W.layer == L].set_index("set")
    for s, y in yw.items():
        r = q.loc[s]
        axW.hlines(y + dy, r.ce_delta_ratio_lo, r.ce_delta_ratio_hi, color=c, lw=0.9)
        axW.plot(r.ce_delta_ratio, y + dy, m, ms=3.8, color=c, mec=c, mfc=c)
        R = pd.concat([R, pd.DataFrame([dict(layer=L, width="65k/16k", set=s, level="width_ratio_65k_over_16k",
                                             ratio=r.ce_delta_ratio, lo=r.ce_delta_ratio_lo, hi=r.ce_delta_ratio_hi,
                                             cost_native=r.ce_delta_16k, cost_en=r.ce_delta_65k, supported=True)])],
                      ignore_index=True)
axW.axvline(1, color=F.INK2, lw=0.7)
axW.set_yticks(list(yw.values())); axW.set_yticklabels([LABEL[s] for s in yw], fontsize=6.2)
axW.grid(axis="y", visible=False)
axW.set_xlim(0.3, 1.05)
axW.set_xlabel("Splice cost, 65k / 16k (95% CI)")
axW.set_title("(c) Width: 16k → 65k", fontsize=7.3)
axW.axhline((yw["bn_web"] + yw["en_pile"]) / 2, color=F.AXIS, lw=0.6)
axW.text(0.32, yw["bn_web"] - 0.1, "native script", fontsize=6, color=F.INK2, va="top")
axW.text(0.32, yw["en_pile"] + 0.55, "Latin script", fontsize=6, color=F.INK2, va="center")
axW.legend(handles=[Line2D([], [], color=F.BLUE, marker="o", ls="none", label="L12"),
                    Line2D([], [], color=F.ORANGE, marker="s", ls="none", label="L18")], loc="upper left", fontsize=6.0)
fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.37, -0.005))
R = R.rename(columns={"cost_native": "cost_native_or_16k", "cost_en": "cost_en_or_65k"})
F.save(fig, "F6_multilingual", R.sort_values(["level", "width", "layer", "set"]))

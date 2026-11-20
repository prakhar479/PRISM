"""Lead claim (2026-11-19, exploratory): share of the causal (DAS) subspace's effect carried by SAE features.
capture = NLDR(feat_A_along_U) / NLDR(full_along_U), per valid DAS fit (train interchange acc >= 0.5), filter f_pilot, native
fits only (fit family == eval family, or pilot). Writes analysis/team/lead_capture.csv with one row per (experiment, fit): fits
shared across experiments with identical settings (E9 = E17 = E18 seed-0 3-epoch rotations, bit-identical) appear once per
experiment, as fig_F7_capture.py expects; the printed summary counts each fit once. Run from the repository root."""
import pandas as pd

T = "tables/final/"
t = pd.concat([pd.read_csv(T + f"{e}_illusion.csv").assign(exp=e) for e in ["E9", "E17", "E18", "E20", "E22"]])
t = t[t.estimator.str.startswith("das") & (t["filter"] == "f_pilot") & t.condition.isin(["full_along_U", "feat_A_along_U"])]
k = ["exp", "task", "layer", "sae", "fit_families", "family", "estimator", "r"]
p = t.pivot_table(index=k, columns="condition", values=["nldr", "flip_rate"], aggfunc="first")
p.columns = [f"{a}_{b}" for a, b in p.columns]
p = p.reset_index().merge(t[t.condition == "full_along_U"][k + ["das_train_acc"]].drop_duplicates(k), on=k)
p = p[(p.das_train_acc >= 0.5) & ((p.family == p.fit_families) | (p.fit_families == "pilot"))]
p["capture_nldr"] = p.nldr_feat_A_along_U / p.nldr_full_along_U
p.to_csv("analysis/team/lead_capture.csv", index=False)
u = p.drop_duplicates(k[1:]).assign(group=lambda d: d.exp.where(d.exp == "E22", "native"))
print(u.groupby(["task", "layer", "sae", "group"]).capture_nldr.agg(["size", "median", "min", "max"]).round(3).to_string())

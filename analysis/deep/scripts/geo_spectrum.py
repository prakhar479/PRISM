"""H6: how concentrated is the last-position error (pca_pooled evr) per task/layer (E8), and does concentration predict the share
of the gap that same-task repair closes (E8_shares, E8 run, NLDR, f_pilot)? Plus E11 people-cities vs object tracking."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
import pandas as pd, glob as g
from scipy.stats import spearmanr
rows = []
for f in g.glob(f"{R}/E8/*/directions/*.json"):
    j = json.load(open(f)); n = f.split("/")[-1][:-5]; t, rest = n.split("_L", 1); L = int(rest.split("_")[0])
    e = j["pca_pooled"]["evr"]; a = j["answer_sep"]["evr"]
    rows.append(dict(task=t, layer=L, evr1=e[0], evr4=sum(e[:4]), evr8=sum(e[:8]), as_evr1=a[0], as_evr4=sum(a[:4]), n_classes=j["answer_sep"]["meta"].get("n_classes")))
ev = pd.DataFrame(rows)
s = pd.read_csv(f"{SP}/hw/tables/E8_shares.csv")
s = s[(s.exp_id == "E8") & (s["filter"] == "f_pilot") & (s.source == "same_task") & (s.metric == "nldr") & s.r.isin([1, 4])]
sh = s.pivot_table(index=["task", "layer"], columns=["estimator", "r"], values="share_closed", aggfunc="first")
sh.columns = [f"share_{a}_r{b}" for a, b in sh.columns]
gp = pd.read_csv(f"{SP}/hw/tables/E8_gaps.csv")
gp = gp[(gp.exp_id == "E8") & (gp["filter"] == "f_pilot") & (gp.metric == "nldr") & (gp.condition == "feat_A_errB")][["task", "layer", "gap"]]
d = ev.merge(sh.reset_index(), on=["task", "layer"], how="left").merge(gp, on=["task", "layer"], how="left")
d.to_csv(f"{SP}/deep/geo_spectrum_E8.csv", index=False)
pd.set_option("display.width", 220)
print(d.sort_values(["task", "layer"]).round(3).to_string(index=False))
ok = d.dropna(subset=["share_pca_pooled_r1", "gap"]); big = ok[ok.gap >= 0.05]
for col in ["share_pca_pooled_r1", "share_pca_pooled_r4", "share_answer_sep_r1"]:
    print(col, "vs evr1: all", spearmanr(ok.evr1, ok[col]), "| gap>=5pts n=", len(big), spearmanr(big.evr1, big[col]))
# E11 PC vs OT evr
r2 = []
for f in g.glob(f"{R}/E11/*/directions/*.json"):
    n = f.split("/")[-1][:-5]
    if n.startswith("people_cities") and "__fit" not in n: continue
    j = json.load(open(f)); t, rest = n.split("_L", 1); L = int(rest.split("_")[0])
    r2.append(dict(task=t, layer=L, evr1=j["pca_pooled"]["evr"][0], evr4=sum(j["pca_pooled"]["evr"][:4]), as_evr1=j["answer_sep"]["evr"][0]))
print(pd.DataFrame(r2).pivot_table(index="layer", columns="task", values=["evr1", "evr4", "as_evr1"]).round(3).to_string())

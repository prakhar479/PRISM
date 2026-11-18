"""H1 (decoder-free): does the 16k task-relevant error direction persist in the errors of wider SAEs (E4, matched / bracketing L0)?
If a wider dictionary absorbs it (feature splitting), the wider SAE's error PCA should no longer contain it."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
import pandas as pd, glob as g
rows = []
for t, L, ref in [("entity_track", 12, "16k_l082"), ("ioi_control", 12, "16k_l082"), ("add_chain", 18, "16k_l074")]:
    A = load("E4", f"{t}_L{L}_L{L}_{ref}")[0]
    for f in sorted(g.glob(f"{R}/E4/*/directions/{t}_L{L}_L{L}_*.npz")):
        s = f.split(f"_L{L}_L{L}_")[1][:-4]
        B = dict(np.load(f)); j = json.load(open(f[:-4] + ".json"))
        w = int(s.split("k_")[0]); l0 = int(s.split("_l0")[1])
        for est in ["pca_pooled", "answer_sep"]:
            rows.append(dict(task=t, sae=s, width_k=w, l0=l0, est=est, **{f"top_r{r}": top(A[est][:, :r], B[est][:, :r]) for r in (1, 2, 4, 8)},
                             cap_16k_r1_in_w_r8=capture(A[est][:, :1], B[est][:, :8]), cap_16k_r4_in_w_r8=capture(A[est][:, :4], B[est][:, :8]),
                             evr1=(j[est]["evr"] or [np.nan])[0], evr_sum4=float(np.sum((j[est]["evr"] or [np.nan])[:4]))))
d = pd.DataFrame(rows); d.to_csv(f"{SP}/deep/geo_width_E4.csv", index=False)
pd.set_option("display.width", 220)
print(d.sort_values(["task", "est", "width_k", "l0"]).round(3).to_string(index=False))
# relate to the gap (OT only; table from release-2 explore E)
gap = {"16k_l082": .252, "32k_l076": .197, "32k_l0155": .177, "65k_l072": .241, "65k_l0141": .134, "131k_l067": .186, "131k_l0129": .232,
       "262k_l067": .147, "262k_l0121": .145, "524k_l065": .190, "524k_l0115": .177}
x = d[(d.task == "entity_track") & (d.est == "pca_pooled") & (d.sae != "16k_l082")].copy(); x["gap"] = x.sae.map(gap)
from scipy.stats import spearmanr
print("OT: spearman(gap_nldr, top_r1 to 16k direction)", spearmanr(x.gap, x.top_r1), "cap r4", spearmanr(x.gap, x.cap_16k_r4_in_w_r8), "n", len(x))
print("OT: spearman(gap, evr1)", spearmanr(x.gap, x.evr1))

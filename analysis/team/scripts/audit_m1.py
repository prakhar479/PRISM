"""Audit preregistered M1 count (44/44): recovered_mean upper CI < English lower CI, per set x layer {12,18} x width {16k,65k}."""
import sys, glob, os; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
rec = {}
for run in glob.glob(f"{RUNS}/E12/*/"):
    m = json.load(open(run + "manifest.json")); sae = json.dumps(m["config"]).split('"width": ')[1].split(",")[0] if '"width": ' in json.dumps(m["config"]) else "?"
    for f in glob.glob(run + "per_window_*_L*.npz"):
        st, L = os.path.basename(f)[11:-4].rsplit("_L", 1)
        z = np.load(f); w = (z["nll_splice"] - z["nll_clean"]); a = (z["nll_mean"] - z["nll_clean"])
        rng = np.random.default_rng(0); n = len(w)
        bs = [1 - w[i].sum() / a[i].sum() for i in (rng.integers(0, n, n) for _ in range(500))]
        rec[(st, int(L), sae)] = (1 - w.sum() / a.sum(), *np.percentile(bs, [2.5, 97.5]))
keys = sorted(rec); sup = []
for (st, L, sae) in keys:
    if L not in (12, 18) or st.startswith("en_"): continue
    en = rec.get(("en_pile", L, sae))
    if en is None: continue
    sup.append((st, L, sae, rec[(st, L, sae)][2] < en[1]))
print("M1 cells", len(sup), "supported", sum(s[-1] for s in sup)); print([s for s in sup if not s[-1]])
print("widths seen", sorted({k[2] for k in keys}))

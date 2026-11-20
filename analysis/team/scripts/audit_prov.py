"""Audit provenance/reproducibility claims: bit-identity of per-pair logits across runs/nodes; laptop vs cluster gap agreement."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
KEY = ["pair_id", "task", "layer", "sae", "condition", "estimator", "r", "seed", "fit_families", "family"]
runs = {"E4(Poseidon-finalized)": ("E4", None), "E8(Poseidon-finalized)": ("E8", None), "E9(gnode084)": ("E9", None), "E2r(gnode084)": ("E2r", None),
        "E17(gnode058)": ("E17", None), "E18(gnode058)": ("E18", None), "E20(gnode058)": ("E20", None)}
conds = ["clean_A", "clean_B", "full_A", "feat_A_errB", "feat_A_meanPos"]
D = {}
for nm, (e, h) in (runs.items() if "--gaps" not in sys.argv else []):
    x = load(e, h); x = x[x.condition.isin(conds)]
    x["family"] = np.where(x.family.astype(str).str.startswith("pilot"), "pilot", x.family.astype(str))
    D[nm] = x[KEY + ["logit_A", "logit_B", "top1_id"]].drop_duplicates(KEY)
names = list(D) if "--gaps" not in sys.argv else []; tot = 0
for i, a in enumerate(names):
    for b in names[i + 1:]:
        m = D[a].merge(D[b], on=KEY, suffixes=("_a", "_b"))
        if len(m) == 0: continue
        dd = max(np.abs(m.logit_A_a - m.logit_A_b).max(), np.abs(m.logit_B_a - m.logit_B_b).max())
        print(f"{a} vs {b}: shared rows {len(m)} (logit values {2*len(m)}), max|Δlogit| {dd:.3g}, top1 agree {np.mean(m.top1_id_a == m.top1_id_b):.4f}")
# laptop (RTX 5060) vs cluster: RS L18 / OT L12 16k pilot gaps
for e in (["E0", "E1", "E9", "E18", "E8"] if "--gaps" not in sys.argv else ["E0", "E9", "E18", "E8", "E2", "E2r"]):
    for h in sorted(glob.glob(f"{RUNS}/{e}/*/")):
        if not os.path.exists(h + "per_pair.parquet"): continue
        x = load(e, h.split("/")[-2])
        for task, L in (("add_chain", 18), ("entity_track", 12)):
            g = x[(x.task == task) & (x.layer == L) & x.sae.str.startswith(f"L{L}_16k") & (x.fit_families.astype(str) == "pilot")]
            g = g[g.family.astype(str).str.startswith("pilot") | g.family.isna()]
            if not {"full_A", "feat_A_errB"} <= set(g.condition): continue
            c = Cell(g, "f_pilot")
            print(f"{e} {h.split('/')[-2]} {task} L{L}: flip gap {flip_rate(c, c.get('full_A')) - flip_rate(c, c.get('feat_A_errB')):.4f} NLDR gap {nldr(c, c.get('full_A')) - nldr(c, c.get('feat_A_errB')):.4f}")
# DAS rotations: E9 (gnode084) vs E17/E18/E20 (gnode058)
if "--das" in sys.argv:
    a = directions("E9")
    for e in ("E17", "E18", "E20"):
        b = directions(e)
        for stem, (za, ja) in a.items():
            for stemb, (zb, jb) in b.items():
                if stemb.split("_L")[0] != stem.split("_L")[0] or stemb.split("_L")[1][:2] != stem.split("_L")[1][:2]: continue
                for k in za.files:
                    if k.startswith("das") and k in zb.files:
                        print(f"DAS {stem} E9 vs {e} {stemb} {k}: array_equal {np.array_equal(za[k], zb[k])}, acc {ja[k]['meta']['train_interchange_acc']:.3f}/{jb[k]['meta']['train_interchange_acc']:.3f}")

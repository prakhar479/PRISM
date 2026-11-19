"""E17 (RS L18): validity vs training, held-out effect vs training, learned-component seed stability, consensus rank-1 direction,
per-pair behavioural agreement, cross-machine reproducibility (E9/E17/E18/E20). Post hoc, exploratory.
Outputs: das_out/e17_configs.csv, e17_pairs.csv, e17_mcnemar.csv, e17_crossmachine.csv (+ printed)."""
import itertools, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from das_lib import *
import pandas as pd
from scipy.stats import binomtest, spearmanr
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
STEM = "add_chain_L18_L18_16k_l074"
Z, J = load("E17", STEM)
das = [k for k in Z if k.startswith("das")]

# ---------- A. config table
il = pd.read_csv(f"{SP}/r5/tables/E17_illusion.csv"); il = il[il["filter"] == "f_pilot"]
rows = []
for k in das:
    p = das_parse(k); m = J[k]["meta"]; Lb, s, cap = learned(Z[k], k)
    g = il[il.estimator == k].set_index("condition")
    rows.append(dict(est=k, k=p["k"], epochs=p["epochs"], lr=p["lr"], seed=p["seed"], loss_first=m["loss_first_epoch"],
                     loss_last=m["loss_last_epoch"], train_acc=m["train_interchange_acc"], init_capture=cap,
                     full_flip=g.loc["full_along_U", "flip_rate"], full_lo=g.loc["full_along_U", "flip_rate_lo"],
                     full_hi=g.loc["full_along_U", "flip_rate_hi"], feat_flip=g.loc["feat_A_along_U", "flip_rate"],
                     full_nldr=g.loc["full_along_U", "nldr"], feat_nldr=g.loc["feat_A_along_U", "nldr"],
                     gap=g.loc["full_along_U", "illusion_diff"], gap_lo=g.loc["full_along_U", "illusion_diff_lo"],
                     gap_hi=g.loc["full_along_U", "illusion_diff_hi"], n=g.loc["full_along_U", "flip_n_eligible"]))
cf = pd.DataFrame(rows).sort_values(["k", "seed", "lr", "epochs"])
cf.to_csv(f"{OUT}/e17_configs.csv", index=False)
print("== A. E17 configs (f_pilot test, n eligible per row shown)"); print(cf.round(3).to_string(index=False))
for k in (1, 2):
    c = cf[(cf.k == k) & (cf.lr == 0.01)]
    print(f"  k={k}: Spearman(train_acc, full_flip) over {len(c)} lr=1e-2 fits:", spearmanr(c.train_acc, c.full_flip))
for k in (1, 2):
    c = cf[(cf.k == k) & (cf.lr == 0.01)]
    print(f"  k={k}: Spearman(init_capture, train_acc) within rank:", spearmanr(c.init_capture, c.train_acc), "n", len(c))
c = cf[cf.lr == 0.01]
print("  all lr=1e-2 fits: Spearman(train_acc, full_flip)", spearmanr(c.train_acc, c.full_flip), "n", len(c))

# ---------- B. per-pair: does more training change held-out flips? exact McNemar on paired flips
fl = flips("E17", das + ["pca_pooled"])
W = fl.pivot_table(index="pair_id", columns="estimator", values="flip")
print("n pairs in per-pair matrix", len(W))
comp = [("das1", "das1_e10"), ("das1_e10", "das1_e30"), ("das1", "das1_e30"), ("das2", "das2_e10"), ("das2_e10", "das2_e30"),
        ("das2", "das2_e30"), ("das2_e10", "das2_e10_lr3e-2"), ("das4", "das4_e10"), ("das8", "das8_e10"),
        ("das1_s1", "das1_e10_s1"), ("das1_s2", "das1_e10_s2"), ("das2_s1", "das2_e10_s1"), ("das2_s2", "das2_e10_s2")]
mc = []
for a, b in comp:
    x, y = W[a], W[b]; n01 = int(((x == 0) & (y == 1)).sum()); n10 = int(((x == 1) & (y == 0)).sum())
    pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    mc.append(dict(a=a, b=b, flip_a=x.mean(), flip_b=y.mean(), diff=y.mean() - x.mean(), n01=n01, n10=n10, p_mcnemar=pv, n=len(x)))
mc = pd.DataFrame(mc)
mc["p_bh"] = bh(mc.p_mcnemar.values)
mc.to_csv(f"{OUT}/e17_mcnemar.csv", index=False)
print("== B. paired held-out flip changes with training (exact McNemar, BH over", len(mc), "tests)"); print(mc.round(4).to_string(index=False))

# ---------- C. geometric seed stability: raw and learned, split by shared vs different init
pr = []
for a, b in itertools.combinations(das, 2):
    pa, pb = das_parse(a), das_parse(b)
    if pa["k"] != pb["k"]:
        continue
    La, _, _ = learned(Z[a], a); Lb, _, _ = learned(Z[b], b)
    xa, xb = W[a], W[b]
    both = int(((xa == 1) & (xb == 1)).sum()); ea = xa.mean() * xb.mean() * len(xa)
    phi = np.corrcoef(xa, xb)[0, 1] if xa.std() > 0 and xb.std() > 0 else np.nan
    pr.append(dict(k=pa["k"], a=a, b=b, same_init=pa["seed"] == pb["seed"], raw_top=top(Z[a], Z[b]),
                   raw_mean=float(pcos(Z[a], Z[b]).mean()), learned_top=top(La, Lb), learned_mean=float(pcos(La, Lb).mean()),
                   flip_a=xa.mean(), flip_b=xb.mean(), both_flip=both, both_expected_indep=ea, phi=phi,
                   valid_a=J[a]["meta"]["train_interchange_acc"] >= .5, valid_b=J[b]["meta"]["train_interchange_acc"] >= .5))
pr = pd.DataFrame(pr); pr.to_csv(f"{OUT}/e17_pairs.csv", index=False)
print("== C. pairwise same-rank comparisons"); print(pr.round(3).to_string(index=False))
ok = ~pr.a.str.contains("lr3e-2") & ~pr.b.str.contains("lr3e-2")
print("summary (excluding the collapsed lr3e-2 fits):")
print(pr[ok].groupby(["k", "same_init"])[["raw_top", "learned_top", "learned_mean", "phi"]].agg(["median", "min", "max", "count"]).round(3).to_string())

# check make_final's table
st = pd.read_csv(f"{SP}/r5/tables/E17_das_stability.csv")
mm = st.merge(pr, left_on=["das_a", "das_b"], right_on=["a", "b"])
print("max |diff| vs make_final cos_top_learned:", float((mm.cos_top_learned - mm.learned_top).abs().max()), "rows", len(mm))

# ---------- D. consensus rank-1 direction across das1 fits (learned component, excluding the collapsed lr3e-2 fit)
d1 = [k for k in das if das_parse(k)["k"] == 1 and "lr3e-2" not in k]
V = np.stack([learned(Z[k], k)[0][:, 0] for k in d1], 1)
w, ev = np.linalg.eigh(V @ V.T if False else (V.T @ V))  # small Gram
u_, s_, _ = np.linalg.svd(V, full_matrices=False); c1 = u_[:, 0]
print("== D. das1 consensus (top singular vector of", len(d1), "learned das1 directions); sv^2 share:",
      np.round(s_ ** 2 / (s_ ** 2).sum(), 3))
print("  cos^2 of each das1 learned dir with consensus:", {k: round(float((V[:, i] @ c1) ** 2), 3) for i, k in enumerate(d1)})
# leave-one-seed-out: consensus of seed-0 fits vs seed-1/2 fits
for grp in ([0], [1], [2]):
    a = [i for i, k in enumerate(d1) if das_parse(k)["seed"] in grp]; b = [i for i, k in enumerate(d1) if das_parse(k)["seed"] not in grp]
    ca = np.linalg.svd(V[:, a], full_matrices=False)[0][:, 0]; cb = np.linalg.svd(V[:, b], full_matrices=False)[0][:, 0]
    print(f"  consensus seed {grp} fits vs other seeds' consensus |cos| = {abs(float(ca @ cb)):.3f}")
print("  consensus inside learned span of higher-rank fits (cos^2; Haar ~k/2304):")
for k in das:
    if das_parse(k)["k"] >= 2:
        L = learned(Z[k], k)[0]
        print(f"    {k:16s} learned dims {L.shape[1]}  cos^2 {float(((L.T @ c1) ** 2).sum()):.3f}   in full U {float(((orth(Z[k]).T @ c1) ** 2).sum()):.3f}")
print("  consensus in pca_pooled[:, :r] capture:", {r: round(float(((Z['pca_pooled'][:, :r].T @ c1) ** 2).sum()), 3) for r in (1, 2, 4, 8, 16, 32)})
np.save(f"{OUT}/rs18_das1_consensus.npy", c1)

# ---------- F. cross-machine reproducibility: identical-settings DAS in E9 (gnode084?) vs E17/E18/E20 (gnode058)
cm = []
srcs = {"E9": "add_chain_L18_L18_16k_l074", "E17": STEM, "E18": "add_chain_L18_L18_16k_l074", "E20": "add_chain_L18_L18_65k_l064"}
L = {e: load(e, s) for e, s in srcs.items()}
for e, s in [("E9", "entity_track_L12_L12_16k_l082"), ("E18", "entity_track_L12_L12_16k_l082"), ("E20", "entity_track_L12_L12_65k_l072"),
             ("E20", "entity_track_L12_L12_262k_l067")]:
    L[e + ":OT:" + s[-4:]] = load(e, s)
import json as _j
def node(e):
    m = _j.load(open(glob.glob(f"{R}/{e}/*/manifest.json")[0]))
    s = _j.dumps(m); import re
    h = re.findall(r"gnode\d+", s)
    return sorted(set(h)), m.get("git", {}).get("commit", m.get("git")) if isinstance(m.get("git"), dict) else m.get("git")
for e in ("E9", "E17", "E18", "E20"):
    print("manifest", e, node(e))
pairs = [("E9", "E17"), ("E9", "E18"), ("E9", "E20"), ("E17", "E18"), ("E9:OT", "E18:OT"), ("E9:OT", "E20:OT65"), ("E9:OT", "E20:OT262")]
key = {"E9": L["E9"], "E17": L["E17"], "E18": L["E18"], "E20": L["E20"], "E9:OT": L["E9:OT:l082"], "E18:OT": L["E18:OT:l082"],
       "E20:OT65": L["E20:OT:l072"], "E20:OT262": L["E20:OT:l067"]}
for a, b in pairs:
    (Za, Ja), (Zb, Jb) = key[a], key[b]
    for k in sorted(set(Za) & set(Zb)):
        if not k.startswith("das"):
            continue
        cm.append(dict(a=a, b=b, est=k, max_abs_diff=float(np.abs(Za[k] - Zb[k]).max()), top=top(Za[k], Zb[k]),
                       min_pcos=float(pcos(Za[k], Zb[k]).min()), acc_a=Ja[k]["meta"]["train_interchange_acc"],
                       acc_b=Jb[k]["meta"]["train_interchange_acc"], loss_a=Ja[k]["meta"]["loss_last_epoch"],
                       loss_b=Jb[k]["meta"]["loss_last_epoch"]))
    for k in ("pca_pooled", "answer_sep"):
        if k in Za and k in Zb:
            cm.append(dict(a=a, b=b, est=k, max_abs_diff=float(np.abs(np.abs(Za[k][:, :8]) - np.abs(Zb[k][:, :8])).max()),
                           top=top(Za[k][:, :1], Zb[k][:, :1]), min_pcos=float(pcos(Za[k][:, :8], Zb[k][:, :8]).min())))
cm = pd.DataFrame(cm); cm.to_csv(f"{OUT}/e17_crossmachine.csv", index=False)
print("== F. cross-run/machine DAS and error-U identity"); print(cm.to_string(index=False))
# per-pair outcomes E9 vs E17 for plain dasK
f9 = flips("E9", ["das1", "das2", "das4", "das8"]); f9 = f9[f9.task == "add_chain"]
for k in ("das1", "das2", "das4", "das8"):
    a = f9[f9.estimator == k].set_index("pair_id").flip; b = W[k]
    j = a.index.intersection(b.index)
    print(f"  per-pair full_along_U flips {k}: E9 {a.loc[j].mean():.3f} E17 {b.loc[j].mean():.3f} agree {(a.loc[j] == b.loc[j]).mean():.3f} n {len(j)}")

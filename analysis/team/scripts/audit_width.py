"""Audit C6 (width: E9 16k vs E20 65k/262k) and the lead's capture-median claim (E9/E17/E18/E20/E22)."""
import sys, hashlib; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
B = 2000
def paired(c16, c, f16, f, stat, B=B):
    n = len(c16.pairs); rng = np.random.default_rng(0)
    pt = stat(c, f, None) - stat(c16, f16, None); s = []
    for _ in range(B):
        i = rng.integers(0, n, n); s.append(stat(c, f, i) - stat(c16, f16, i))
    return pt, np.percentile(s, 2.5), np.percentile(s, 97.5)
def cap(c, fe, fu, i=None): return nldr(c, fe, i) / nldr(c, fu, i)
def gapflip(c, f, i=None):
    full = c.flips(c.get("full_A")); x = c.flips(f); idx = np.flatnonzero(c.elig) if i is None else np.flatnonzero(c.elig[i]) 
    a = full if i is None else full[i]; b = x if i is None else x[i]
    return np.nanmean(a) - np.nanmean(b)
def gapnldr(c, f, i=None): return nldr(c, c.get("full_A"), i) - nldr(c, f, i)
def share(c, rp, i=None):
    fe = nldr(c, c.get("feat_A_errB"), i); fa = nldr(c, c.get("full_A"), i); return (nldr(c, rp, i) - fe) / (fa - fe)
e9, e20 = load("E9"), load("E20")
for task, L, wide in (() if "--capture-only" in sys.argv else (("add_chain", 18, ["L18_65k_l064"]), ("entity_track", 12, ["L12_65k_l072", "L12_262k_l067"]))):
  for flt in ("f_pilot", "f_A_correct", "f_none"):
    g16 = e9[(e9.task == task) & (e9.layer == L)]; c16 = Cell(g16, flt)
    print(f"\n== {task} L{L} {flt}: n {len(c16.pairs)} elig {c16.elig.sum()}")
    for est in ("das4", "das8"):
        fe16, fu16 = c16.get("feat_A_along_U", est), c16.get("full_along_U", est)
        print(f"  16k {est} capture {cap(c16, fe16, fu16):.3f}")
    for sae in wide:
        c = Cell(e20[e20.sae == sae], flt); assert (c.pairs == c16.pairs).all()
        for est in ("das4", "das8"):
            fe, fu = c.get("feat_A_along_U", est), c.get("full_along_U", est)
            pt, lo, hi = paired(c16, c, c16.get("feat_A_along_U", est), fe, lambda cc, f, i: nldr(cc, f, i), B=1000)
            print(f"  {sae} {est}: capture {cap(c, fe, fu):.3f} ; Δ feat NLDR {pt:+.3f} [{lo:+.3f},{hi:+.3f}] ; feat flip {flip_rate(c, fe):.3f} vs 16k {flip_rate(c16, c16.get('feat_A_along_U', est)):.3f}")
        for nm, st in (("flip gap", gapflip), ("NLDR gap", gapnldr)):
            pt, lo, hi = paired(c16, c, c16.get("feat_A_errB"), c.get("feat_A_errB"), st, B=1000)
            print(f"  {sae} feat_A_errB {nm}: 16k {st(c16, c16.get('feat_A_errB')):.3f} wide {st(c, c.get('feat_A_errB')):.3f} Δ {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
        if flt == "f_pilot":
            for est, r in (("pca_pooled", 1), ("pca_pooled", 8), ("answer_sep", 1)):
                print(f"  {sae} repair share {est} r{r}: 16k {share(c16, c16.get('repair', est, r=r)):.3f} wide {share(c, c.get('repair', est, r=r)):.3f}")
# ---------- lead capture claim
print("\n== capture = NLDR(feat_A_along_U)/NLDR(full_along_U), valid DAS fits, f_pilot, native fits")
rows = []
for exp in ("E9", "E17", "E18", "E20", "E22"):
    d = load(exp); dirs = directions(exp)
    d["family"] = np.where(d.family.astype(str).str.startswith("pilot"), "pilot", d.family.astype(str))
    for (task, L, sae, fit, fam), g in d.groupby(["task", "layer", "sae", "fit_families", "family"]):
        if exp == "E22" and str(fit) != str(fam): continue
        stem = [k for k in dirs if k.startswith(f"{task}_L{L}_") and (exp != "E22" or k.endswith(f"fit-{fit}") or (fit in ("pilot",) ))]
        c = Cell(g, "f_pilot")
        for est in sorted(e for e in g.estimator.unique() if e.startswith("das")):
            zz, jj = None, None
            for k in dirs:
                z, j = dirs[k]
                if k.startswith(f"{task}_L{L}_") and est in z.files and (exp != "E22" or k.endswith(f"fit-{fit}")):
                    zz, jj = z, j; break
            if zz is None: continue
            acc = jj[est]["meta"]["train_interchange_acc"]
            fe, fu = c.get("feat_A_along_U", est), c.get("full_along_U", est)
            if fe.top1_id.isna().any(): continue
            h = hashlib.md5(np.ascontiguousarray(zz[est]).tobytes()).hexdigest()[:8]
            rows.append(dict(exp=exp, task=task, layer=L, sae=sae, fam=fam, est=est, acc=acc, valid=acc >= .5, capture=cap(c, fe, fu), U=h, n=len(c.pairs)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/capture_all.csv", index=False)
V = R[R.valid]
for (task, L, sae), g in V.groupby(["task", "layer", "sae"]):
    u = g.drop_duplicates(["U", "fam"])
    print(f"  {task} L{L} {sae}: rows {len(g)} median {g.capture.median():.3f} | unique (U,family) {len(u)} median {u.capture.median():.3f} | excl E22 unique {u[u.exp!='E22'].capture.median():.3f} (n {len(u[u.exp!='E22'])})")
print(V.sort_values(["task", "layer", "sae", "exp"]).round(3).to_string(index=False))

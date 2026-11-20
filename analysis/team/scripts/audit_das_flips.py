"""Audit C1/C2/C5 behavioural DAS / error-U claims from E17 and E18 per_pair.parquet (f_pilot unless stated)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
from scipy.stats import spearmanr, binomtest
B = 2000

def rnull(c, cond, r):
    v = [flip_rate(c, c.get(cond, "random", r=r, seed=s)) for s in range(20)]
    return np.nanpercentile(v, 95)

def eff(c, cond, est, r, p95):
    x = c.flips(c.get(cond, est, r=r))[c.elig]; pt, lo, hi, _ = boot(lambda i: x[i].mean(), len(x), B)
    return pt, lo, hi, (pt >= p95 + .10) and (lo > p95)

def gapclosed(c, est, r):
    fe = nldr(c, c.get("feat_A_errB")); rp = nldr(c, c.get("repair", est, r=r)); return (rp - fe) / (1 - fe), fe

# ---------------- E17
d = load("E17"); c = Cell(d, "f_pilot")
z, j = directions("E17")["add_chain_L18_L18_16k_l074"]
das = sorted(k for k in z.files if k.startswith("das"))
print("E17 n eligible", c.elig.sum())
p95 = {r: rnull(c, "full_along_U", r) for r in (1, 2, 4, 8)}
print("random p95 full_along_U:", p95)
rows = []
for k in das:
    f = c.flips(c.get("full_along_U", k))[c.elig]; g = c.flips(c.get("feat_A_along_U", k))[c.elig]
    pt, lo, hi, _ = boot(lambda i: f[i].mean(), len(f), B)
    rows.append(dict(fit=k, k=parse_das(k)["k"], seed=parse_das(k)["seed"], acc=j[k]["meta"]["train_interchange_acc"], full=pt, lo=lo, hi=hi,
                     halfw=(hi - lo) / 2, feat=g.mean()))
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(f"{OUT}/e17_flips.csv", index=False)
print("Spearman acc vs full flip (all %d fits):" % len(R), spearmanr(R.acc, R.full))
q = R[~R.fit.str.contains("lr3e-2")]; print("Spearman lr1e-2 only (n=%d):" % len(q), spearmanr(q.acc, q.full))
r1 = R[(R.k == 1) & ~R.fit.str.contains("lr3e-2")]
print("rank-1 lr1e-2 flips range %.3f-%.3f; seed-0 rank-1 %s" % (r1.full.min(), r1.full.max(), r1[r1.seed == 0].full.round(3).tolist()))
print("median CI half-width over fits %.3f (range %.3f-%.3f)" % (R.halfw.median(), R.halfw.min(), R.halfw.max()))
# seed contrasts: exact McNemar per contrast, BH within seed family
tests = []
for k in das:
    p = parse_das(k)
    if p["seed"] == 0: continue
    base = k.replace(f"_s{p['seed']}", "")
    for cond in ("full_along_U", "feat_A_along_U"):
        a = c.flips(c.get(cond, base))[c.elig]; b = c.flips(c.get(cond, k))[c.elig]
        n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
        pv = binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0
        tests.append(dict(base=base, seedfit=k, cond=cond, base_flip=a.mean(), seed_flip=b.mean(), diff=b.mean() - a.mean(), n01=n01, n10=n10, p=pv))
T = pd.DataFrame(tests); T["q_bh"] = bh(T.p); print(T.round(4).to_string(index=False)); T.to_csv(f"{OUT}/e17_seed_tests.csv", index=False)

# ---------------- E18
d18 = load("E18")
for (task, L), g in d18.groupby(["task", "layer"]):
    c = Cell(g, "f_pilot"); print(f"\n== E18 {task} L{L}: n pairs {len(c.pairs)} eligible {c.elig.sum()}")
    zz, jj = [v for kk, v in directions("E18").items() if kk.startswith(f"{task}_L{L}_")][0]
    for r in (1, 2, 4, 8, 16, 32, 64):
        P = rnull(c, "full_along_U", r); Pf = rnull(c, "feat_A_along_U", r)
        for est in ("pca_pooled", "answer_sep"):
            fu = eff(c, "full_along_U", est, r, P); fe = eff(c, "feat_A_along_U", est, r, Pf)
            gc, fb = gapclosed(c, est, r)
            nflip = int(np.nansum(c.flips(c.get("full_along_U", est, r=r))))
            print(f"  {est:10s} r{r:2d}: full {fu[0]:.3f} [{fu[1]:.3f},{fu[2]:.3f}] p95 {P:.3f} eff={fu[3]} | feat {fe[0]:.3f} p95 {Pf:.3f} eff={fe[3]} | repair gap closed {gc:.3f} (featErrB NLDR {fb:.3f}) | nflips {nflip}")
        dk = f"das{r}"
        if dk in zz.files:
            fu = eff(c, "full_along_U", dk, r, P); fe = eff(c, "feat_A_along_U", dk, r, Pf); gc, _ = gapclosed(c, dk, r)
            a = c.flips(c.get("full_along_U", dk))[c.elig]; b = c.flips(c.get("feat_A_along_U", dk))[c.elig]
            dd = a - b; pt, lo, hi, _ = boot(lambda i: dd[i].mean(), len(dd), B)
            print(f"  {dk:10s}    : acc {jj[dk]['meta']['train_interchange_acc']:.3f} full {fu[0]:.3f} eff={fu[3]} | feat {fe[0]:.3f} eff={fe[3]} | full-feat {pt:.3f} [{lo:.3f},{hi:.3f}] | repair gap closed {gc:.3f}")
    if task == "add_chain":
        print("  das16 init capture", round(learned(zz["das16"], "das16")[1], 3))

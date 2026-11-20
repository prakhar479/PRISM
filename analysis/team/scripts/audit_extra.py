"""Audit leftovers: donor-order bug share (Corrections), cost-vs-patchability per pair (Other patterns), E10 Q2 max flip."""
import sys, gzip; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
from scipy.stats import spearmanr
def prompts(name):
    return {json.loads(l)["pair_id"]: json.loads(l) for l in gzip.open(f"{SP}/hw/data/prompt_sets/{name}.jsonl.gz")}
def select(rs, seed=0):
    n = len(rs); fam = np.array([r.get("family", "pilot") for r in rs], dtype=object)
    T = np.array([r["T"] for r in rs]); fd = np.array([r["first_diff"] for r in rs]); aA = np.array([r["ans_A"] for r in rs]); aB = np.array([r["ans_B"] for r in rs])
    da = np.concatenate([aA, aB]); dp = np.concatenate([np.arange(n)] * 2); pr = np.array(["A"] * n + ["B"] * n); rng = np.random.default_rng(seed); out = {}
    for i in range(n):
        ok = (dp != i) & (fam[dp] == fam[i]) & (T[dp] == T[i]) & (fd[dp] <= fd[i])
        cand = np.flatnonzero(ok & (da != aA[i]) & (da != aB[i]))
        if not len(cand): cand = np.flatnonzero(ok & (da != aA[i]))
        if not len(cand): out[rs[i]["pair_id"]] = None; continue
        k = int(rng.choice(cand)); out[rs[i]["pair_id"]] = (rs[dp[k]]["pair_id"], pr[k])
    return out
for exp, pset, sub in (("E2", "add_chain_pilot", None), ("E2r", "add_chain_pilot", None), ("E3", "add_chain_e3", "pilot")):
    P = prompts(pset); d = load(exp, "5ad7ada7f971" if exp == "E2" else None); d = d[(d.task == "add_chain")]
    if sub: d = d[d.family.astype(str).str.startswith("pilot") & (d.fit_families == "fresh")]
    ids = list(d.pair_id.unique()); rs = [P[i] for i in ids]
    a = select(sorted(rs, key=lambda r: (r.get("family", "pilot"), r["rank"]))); b = select(sorted(rs, key=lambda r: r["pair_id"]))
    print(f"donor order {exp}: n {len(ids)} share identical diff-answer donors {np.mean([a[p] == b[p] for p in ids]):.3f}")
# cost vs patchability
d = load("E18")
for (task, L), g in d.groupby(["task", "layer"]):
    c = Cell(g, "f_none"); both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
    pp = lambda f: ((c.D(f) - c.DB) / (c.DA - c.DB))[both]
    print(f"cost-vs-patch E18 {task} L{L}: Spearman(featErrB NLDR, das4 full NLDR) {spearmanr(pp(c.get('feat_A_errB')), pp(c.get('full_along_U','das4')))[0]:+.2f} n {both.sum()}")
# E10 Q2 (RS L18): best flip with random p95 <= 0.05
d = load("E10"); g = d[d.task == "add_chain"]; c = Cell(g, "f_pilot"); best = []
for cond in sorted(x for x in g.condition.unique() if x.startswith("steer")):
    for (est, r), gg in g[(g.condition == cond) & (g.estimator != "random")].groupby(["estimator", "r"]):
        rp = np.nanpercentile([flip_rate(c, c.get(cond, "random", r=r, seed=s)) for s in sorted(g[(g.condition == cond) & (g.estimator == "random") & (g.r == r)].seed.unique())] or [np.nan], 95)
        best.append((flip_rate(c, c.get(cond, est, r=r)), rp, cond, est, r))
best.sort(reverse=True); print("E10 RS top steer flips (flip, random p95, cond, est, r):", [(round(a, 3), round(b, 3), c_, e, r) for a, b, c_, e, r in best[:4]])
print("E10 Q2 met anywhere (flip>=0.5 & p95<=0.05):", any(a >= .5 and b <= .05 for a, b, *_ in best))

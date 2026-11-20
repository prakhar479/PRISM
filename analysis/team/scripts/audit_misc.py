"""Audit C5 (PC city selectivity, steering landing), C7 per-pair (E22 task-state loss, transfer), C8 (decision vs logit,
gain slope), 'other robust patterns' (start-0, E22 replication, low-rank profiles, graded resample), C1 error-U filters."""
import sys, gzip, re; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
from scipy.stats import spearmanr
def prompts(name):
    out = {}
    for l in gzip.open(f"{SP}/hw/data/prompt_sets/{name}.jsonl.gz"):
        x = json.loads(l); out[x["pair_id"]] = x
    return out
def pp_nldr(c, f): return (c.D(f) - c.DB) / (c.DA - c.DB)
def eta2(y, g):
    y = np.asarray(y, float); m = np.isfinite(y); y, g = y[m], np.asarray(g)[m]
    return sum(((y[g == k].mean() - y.mean()) ** 2) * (g == k).sum() for k in np.unique(g)) / ((y - y.mean()) ** 2).sum()
SEL = sys.argv[1:] or ["c1", "c5", "c7", "c8", "start", "modes", "resamp"]
# ---------------- C1 error-U filters
if "c1" in SEL:
    d = load("E18"); g = d[d.task == "add_chain"]
    for flt in ("f_pilot", "f_A_correct", "f_none"):
        c = Cell(g, flt)
        print(f"C1 E18 RS {flt} elig {c.elig.sum()}: " + ", ".join(f"{e[:3]} r{r} {flip_rate(c, c.get('full_along_U', e, r=r)):.3f}" for e in ("pca_pooled", "answer_sep") for r in (32, 64)))
# ---------------- C5
if "c5" in SEL:
    P = prompts("people_cities_en"); d = load("E18")
    for L in (12, 16):
        g = d[(d.task == "people_cities") & (d.layer == L)]; c = Cell(g, "f_none")
        both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
        city = np.array([P[p]["clean_target"].strip() for p in c.pairs])
        e = []
        for cond in ("full_along_U", "feat_A_along_U"):
            for k in (1, 2, 4, 8, 16):
                y = pp_nldr(c, c.get(cond, f"das{k}"))[both]; e.append(eta2(np.clip(y, -1, 2), city[both]))
        # permutation null for one key
        y = np.clip(pp_nldr(c, c.get("full_along_U", "das4"))[both], -1, 2); rng = np.random.default_rng(0)
        nul = [eta2(y, rng.permutation(city[both])) for _ in range(300)]
        y1 = pp_nldr(c, c.get("full_along_U", "das1"))[both]; cm = pd.Series(y1).groupby(city[both]).mean().sort_values()
        print(f"C5 PC L{L}: n both-correct {both.sum()}, cities {len(set(city[both]))}; eta2 median over 10 DAS keys {np.median(e):.3f} (range {min(e):.3f}-{max(e):.3f}); unclipped-das4 null p95 {np.percentile(nul,95):.3f}")
        print(f"   das1 full NLDR by A city: top {cm.index[-1]} {cm.iloc[-1]:.2f}; next max {cm.iloc[-2]:.2f} ({cm.index[-2]})")
    d19 = load("E19"); g = d19[d19.task == "people_cities"]; c = Cell(g, "f_pilot")
    name = {}
    for x in P.values(): name[x["ans_A"]] = x["clean_target"].strip(); name[x["ans_B"]] = x["clean_target"].strip() if False else x["corrupt_target"].strip()
    for est in ("das8", "das4"):
        t = c.get("steer_x16", est).top1_id.to_numpy()
        land = [name.get(v, f"id{v}") for v, a, b, e in zip(t, c.ansA, c.ansB, c.elig) if e and v != a and v != b and v in name]
        s = pd.Series(land).value_counts()
        print(f"C5 E19 PC {est} s=16: non-A/B candidate landings {len(land)}; top {s.index[0]} {s.iloc[0]/len(land):.3f}")
# ---------------- C7 per-pair (E22)
if "c7" in SEL:
    d = load("E22")
    for (task, fit), g in d.groupby(["task", "fit_families"]):
        g = g[g.family.astype(str) == str(fit)]
        if g.empty: continue
        c = Cell(g, "f_A_correct"); gap = nldr(c, c.get("full_A")) - nldr(c, c.get("feat_A_errB"))
        print(f"C7 E22 {task} {fit}: n {len(c.pairs)} feat_A_errB NLDR gap {gap:.3f}")
    # transfer: English-fit rows evaluated on other families vs English home
    for task, home in (("add_chain", "rs_en"), ("people_cities", "pc_en")):
        h = Cell(d[(d.task == task) & (d.family == home) & (d.fit_families == home)], "f_A_correct")
        for est in ("das4", "das8"):
            hv = flip_rate(h, h.get("full_along_U", est))
            out = []
            for fam in sorted(d[d.task == task].family.unique()):
                if fam == home: continue
                gg = d[(d.task == task) & (d.family == fam)]
                ce = Cell(gg[gg.fit_families == home], "f_A_correct"); cn = Cell(gg[gg.fit_families == fam], "f_A_correct")
                fe = ce.flips(ce.get("full_along_U", est))[ce.elig]; hn = h.flips(h.get("full_along_U", est))[h.elig]
                rng = np.random.default_rng(0); s = [fe[rng.integers(0, len(fe), len(fe))].mean() - hn[rng.integers(0, len(hn), len(hn))].mean() for _ in range(2000)]
                out.append(f"{fam}: eng-fit {np.nanmean(fe):.3f} (n {len(fe)}) native {flip_rate(cn, cn.get('full_along_U', est)):.3f}; eng-fit minus home {np.nanmean(fe)-hv:+.3f} [{np.percentile(s,2.5):+.3f},{np.percentile(s,97.5):+.3f}]")
            print(f"C7 E22 {task} {est} full_along_U home {hv:.3f} (n {h.elig.sum()}) | " + " | ".join(out))
# ---------------- C8
if "c8" in SEL:
    d = load("E1", h="854a2370f878")
    for task, L in (("ioi_control", 20), ("entity_track", 12), ("add_chain", 18)):
        c = Cell(d[(d.task == task) & (d.layer == L)], "f_pilot"); f = c.get("feat_A_errB")
        print(f"C8 E1 {task} L{L}: NLDR gap {nldr(c, c.get('full_A')) - nldr(c, f):.3f}, flip gap {flip_rate(c, c.get('full_A')) - flip_rate(c, f):.3f}, elig {c.elig.sum()}")
    e8 = load("E8"); res = []
    for (task, L), g in e8.groupby(["task", "layer"]):
        c = Cell(g, "f_pilot"); both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
        f = c.get("feat_A_errB"); loss = (c.DA - c.D(f))[both]; sw = (c.DA - c.DB)[both]
        X = np.c_[np.ones(len(sw)), sw]; rng = np.random.default_rng(0)
        bt = [np.linalg.lstsq(X[i], loss[i], rcond=None)[0] for i in (rng.integers(0, len(sw), len(sw)) for _ in range(1000))]
        b = np.linalg.lstsq(X, loss, rcond=None)[0]; lo, hi = np.percentile(np.array(bt)[:, 1], [2.5, 97.5])
        res.append(dict(task=task, L=L, n=both.sum(), slope=b[1], lo=lo, hi=hi, icpt=b[0]))
    R = pd.DataFrame(res); R.to_csv(f"{OUT}/gain_slopes.csv", index=False)
    print("C8 gain: cells", len(R), "slope CI > 0 in", int((R.lo > 0).sum())); print(R[(R.task == "add_chain") & (R.L == 18)].round(3).to_string(index=False))
    print(R[R.lo > 0][["task", "L", "slope", "lo"]].round(3).to_string(index=False))
# ---------------- start-0 pattern
def start_of(x):
    return int(re.findall(r"\d+", x["clean_prompt"].split("\n")[-1])[0])
if "start" in SEL:
    P = prompts("add_chain_pilot"); d = load("E18"); c = Cell(d[d.task == "add_chain"], "f_none")
    both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
    st = np.array([start_of(P[p]) for p in c.pairs])[both]
    for est, r in (("das4", 4), ("pca_pooled", 8)):
        y = pp_nldr(c, c.get("full_along_U", est, r=r))[both]
        print(f"start E18 RS n {both.sum()} {est} r{r}: " + ", ".join(f"start {k}: {y[st==k].mean():.2f} (n {(st==k).sum()})" for k in sorted(set(st))))
    P13 = prompts("add_chain_e13"); d = load("E22"); d = d[d.task == "add_chain"]; pos = 0; tot = 0; lines = []
    for fam in sorted(d.family.unique()):
        for est, r in (("das4", 4), ("das8", 8), ("pca_pooled", 4), ("pca_pooled", 8)):
            ys, ss = [], []
            for fit in d[d.family == fam].fit_families.unique():
                c = Cell(d[(d.family == fam) & (d.fit_families == fit)], "f_none"); both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
                ys.append(pp_nldr(c, c.get("full_along_U", est, r=r))[both]); ss.append(np.array([start_of(P13[p]) for p in c.pairs])[both])
            rho = spearmanr(np.concatenate(ys), np.concatenate(ss))[0]; tot += 1; pos += rho > 0; lines.append(f"{fam} {est} r{r} {rho:+.2f}")
    print(f"start E22: rho>0 in {pos}/{tot}: " + "; ".join(lines))
# ---------------- low-rank profiles
if "modes" in SEL:
    d = load("E18")
    for (task, L), g in d.groupby(["task", "layer"]):
        c = Cell(g, "f_none"); both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
        cols = []
        for (cond, est, r), gg in g[~g.condition.isin(["clean_A", "clean_B", "full_A", "full_B"])].groupby(["condition", "estimator", "r"]):
            if est == "random":
                v = np.nanmean([pp_nldr(c, c.get(cond, "random", r=r, seed=s)) for s in range(20)], 0)
            else:
                v = pp_nldr(c, c.get(cond, est, r=r))
            cols.append(v[both])
        M = np.clip(np.array(cols).T, -1, 2); M = M[:, np.isfinite(M).all(0)]; M = M - M.mean(0)
        s = np.linalg.svd(M, compute_uv=False); ev = s ** 2 / (s ** 2).sum()
        rng = np.random.default_rng(0); h = rng.permutation(len(M)); A_, B_ = M[h[: len(M)//2]], M[h[len(M)//2:]]
        v1 = lambda X: np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
        print(f"modes E18 {task} L{L}: n {len(M)} cols {M.shape[1]} evr PC1-3 {np.round(ev[:3],3).tolist()} split-half |cos V1| {abs(v1(A_) @ v1(B_)):.2f}")
# ---------------- graded resample (E3 RS L18, eval fresh)
if "resamp" in SEL:
    DIGIDS = [235276, 235274, 235284, 235304, 235310, 235308, 235318, 235324, 235321, 235315]
    P = prompts("add_chain_e3"); d = load("E3", extra=("donor_status",)); d = d[(d.task == "add_chain") & (d.layer == 18) & (d.fit_families == "fresh")]
    d = d[~d.family.astype(str).str.startswith("pilot")]   # E3 config: the fit=fresh/eval=fresh cell's te holds fresh-family pairs only
    ids = sorted(d.pair_id.unique(), key=lambda p: (P[p].get("family", "pilot"), P[p]["rank"]))
    meta = {k: np.array([P[p][k] for p in ids]) for k in ("T", "first_diff", "ans_A", "ans_B")}; fam = np.array([P[p].get("family", "pilot") for p in ids])
    def select(same, seed=0):
        rng = np.random.default_rng(seed); n = len(ids); da = np.concatenate([meta["ans_A"], meta["ans_B"]]); dp = np.concatenate([np.arange(n)] * 2); out = []
        for i in range(n):
            ok = (dp != i) & (fam[dp] == fam[i]) & (meta["T"][dp] == meta["T"][i]) & (meta["first_diff"][dp] <= meta["first_diff"][i])
            cand = np.flatnonzero(ok & (da == meta["ans_A"][i])) if same else np.flatnonzero(ok & (da != meta["ans_A"][i]) & (da != meta["ans_B"][i]))
            if not same and not len(cand): cand = np.flatnonzero(ok & (da != meta["ans_A"][i]))
            out.append(-1 if not len(cand) else da[int(rng.choice(cand))])
        return np.array(out)
    sa, di = select(True), select(False)
    st = d[d.condition == "feat_A_resampC_sameAns"].drop_duplicates("pair_id").set_index("pair_id").donor_status.reindex(ids)
    print("resamp: reconstructed 'none' same-answer donors", int((sa == -1).sum()), "vs stored", int((st == "none").sum()), "; agree on which pairs:", bool(((sa == -1) == (st == "none").to_numpy()).all()))
    dist = {p: abs(DIGIDS.index(a) - DIGIDS.index(b)) if a in DIGIDS and b in DIGIDS else np.nan for p, a, b in zip(ids, meta["ans_A"], di)}
    ev = d[~d.family.astype(str).str.startswith("pilot")]
    for flt in ("f_pilot", "f_A_correct", "f_none"):
        c = Cell(ev, flt); dd = pp_nldr(c, c.get("feat_A_resampC_sameAns")) - pp_nldr(c, c.get("feat_A_resampC_diffAns"))
        dv = np.array([dist[p] for p in c.pairs]); ok = (c.get("feat_A_resampC_sameAns").donor_status == "ok").to_numpy() & c.elig
        parts = []
        for k in (1, 2, 3, 4):
            x = dd[ok & (dv == k)]
            if len(x) < 5: continue
            rng = np.random.default_rng(0); s = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
            parts.append(f"±{k}: {x.mean():+.3f} [{np.percentile(s,2.5):+.3f},{np.percentile(s,97.5):+.3f}] n {len(x)}")
        print(f"resamp E3 fresh {flt}: " + " | ".join(parts))
    # red-team-style: both-correct pairs, common denominator mean(D_fullA - D_B)
    c = Cell(ev, None); both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA) & (c.get("feat_A_resampC_sameAns").donor_status == "ok").to_numpy()
    den = (c.D(c.get("full_A")) - c.DB)[both].mean()
    dd = ((c.D(c.get("feat_A_resampC_sameAns")) - c.D(c.get("feat_A_resampC_diffAns"))) / den)[both]; dv = np.array([dist[p] for p in c.pairs])[both]
    parts = []
    for k in (1, 2, 3, 4):
        x = dd[dv == k]; rng = np.random.default_rng(0); sb = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        parts.append(f"±{k}: {x.mean():+.3f} [{np.percentile(sb,2.5):+.3f},{np.percentile(sb,97.5):+.3f}] n {len(x)}")
    print("resamp E3 fresh both-correct, common denominator: " + " | ".join(parts))

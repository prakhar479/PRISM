"""Audit C3 (RS steering) and C4 (OT prefix fallback) from E19 / E10b per_pair + prompt files."""
import sys, gzip; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
from scipy.stats import fisher_exact, spearmanr
S = ["0.25", "0.5", "1", "2", "4", "8", "16"]
DIGIDS = [235276, 235274, 235284, 235304, 235310, 235308, 235318, 235324, 235321, 235315]  # cand_ids of add_chain prompts = digits 0..9 (NOT contiguous)
DIG = np.vectorize(lambda t: DIGIDS.index(t) if t in DIGIDS else -100)
d = load("E19")
# ---------------- RS
rs = d[d.task == "add_chain"]
for flt in ("f_pilot", "f_none"):
    c = Cell(rs, flt)
    dA = DIG(c.ansA); dB = DIG(c.ansB)
    both = c.elig & (c.ca.top1_id.to_numpy() == c.ansA)
    print(f"\n== RS {flt}: pairs {len(c.pairs)} eligible(cleanB=B) {c.elig.sum()} both-correct {both.sum()}")
    for nm, m in (("eligible", c.elig), ("both", both)):
        dl = (dA - dB)[m]; vals, cnt = np.unique(dl, return_counts=True)
        print(f"  Δ=A-B dist ({nm}, n={m.sum()}):", {int(v): round(k / m.sum(), 3) for v, k in zip(vals, cnt)}, "max", round(cnt.max() / m.sum(), 3))
    # one-sided flips
    for est, s in (("das4", "4"), ("das8_e10", "8"), ("das8", "16"), ("answer_sep", "16"), ("pca_pooled", "16")):
        for r in sorted(rs[rs.estimator == est].r.unique()):
            f = c.flips(c.get(f"steer_x{s}", est, r=r))
            for nm, m in (("eligible", c.elig), ("both", both)):
                up = m & (dA > dB); dn = m & (dA < dB)
                a, b = np.nansum(f[up]), np.nansum(f[dn])
                p = fisher_exact([[a, up.sum() - a], [b, dn.sum() - b]])[1]
                if nm == "both" or est == "das4":
                    print(f"  {est} r{r} s={s} [{nm}]: flip A>B {a/up.sum():.3f} (n {up.sum()}), A<B {b/dn.sum():.3f} (n {dn.sum()}), Fisher p {p:.2g}")
    # specificity das4 s=4 (f_pilot)
    f4 = c.flips(c.get("steer_x4", "das4"))[c.elig]
    rnd = np.array([c.flips(c.get("steer_x4", "random", r=4, seed=k))[c.elig] for k in range(20)])
    print(f"  das4 s=4 flip {f4.mean():.3f}; random r4 mean {rnd.mean():.4f} p95 {np.percentile(rnd.mean(1), 95):.4f} max {rnd.mean(1).max():.4f}")
    rng = np.random.default_rng(0); sp = []
    for _ in range(1000):
        i = rng.integers(0, len(f4), len(f4)); sp.append(f4[i].mean() - np.percentile(rnd[:, i].mean(1), 95))
    sp = np.array(sp); print(f"  spec {f4.mean()-np.percentile(rnd.mean(1),95):.3f} [{np.percentile(sp,2.5):.3f},{np.percentile(sp,97.5):.3f}], boot p(spec<=0) {(np.sum(sp<=0)+1)/(len(sp)+1):.4f}")
    # trajectories (both-correct pairs): top1 digit across doses ('1' coded 10)
    if flt == "f_none":
        for est, r in (("das4", 4), ("pca_pooled", 1), ("pca_pooled", 4)):
            T = DIG(np.array([c.get(f"steer_x{s}", est, r=r).top1_id.to_numpy() for s in S]).T[both]).astype(float)
            T = np.where(T == 1, 10, T); T = np.where((T < 0) | (T > 10), np.nan, T)
            ok = ~np.isnan(T).any(1); T = T[ok]
            moved = (T.max(1) != T.min(1)); df = np.diff(T, axis=1)
            mono_up = moved & (df >= 0).all(1); mono_dn = moved & (df <= 0).all(1)
            below = (T < (dB[both][ok])[:, None]).any(1)
            print(f"  traj {est} r{r}: n {len(T)} moved {moved.mean():.3f}; of moved: mono up {mono_up.sum()/moved.sum():.3f}, mono down {mono_dn.sum()/moved.sum():.3f}; any dose below B {below.mean():.3f}; mean change s.25->16 {np.mean(T[:,-1]-T[:,0]):.2f}")
# ---------------- E10b replication
d10 = load("E10b")
for fam, g in d10[d10.task == "add_chain"].groupby("family"):
    c = Cell(g, "f_none"); dA = DIG(c.ansA); dB = DIG(c.ansB)
    for est in ("das1", "das2", "das4"):
        for s in sorted({x.split("x")[1] for x in g.condition.unique() if x.startswith("steer")}):
            f = c.flips(c.get(f"steer_x{s}", est))
            if np.all(np.isnan(f)): continue
            up = c.elig & (dA > dB); dn = c.elig & (dA < dB)
            if s == "4" or np.nansum(f[dn]) > 0:
                print(f"E10b {fam} {est} s={s}: A>B {np.nanmean(f[up]):.3f} A<B {np.nanmean(f[dn]):.3f} (n {up.sum()}/{dn.sum()})")
# ---------------- OT
ot = d[d.task == "entity_track"]
cand = {}
with gzip.open(f"{SP}/hw/data/prompt_sets/entity_track_pilot.jsonl.gz") as fh:
    for line in fh:
        x = json.loads(line); cand[x["pair_id"]] = x["cand_ids"]
c = Cell(ot, "f_pilot"); print(f"\n== OT f_pilot pairs {len(c.pairs)} eligible {c.elig.sum()}")
cands = [cand.get(p) for p in c.pairs]; print("pairs with cand ids:", sum(x is not None for x in cands), "n cands", set(len(x) for x in cands if x))
for r in (1, 4, 8):
    for s in ("0.25", "0.5", "1", "2"):
        nA = nnb = 0
        for k in range(20):
            f = c.get(f"steer_x{s}", "random", r=r, seed=k); t = f.top1_id.to_numpy()
            for i in np.flatnonzero(c.elig):
                if t[i] != c.ansB[i] and t[i] in cands[i]:
                    nnb += 1; nA += t[i] == c.ansA[i]
        print(f"  random r{r} s={s}: A share of non-B candidate outcomes {nA/nnb:.4f} (n {nnb})")
margin = (c.cb.logit_B - c.cb.logit_A).to_numpy()[c.elig]
for r in (1, 4, 8):
    fr = np.mean([c.flips(c.get("steer_x1", "random", r=r, seed=k))[c.elig] for k in range(20)], 0)
    print(f"  Spearman(frac seeds flipping, clean-B margin) r{r} s=1: {spearmanr(fr, margin)[0]:.3f} n {len(fr)}")
for est in ("das4", "das8", "das8_e10", "pca_pooled", "answer_sep"):
    for r in sorted(ot[ot.estimator == est].r.unique()):
        print(f"  {est} r{r} s=0.25 flip {flip_rate(c, c.get('steer_x0.25', est, r=r)):.3f}")
for r in (1, 4, 8):
    v = [flip_rate(c, c.get("steer_x0.25", "random", r=r, seed=k)) for k in range(20)]
    print(f"  random r{r} s=0.25 mean {np.mean(v):.3f} p95 {np.nanpercentile(v,95):.3f} max {np.max(v):.3f}")
# noise control x̂_B + η, OT L12 16k pilot
for exp in ("E8", "E3", "E1"):
    try:
        g = load(exp)
    except Exception as e:
        continue
    g = g[(g.task == "entity_track") & (g.layer == 12) & g.sae.str.startswith("L12_16k") & ~g.family.astype(str).str.match("^(et_|fresh)")]
    if "feat_B_noise" not in set(g.condition): continue
    for (fit, h), gg in g.groupby(["fit_families", run_dir(exp)]) if False else g.groupby(["fit_families", "sae"]):
        for flt in ("f_pilot", "f_A_correct", "f_none"):
            cc = Cell(gg, flt); v = [flip_rate(cc, cc.get("feat_B_noise", None, seed=k)) for k in sorted(gg[gg.condition == "feat_B_noise"].seed.unique())]
            print(f"  {exp} OT L12 {h} fit {fit} {flt}: feat_B_noise flip mean over seeds {np.mean(v):.3f} ({len(v)} seeds; per-seed {np.round(v,3).tolist()}) elig {cc.elig.sum()}")

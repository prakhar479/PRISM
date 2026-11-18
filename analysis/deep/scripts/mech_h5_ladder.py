"""H4/H5 (exploratory, post hoc): per-condition answer structure in RS L18 and OT L12 (E3 fresh-template run + E2).
For eligible pairs (clean A top-1 = A, clean B top-1 = B): top-1 class (A, B, near-miss = |digit - A| = 1 and not B, other);
for RS a uniform-wrong-answer null for near misses; candidate-softmax distances; NLDR (ratio of means vs full_A)."""
from mech_common import *
CANDS = [235276, 235274, 235284, 235304, 235310, 235308, 235318, 235324, 235321, 235315]
DIG = {t: i for i, t in enumerate(CANDS)}
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)

def ladder(pp, task):
    pp = wide(pp[(pp.task == task) & (pp.estimator.fillna("") == "")]).drop_duplicates(["pair_id", "condition", "seed"])
    ref = pp[pp.condition.isin(["clean_A", "clean_B", "full_A"])].pivot_table(index="pair_id", columns="condition", values=["D", "isA", "isB"], aggfunc="first")
    elig = ref.index[(ref[("isB", "clean_B")] == True) & (ref[("isA", "clean_A")] == True)] if ("isA", "clean_A") in ref else ref.index[ref[("isB", "clean_B")] == True]
    DA = ref[("D", "full_A")]; DB = ref[("D", "clean_B")]
    rows = []
    for cond, g in pp[pp.pair_id.isin(elig)].groupby("condition"):
        g = g.groupby("pair_id").agg(D=("D", "mean"), isA=("isA", "mean"), isB=("isB", "mean"), top1=("top1_id", "first"),
                                     a=("ans_A_id", "first"), b=("ans_B_id", "first"), cl=("cand_logits", "first"))
        num = g.D - DB.loc[g.index]; den = DA.loc[g.index] - DB.loc[g.index]
        nl, lo, hi = boot_ratio(num.values, den.values)
        r = dict(cond=cond, n=len(g), flip=g.isA.mean(), toB=g.isB.mean(), nldr=nl, nldr_lo=lo, nldr_hi=hi)
        if task == "add_chain":
            t = g.top1.map(DIG); a = g.a.map(DIG); b = g.b.map(DIG)
            wrong = (g.isA < 0.5) & (g.isB < 0.5) & t.notna()
            near = wrong & ((t - a).abs() == 1)
            # null: a wrong non-B digit drawn uniformly from the 8 candidates != A, B
            null = np.array([sum(1 for d in range(10) if d not in (ai, bi) and abs(d - ai) == 1) / 8 for ai, bi in zip(a, b)])
            r.update(n_wrong=int(wrong.sum()), near=int(near.sum()), near_frac=near.sum() / max(wrong.sum(), 1),
                     near_null=null[wrong.values].mean() if wrong.any() else np.nan,
                     towardB=(np.sign(t - a) == np.sign(b - a))[wrong].mean() if wrong.any() else np.nan)
            # candidate softmax: expected |digit - A| and mass on A-adjacent digits (continuous)
            P = np.stack([np.exp(c - np.nanmax(c)) / np.nansum(np.exp(c - np.nanmax(c))) for c in g.cl])
            dist = np.abs(np.arange(10)[None, :] - a.values[:, None])
            r.update(E_absdiff=float((P * dist).sum(1).mean()), pA=float(P[np.arange(len(P)), a.values].mean()),
                     p_adj=float((P * (dist == 1) * (np.arange(10)[None, :] != b.values[:, None])).sum(1).mean()))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("flip")

for exp in ["E3", "E2"]:
    pp = load(exp)
    if exp == "E3":
        pp = pp[pp.fit_families == "fresh"]
    for task in ["add_chain", "entity_track"]:
        t = ladder(pp, task)
        print(f"== {exp} {task}"); print(t.round(3).to_string(index=False))
        t.to_csv(SP / "deep" / f"h5_ladder_{exp}_{task}.csv", index=False)

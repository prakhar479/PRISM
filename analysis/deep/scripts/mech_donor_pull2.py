"""Donor pull with a matched control (exploratory, post hoc): for each RS pair, d_i = the diffAns donor's answer (reconstructed,
see mech_donor_pull.py). Compare P(top1 = d_i) under resampC_diffAns with P(top1 = d_i) under conditions that do NOT carry the
donor's error (feat_A_zero, feat_A_meanPos, feat_A_noise, feat_A_covnoise, feat_A_meanTask, resampC_sameAns) on the same pairs.
Also candidate-softmax mass on d_i (continuous). Paired bootstrap CIs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_common import *
sys.path.insert(0, str(SP / "hw"))
from harness.conditions import select_donors
pd.set_option("display.width", 250)
CTRL = ["feat_A_resampC_diffAns", "feat_A_resampC_sameAns", "feat_A_zero", "feat_A_meanPos", "feat_A_meanTask", "feat_A_noise", "feat_A_covnoise", "feat_A_errB"]
rows = []
for exp, seed, prom in [("E2", 0, "add_chain_pilot"), ("E2r", 1, "add_chain_pilot"), ("E3", 0, "add_chain_e3")]:
    pp = load(exp, cols=["pair_id", "task", "fit_families", "condition", "estimator", "seed", "ans_A_id", "ans_B_id", "top1_id", "cand_logits"])
    pp = pp[(pp.task == "add_chain") & (pp.estimator.fillna("") == "")]
    if exp == "E3": pp = pp[pp.fit_families == "fresh"]
    pr = prompts(prom).set_index("pair_id")
    if exp == "E3":
        pr = pd.concat([pr, prompts("add_chain_pilot").set_index("pair_id")]); pr = pr[~pr.index.duplicated()]
    ids = sorted(pp.pair_id.unique()); m = pr.loc[ids]
    meta = {"T": m["T"].astype(int).values, "first_diff": m.first_diff.astype(int).values, "ans_A": m.ans_A.astype(int).values,
            "ans_B": m.ans_B.astype(int).values, "family": m.family.values}
    dn = select_donors(meta, False, seed)
    dans = pd.Series([(meta["ans_A"][d["j"]] if d["prompt"] == "A" else meta["ans_B"][d["j"]]) if d["j"] >= 0 else -1 for d in dn], index=ids)
    cand = m.cand_ids.apply(lambda s: [int(x) for x in (json.loads(s) if isinstance(s, str) else s)])
    top = pp.pivot_table(index="pair_id", columns="condition", values="top1_id", aggfunc="first")
    A = pp.drop_duplicates("pair_id").set_index("pair_id").reindex(top.index)
    el = top.index[(top.clean_A == A.ans_A_id) & (top.clean_B == A.ans_B_id) & (dans.loc[top.index] >= 0)]
    hit = {c: (top.loc[el, c] == dans.loc[el]).astype(float) for c in CTRL if c in top}
    # candidate-softmax mass on the donor answer (noise conditions: mean over seeds)
    mass = {}
    for c in CTRL:
        g = pp[pp.condition == c]
        if g.empty: continue
        v = {}
        for pid, cl in g.groupby("pair_id").cand_logits:
            if pid not in el: continue
            ci = cand.loc[pid].index(dans.loc[pid]) if dans.loc[pid] in cand.loc[pid] else None
            if ci is None: continue
            ps = [np.exp(x - np.nanmax(x)) / np.nansum(np.exp(x - np.nanmax(x))) for x in cl]
            v[pid] = float(np.mean([p[ci] for p in ps]))
        mass[c] = pd.Series(v)
    base = hit["feat_A_resampC_diffAns"]
    for c in hit:
        d_ = (base - hit[c]).values
        m_, lo, hi = boot_ci(d_)
        mm = mass[c].reindex(el).dropna(); mb = mass["feat_A_resampC_diffAns"].reindex(mm.index)
        md, mlo, mhi = boot_ci((mb - mm).values)
        rows.append(dict(exp=exp, cond=c, n=len(el), p_top1_eq_donorAns=hit[c].mean(), diffAns_minus_this=m_, lo=lo, hi=hi,
                         mass_on_donorAns=mm.mean(), mass_diff=md, mass_lo=mlo, mass_hi=mhi))
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "donor_pull_controlled.csv", index=False)
print(t.round(4).to_string(index=False))

"""Does a transplanted error pull the answer toward the DONOR's answer? (exploratory, post hoc)
Reconstructs the donor choice of harness.conditions.select_donors (same inputs, same seed) for E2/E2r/E3 cells, validates it
against the stored donor_status, then asks: under feat_A_resampC_diffAns, how often is top-1 = the donor's answer, versus the
rate for a matched non-donor wrong candidate (chance)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_common import *
sys.path.insert(0, str(SP / "hw"))
from harness.conditions import select_donors
pd.set_option("display.width", 250)

def run(exp, seed):
    rows = []
    pp = load(exp, cols=["pair_id", "task", "layer", "family", "fit_families", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "donor_status", "cand_logits", "logit_A"])
    pp = pp[pp.condition.isin(["feat_A_resampC_diffAns", "feat_A_resampC_sameAns", "clean_A", "clean_B"])]
    for (task, fit), g in pp.groupby(["task", "fit_families"]):
        pr = prompts({"add_chain": "add_chain_pilot" if exp != "E3" else "add_chain_e3", "entity_track": "entity_track_pilot" if exp != "E3" else "entity_track_e3"}[task]).set_index("pair_id")
        if exp == "E3":
            pr = pd.concat([pr, prompts(task + "_pilot").set_index("pair_id")])
            pr = pr[~pr.index.duplicated()]
        ids = sorted(g.pair_id.unique())
        # one cell per eval-family group in E3; in E2 one cell. Group by the stored cell = (layer) and eval families set
        for fams, sub_ids in [("all", ids)]:
            m = pr.loc[sub_ids]
            meta = {"T": m["T"].astype(int).values, "first_diff": m.first_diff.astype(int).values,
                    "ans_A": m.ans_A.astype(int).values, "ans_B": m.ans_B.astype(int).values, "family": m.family.values}
            for same in (False, True):
                cond = "feat_A_resampC_sameAns" if same else "feat_A_resampC_diffAns"
                dn = select_donors(meta, same, seed)
                st = g[g.condition == cond].drop_duplicates("pair_id").set_index("pair_id").loc[sub_ids]
                rec = np.array([d["status"] for d in dn]); stored = st.donor_status.values
                match = (rec == stored).mean()
                dans = np.array([(meta["ans_A"][d["j"]] if d["prompt"] == "A" else meta["ans_B"][d["j"]]) if d["j"] >= 0 else -1 for d in dn])
                cA = g[g.condition == "clean_A"].drop_duplicates("pair_id").set_index("pair_id").loc[sub_ids]
                cB = g[g.condition == "clean_B"].drop_duplicates("pair_id").set_index("pair_id").loc[sub_ids]
                el = (cA.top1_id.values == cA.ans_A_id.values) & (cB.top1_id.values == cB.ans_B_id.values) & (dans >= 0)
                top = st.top1_id.values
                cands = pr.loc[sub_ids].cand_ids.apply(lambda s: json.loads(s) if isinstance(s, str) else list(s)).values
                # chance: for each eligible pair, prob a uniformly chosen wrong candidate (not A, not B) equals donor = 1/(K-2)
                pull = (top == dans)[el].mean()
                other = ((top != cA.ans_A_id.values) & (top != cB.ans_B_id.values))[el]
                chance_pull_given_other = np.mean([1 / max(1, len([c for c in cc if c not in (a, b)])) for cc, a, b in zip(cands[el], meta["ans_A"][el], meta["ans_B"][el])])
                rows.append(dict(exp=exp, task=task, fit=fit, cond=cond, n=int(el.sum()), status_match=match, top1_eq_donor=pull,
                                 other_rate=other.mean(), donor_given_other=(top == dans)[el][other].mean() if other.any() else np.nan,
                                 chance_given_other=chance_pull_given_other, n_other=int(other.sum())))
    return rows

rows = run("E2", 0) + run("E2r", 1) + run("E3", 0)
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "donor_pull.csv", index=False)
print(t.round(3).to_string(index=False))

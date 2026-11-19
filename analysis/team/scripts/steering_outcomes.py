"""Where does the probability mass / top-1 go under steering? (E19, filter f_pilot, eligible pairs)

OT L12: role of the top-1 box per pair. Roles from the final story line (the query story):
  A      = A's answer (= the queried object's box before the changed step: implied by the unpatched prefix)
  B      = B's answer (= destination of the changed move; in A that same move is made by another object)
  other_mentioned = another box named in the query story (other objects' initial boxes / move destinations)
  unmentioned     = a box letter A-G not named in the query story
  noncand = top-1 is not one of the 7 box tokens
  Null for 'lands on A specifically': among non-B candidate outcomes, uniform over the 6 non-B boxes -> 1/6; and a role-weighted
  null (uniform over non-B boxes, so P(A) = 1/6, P(other_mentioned) = n_mentioned_nonAB/6, ...).
  Also: per-pair fragility (fraction of the 20 random seeds that flip the pair) vs clean-B margin D_B and p_B.
RS L18: flip rate split by delta = A - B (sign / size); the shift of the top-1 value relative to B (top1 - B) under steering.
PC L12: concentration of 'other candidate' landings (which city; is it a city named in the query story?).
Outputs: steering_out/ot_roles.csv, ot_fragility.csv, rs_delta.csv, rs_shift.csv, pc_landing.csv
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from steering_common import prompts, OUT

pp = pd.read_parquet(OUT / "pp_E19.parquet")
pp = pp[pp.f_pilot]
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ OT roles
P = prompts("entity_track")
BOX = {}  # token id -> letter
roles = {}
for pid, r in P.iterrows():
    q = r.clean_prompt.split("\n")[-1]
    letters = set(re.findall(r"box (\w)\b", q))
    A, Bv = r.clean_target.strip(), r.corrupt_target.strip()
    cands = r.cand_ids
    # candidate token order: map token ids to letters via the answer ids of A and B is not enough; use ' A'..' G' order
    roles[pid] = dict(A=A, B=Bv, mentioned=letters, cand=cands, ansA=r.ans_A, ansB=r.ans_B)
# recover token id -> letter from the pairs (ans_A/ans_B ids with their letters)
for pid, d in roles.items():
    BOX[d["ansA"]] = d["A"]
    BOX[d["ansB"]] = d["B"]
assert len(BOX) == 7, BOX


def ot_role(pid, tid):
    d = roles[pid]
    if tid not in BOX:
        return "noncand"
    L = BOX[tid]
    if L == d["A"]:
        return "A"
    if L == d["B"]:
        return "B"
    return "other_mentioned" if L in d["mentioned"] else "unmentioned"


ot = pp[(pp.task == "entity_track") & pp.elig & pp.condition.str.startswith("steer_")].copy()
ot["role"] = [ot_role(p, t) for p, t in zip(ot.pair_id, ot.top1_id)]
n_ment = {pid: len(d["mentioned"] - {d["A"], d["B"]}) for pid, d in roles.items()}
ot["n_ment_other"] = ot.pair_id.map(n_ment)
ot["grp"] = np.where(ot.estimator == "random", "random", ot.estimator)
rows = []
for (grp, r, s), g in ot.groupby(["grp", "r", "s"]):
    c = g.role.value_counts(normalize=True)
    nonB = g[g.role.isin(["A", "other_mentioned", "unmentioned"])]
    d = dict(grp=grp, r=r, s=s, n_rows=len(g), n_pairs=g.pair_id.nunique(),
             **{f"share_{k}": float(c.get(k, 0)) for k in ["A", "B", "other_mentioned", "unmentioned", "noncand"]},
             n_nonB_cand=len(nonB))
    if len(nonB):
        fA = float((nonB.role == "A").mean())
        exp_ment = float((nonB.n_ment_other / 6).mean())
        # pair-cluster bootstrap of the A share among non-B candidate outcomes
        pids = nonB.pair_id.unique()
        grpd = {p: (nonB.role[nonB.pair_id == p] == "A").to_numpy() for p in pids}
        bs = []
        for _ in range(1000):
            sp = rng.choice(pids, len(pids))
            v = np.concatenate([grpd[p] for p in sp])
            bs.append(v.mean())
        d.update(A_among_nonB=fA, A_among_nonB_lo=np.percentile(bs, 2.5), A_among_nonB_hi=np.percentile(bs, 97.5),
                 null_A=1 / 6, ment_among_nonB=float((nonB.role == "other_mentioned").mean()), null_ment=exp_ment)
    rows.append(d)
R = pd.DataFrame(rows)
R.to_csv(OUT / "ot_roles.csv", index=False)
pd.set_option("display.width", 250)
print("OT roles (eligible, f_pilot)")
print(R.round(3).to_string(index=False))

# fragility: per pair fraction of random seeds flipping, vs clean B margin
ref = pp[(pp.task == "entity_track") & (pp.condition == "clean_B") & pp.elig][["pair_id", "D", "p_B"]].drop_duplicates("pair_id")
fr = []
for (r, s), g in ot[ot.grp == "random"].groupby(["r", "s"]):
    f = g.groupby("pair_id").apply(lambda x: (x.role == "A").mean()).rename("frac_A").reset_index().merge(ref, on="pair_id")
    rho, p = spearmanr(f.frac_A, -f.D) if f.frac_A.std() > 0 else (np.nan, np.nan)
    rho2, p2 = spearmanr(f.frac_A, -f.p_B) if f.frac_A.std() > 0 else (np.nan, np.nan)
    # do the same pairs flip under every seed? share of pairs flipped by >= 1 seed / by all seeds
    fr.append(dict(r=r, s=s, n_pairs=len(f), mean_frac=f.frac_A.mean(), pairs_any=(f.frac_A > 0).mean(),
                   pairs_all=(f.frac_A == 1).mean(), pairs_none=(f.frac_A == 0).mean(),
                   rho_fragility_vs_negDB=rho, p=p, rho_vs_negpB=rho2, p2=p2))
F = pd.DataFrame(fr)
F.to_csv(OUT / "ot_fragility.csv", index=False)
print(F.round(3).to_string(index=False))

# ------------------------------------------------------------------ RS delta
Pr = prompts("add_chain")
dig = {}
for pid, r in Pr.iterrows():
    dig[r.ans_A] = int(r.clean_target.strip()) if r.clean_target.strip().isdigit() else None
    dig[r.ans_B] = int(r.corrupt_target.strip()) if r.corrupt_target.strip().isdigit() else None
rs = pp[(pp.task == "add_chain") & pp.elig & pp.condition.str.startswith("steer_")].copy()
rs["delta"] = rs.pair_id.map(Pr.state_clean - Pr.state_corrupt)
rs["sB"] = rs.pair_id.map(Pr.state_corrupt)
rs["top_val"] = rs.top1_id.map(dig)
rs["shift"] = rs.top_val - rs.sB
rs["grp"] = np.where(rs.estimator == "random", "random", rs.estimator + "_r" + rs.r.astype(str))
out = []
for (grp, s), g in rs.groupby(["grp", "s"]):
    d = dict(grp=grp, s=s, n=len(g))
    for dl in [-3, -2, -1, 1, 2, 3]:
        x = g[g.delta == dl]
        d[f"flip_d{dl:+d}"] = float(x.isA.mean()) if len(x) else np.nan
    d["flip_pos"] = float(g[g.delta > 0].isA.mean())
    d["flip_neg"] = float(g[g.delta < 0].isA.mean())
    d["n_pos"] = int((g.delta > 0).sum())
    d["n_neg"] = int((g.delta < 0).sum())
    sh = g["shift"].dropna()
    d["mean_shift"] = float(sh.mean())
    d["share_shift_pos"] = float((sh > 0).mean())
    d["share_shift_neg"] = float((sh < 0).mean())
    d["share_noshift"] = float((sh == 0).mean())
    out.append(d)
D = pd.DataFrame(out)
D.to_csv(OUT / "rs_delta.csv", index=False)
print(D[D.grp != "random"].round(3).to_string(index=False))
print(D[D.grp == "random"].round(3).to_string(index=False))

# shift distribution for das + random at s = 4, 8, 16
sh = (rs[rs.s.isin([4, 8, 16])].groupby(["grp", "s"])["shift"].value_counts(normalize=True).rename("share").reset_index())
sh.to_csv(OUT / "rs_shift.csv", index=False)

# ------------------------------------------------------------------ PC landing
Pc = prompts("people_cities")
city = {}
for pid, r in Pc.iterrows():
    city[r.ans_A] = r.clean_target.strip()
    city[r.ans_B] = r.corrupt_target.strip()
ment = {pid: set(re.findall(r"in (\w+)", r.clean_prompt.split("\n")[-1])) for pid, r in Pc.iterrows()}
pc = pp[(pp.task == "people_cities") & pp.elig & pp.condition.str.startswith("steer_") & (pp.outcome == "other_cand")].copy()
pc["city"] = pc.top1_id.map(city)
pc["mentioned"] = [c in ment[p] for p, c in zip(pc.pair_id, pc.city)]
pc["grp"] = np.where(pc.estimator == "random", "random", pc.estimator + "_r" + pc.r.astype(str))
L = []
for (grp, s), g in pc.groupby(["grp", "s"]):
    vc = g.city.value_counts(normalize=True)
    L.append(dict(grp=grp, s=s, n=len(g), top_city=vc.index[0], top_share=vc.iloc[0], n_cities=g.city.nunique(),
                  share_mentioned=g.mentioned.mean()))
L = pd.DataFrame(L)
L.to_csv(OUT / "pc_landing.csv", index=False)
print(L.round(3).to_string(index=False))

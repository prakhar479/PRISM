"""Dose-response of steering (E19 s = 0.25..16; E10 / E10b s = 1, 2, 4).

Per (exp, task, eval family, filter, estimator, r, seed, scale): flip rate (eligible pairs), NLDR, prob recovery, mean p_A /
p_B / p(other candidates) / p(non-candidate tokens), and the top-1 outcome shares on eligible pairs (A, B, other candidate,
non-candidate). Random-U envelope per (task, r, scale): mean / p95 (np.nanpercentile, as harness/final.py) / max over the 20 seeds.
Specificity = estimator flip - random p95, with a pair-bootstrap CI (1000 resamples; the random p95 is recomputed in every
resample, so the CI includes the null's sampling noise).

Outputs (steering_out/): dose_rows.csv (every key), dose_summary.csv (estimators + envelope + specificity),
pp_E19.parquet / pp_E10.parquet / pp_E10b.parquet (tidy per-pair frames used by the other steering_* scripts).
"""
import numpy as np
import pandas as pd
from steering_common import load, attach_refs, flip, nldr, prob_rec, OUT

B = 1000
rng = np.random.default_rng(0)


def fam(df):
    f = df.family.fillna("pilot").astype(str)
    return df.assign(family=np.where(f.str.startswith("pilot"), "pilot", f))


def summarise(g):
    e = g[g.elig]
    out = dict(n_pairs=len(g), n_elig=len(e), flip=flip(g), nldr=nldr(g), prob_rec=prob_rec(g),
               pA=g.p_A.mean(), pB=g.p_B.mean(), p_other_cand=g.p_other_cand.mean(), p_noncand=(1 - g.p_cand).mean(),
               D=g.D.mean())
    if len(e):
        cand = e.p_cand  # placeholder to keep e in scope
        oc = np.where(e.isA, "A", np.where(e.isB, "B", "x"))
        out.update(top_A=(oc == "A").mean(), top_B=(oc == "B").mean())
    return out


frames = {}
for exp in ["E19", "E10", "E10b"]:
    pp = fam(load(exp))
    pp = attach_refs(pp, keycols=("task", "layer", "family"))
    # top-1 among candidates? we need whether a non-A/B top-1 is a candidate token: use cand ids from prompts
    frames[exp] = pp

# candidate membership of top-1 (is the top-1 token one of the task's candidate answers?)
from steering_common import prompts
cand_sets = {}
for t in ["add_chain", "entity_track", "people_cities"]:
    P = prompts(t)
    cand_sets.update({pid: set(c) for pid, c in P.cand_ids.items()})
for exp, pp in frames.items():
    missing = ~pp.pair_id.isin(cand_sets.keys())
    if missing.any():                         # E10b fresh families: load their prompt sets lazily
        import glob, gzip, json
        from steering_common import PS
        for f in glob.glob(str(PS / "add_chain*.jsonl.gz")):
            for l in gzip.open(f, "rt"):
                r = json.loads(l)
                cand_sets.setdefault(r["pair_id"], set(r["cand_ids"]))
    pp["top_is_cand"] = [t in cand_sets.get(p, ()) for p, t in zip(pp.pair_id, pp.top1_id)]
    pp["outcome"] = np.where(pp.isA, "A", np.where(pp.isB, "B", np.where(pp.top_is_cand, "other_cand", "noncand")))
    pp.to_parquet(OUT / f"pp_{exp}.parquet")

GK = ["task", "layer", "family"]
rows = []
for exp, pp in frames.items():
    st = pp[pp.condition.str.startswith("steer_") | pp.condition.isin(["clean_B", "full_A"])]
    for flt in ["f_pilot", "f_none", "f_A_correct"]:
        sel = st[st[flt]]
        for key, g in sel.groupby(GK + ["condition", "estimator", "r", "seed"], sort=False):
            d = dict(zip(GK + ["condition", "estimator", "r", "seed"], key))
            d.update(exp=exp, filter=flt, s=g.s.iloc[0])
            d.update(summarise(g))
            e = g[g.elig]
            for k in ["A", "B", "other_cand", "noncand"]:
                d[f"out_{k}"] = float((e.outcome == k).mean()) if len(e) else np.nan
            rows.append(d)
R = pd.DataFrame(rows)
R.to_csv(OUT / "dose_rows.csv", index=False)

# ---- envelope + specificity with pair bootstrap
summ = []
for exp, pp in frames.items():
    st = pp[pp.condition.str.startswith("steer_")]
    for flt in ["f_pilot", "f_none"]:
        sel = st[st[flt] & st.elig]
        for (task, layer, family, cond, r), g in sel.groupby(GK + ["condition", "r"], sort=False):
            wide = g.pivot_table(index="pair_id", columns=["estimator", "seed"], values="isA", aggfunc="first").astype(float)
            rnd = [c for c in wide.columns if c[0] == "random"]
            ests = [c for c in wide.columns if c[0] != "random"]
            X = wide.to_numpy()
            n = len(X)
            idx = np.vstack([np.arange(n), rng.integers(0, n, (B, n))])
            M = np.stack([X[i].mean(0) for i in idx])               # [B+1, keys]
            ri = [wide.columns.get_loc(c) for c in rnd]
            p95 = np.nanpercentile(M[:, ri], 95, axis=1) if ri else np.full(B + 1, np.nan)
            rmean = M[:, ri].mean(1) if ri else np.full(B + 1, np.nan)
            for c in ests:
                j = wide.columns.get_loc(c)
                spec = M[:, j] - p95
                summ.append(dict(exp=exp, filter=flt, task=task, layer=layer, family=family, condition=cond,
                                 s=g.s.iloc[0], estimator=c[0], r=r, n_elig=n, flip=M[0, j],
                                 flip_lo=np.percentile(M[1:, j], 2.5), flip_hi=np.percentile(M[1:, j], 97.5),
                                 rnd_mean=rmean[0], rnd_p95=p95[0], rnd_max=M[0, ri].max() if ri else np.nan,
                                 rnd_n=len(ri), spec=spec[0], spec_lo=np.percentile(spec[1:], 2.5),
                                 spec_hi=np.percentile(spec[1:], 97.5),
                                 p_spec=(1 + np.sum(spec[1:] <= 0)) / (B + 1),
                                 ratio_to_rnd_mean=M[0, j] / rmean[0] if ri and rmean[0] > 0 else np.nan,
                                 q2_like=bool(M[0, j] >= 0.5 and p95[0] <= 0.05)))
S = pd.DataFrame(summ)
# attach nldr / mass columns for the estimator rows and the random mean of nldr
key = ["exp", "filter", "task", "layer", "family", "condition", "estimator", "r"]
Rm = R[R.estimator != "random"][key + ["nldr", "prob_rec", "pA", "pB", "p_other_cand", "p_noncand", "out_A", "out_B",
                                        "out_other_cand", "out_noncand"]]
S = S.merge(Rm, on=key, how="left")
rn = (R[R.estimator == "random"].groupby(["exp", "filter", "task", "layer", "family", "condition", "r"])
      .agg(rnd_nldr_mean=("nldr", "mean"), rnd_nldr_p95=("nldr", lambda s: np.nanpercentile(s, 95)),
           rnd_out_B=("out_B", "mean"), rnd_out_A=("out_A", "mean"), rnd_out_other=("out_other_cand", "mean"),
           rnd_out_noncand=("out_noncand", "mean")).reset_index())
S = S.merge(rn, on=["exp", "filter", "task", "layer", "family", "condition", "r"], how="left")
from steering_common import bh
S["q_spec_bh"] = np.nan
for (exp, flt), idx in S.groupby(["exp", "filter"]).groups.items():
    S.loc[idx, "q_spec_bh"] = bh(S.loc[idx, "p_spec"].to_numpy())
S.to_csv(OUT / "dose_summary.csv", index=False)
print(S.groupby(["exp", "filter"]).apply(lambda g: pd.Series(dict(n_tests=len(g), n_q05=int((g.q_spec_bh < 0.05).sum())))))
print("rows", len(R), "summary", len(S), "q2_like rows:", int(S.q2_like.sum()))
print(S[S.q2_like][["exp", "filter", "task", "s", "estimator", "r", "flip", "rnd_p95"]].to_string())

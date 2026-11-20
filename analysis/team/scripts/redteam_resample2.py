"""RED TEAM (exploratory, post hoc). Claim D follow-ups, running sum L18.

1. Donor identity: the harness selects donors over a cell's test pairs in (family, rank) order (harness/run.py run_stage3:
   te sorted by (family, rank); harness/model/stages.load_spans keeps that order). deep/scripts/mech_donor_pull*.py rebuilt
   donors over pairs sorted by pair_id (and pooled E3 cells). Compare the two reconstructions (share of identical donors).
   The stored donor_status cannot tell them apart (almost always 'ok').
2. Donor pull with the harness-order donors: P(top1 = donor's answer) under resampC_diffAns vs the same pairs under
   sameAns / meanPos / errB / zero / noise (paired bootstrap), i.e. mechanism.md section 5 recomputed.
3. Graded effect: NLDR under diffAns vs |value(C) - value(A)| (digit values via cand_ids order).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
sys.path.insert(0, str(SP / "hw"))
from harness.conditions import select_donors  # noqa: E402

rng = np.random.default_rng(0)
P = {}
for f in ("add_chain_pilot", "add_chain_e3"):
    for l in gzip.open(SP / "hw/data/prompt_sets" / f"{f}.jsonl.gz", "rt"):
        r = json.loads(l)
        P[r["pair_id"]] = r
CAND = P[next(iter(P))]["cand_ids"]
VAL = {t: k for k, t in enumerate(CAND)}          # token id -> digit value


def boot(x, n=2000):
    x = np.asarray(x, float)
    v = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def meta(rr):
    return {"T": np.array([r["T"] for r in rr]), "first_diff": np.array([r["first_diff"] for r in rr]),
            "ans_A": np.array([r["ans_A"] for r in rr]), "ans_B": np.array([r["ans_B"] for r in rr]),
            "family": np.array([r.get("family", "pilot") for r in rr], dtype=object)}


CMP = ["feat_A_resampC_sameAns", "feat_A_meanPos", "feat_A_errB", "feat_A_zero", "feat_A_noise", "feat_A_covnoise"]
pull, dist_rows, ident = [], [], []
for exp, run in [("E3", "7bf755d092fa"), ("E2", "5ad7ada7f971"), ("E2r", "747ea7736718")]:
    man = json.loads((SP / "ana_runs" / exp / run / "manifest.json").read_text())
    seed = int(man["config"].get("seed", 0))
    pp = pd.read_parquet(SP / "ana_runs" / exp / run / "per_pair.parquet",
                         columns=["pair_id", "task", "family", "fit_families", "condition", "estimator", "seed", "ans_A_id",
                                  "ans_B_id", "top1_id", "logit_A", "logit_B"])
    pp = pp[(pp.task == "add_chain") & (pp.estimator.fillna("") == "")]
    pp = pp.assign(evalgrp=np.where(pp.family.fillna("pilot").str.startswith("pilot"), "pilot", "fresh"))
    for (fit, evg), g in pp.groupby(["fit_families", "evalgrp"]):
        ids = list(g.pair_id.unique())
        rs = sorted([P[i] for i in ids], key=lambda r: (r.get("family", "pilot"), r["rank"]))
        m1 = meta(rs)
        dn = select_donors(m1, False, seed)
        # deep's reconstruction: pair_id order (and, for E3, fresh-fit cells pooled)
        rs2 = sorted([P[i] for i in ids], key=lambda r: r["pair_id"])
        dn2 = select_donors(meta(rs2), False, seed)
        k2 = {r["pair_id"]: k for k, r in enumerate(rs2)}
        key = lambda rr, d: (rr[d["j"]]["pair_id"], d["prompt"]) if d["j"] >= 0 else None
        same = np.mean([key(rs, d) == key(rs2, dn2[k2[r["pair_id"]]]) for r, d in zip(rs, dn)])
        ident.append({"exp": exp, "fit": fit, "eval": evg, "n_pairs": len(rs), "share_identical_diff_donors": same})
        ansC = {r["pair_id"]: (m1["ans_A"][d["j"]] if d["prompt"] == "A" else m1["ans_B"][d["j"]]) for r, d in zip(rs, dn) if d["j"] >= 0}
        gg = g.copy()
        gg["D"] = gg.logit_A - gg.logit_B
        a = gg.drop_duplicates("pair_id").set_index("pair_id")
        top = gg[gg.seed.fillna(-1) <= 0].drop_duplicates(["pair_id", "condition"]).pivot(index="pair_id", columns="condition", values="top1_id")
        D = gg[gg.seed.fillna(-1) <= 0].drop_duplicates(["pair_id", "condition"]).pivot(index="pair_id", columns="condition", values="D")
        el = [p for p in top.index if top.clean_A[p] == a.ans_A_id[p] and top.clean_B[p] == a.ans_B_id[p] and p in ansC]
        dC = pd.Series({p: ansC[p] for p in el})
        hit = {c: (top.loc[el, c] == dC).astype(float) for c in ["feat_A_resampC_diffAns"] + CMP if c in top}
        for c in CMP:
            if c not in hit:
                continue
            mm, lo, hi = boot(hit["feat_A_resampC_diffAns"] - hit[c])
            pull.append({"exp": exp, "fit": fit, "eval": evg, "n": len(el), "control": c,
                         "p_donorAns_diffAns": hit["feat_A_resampC_diffAns"].mean(), "p_donorAns_control": hit[c].mean(),
                         "diff": mm, "lo": lo, "hi": hi})
        den = (D.full_A - D.clean_B).loc[el].mean()
        dist = pd.Series({p: abs(VAL[ansC[p]] - VAL[a.ans_A_id[p]]) for p in el})
        nd = (D.loc[el, "feat_A_resampC_diffAns"] - D.loc[el, "clean_B"]) / den
        ns = (D.loc[el, "feat_A_resampC_sameAns"] - D.loc[el, "clean_B"]) / den
        for k, sub in nd.groupby(dist):
            dist_rows.append({"exp": exp, "fit": fit, "eval": evg, "abs_dist": int(k), "n": len(sub), "nldr_diffAns": sub.mean(),
                              "nldr_sameAns_same_pairs": ns.loc[sub.index].mean(),
                              **dict(zip(("same_minus_diff", "smd_lo", "smd_hi"), boot(ns.loc[sub.index] - sub)))})
        X = np.c_[np.ones(len(el)), dist.loc[el].values]
        b = np.linalg.lstsq(X, nd.values, rcond=None)[0]
        bs = [np.linalg.lstsq(X[i], nd.values[i], rcond=None)[0][1] for i in (rng.integers(0, len(el), len(el)) for _ in range(1000))]
        dist_rows.append({"exp": exp, "fit": fit, "eval": evg, "abs_dist": -1, "n": len(el), "nldr_diffAns": b[1],
                          "nldr_sameAns_same_pairs": np.nan, "slope_lo": np.percentile(bs, 2.5), "slope_hi": np.percentile(bs, 97.5)})
pd.set_option("display.width", 220)
for name, rows in (("identity", ident), ("pull", pull), ("distance", dist_rows)):
    T = pd.DataFrame(rows)
    T.to_csv(SP / f"team/scripts/out/redteam_resample2_{name}.csv", index=False)
    print(name)
    print(T.round(3).to_string())

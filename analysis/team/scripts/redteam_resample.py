"""RED TEAM (exploratory, post hoc). Attack on claim D: "running-sum error carries an answer-specific component"
(feat_A_resampC_sameAns beats feat_A_resampC_diffAns).

Hypothesis to test: same-answer donors share more TOKENS with pair i's clean prompt at the patched positions (same sum ->
similar operands), so the "answer-specific" component could be a token/lexical-context component.
Steps: rebuild each run's donors with the harness's own select_donors, using the harness's pair order (family, rank) and,
for comparison, the pair_id order used by deep/scripts/mech_donor_pull*.py; check them against the stored donor_status;
measure donor-vs-A token agreement on the patched span; regress the per-pair same-minus-diff effect on the overlap difference.
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

OUT = SP / "team" / "scripts" / "out"
rng = np.random.default_rng(0)


def prompts(name):
    rows = [json.loads(l) for l in gzip.open(SP / "hw" / "data" / "prompt_sets" / f"{name}.jsonl.gz", "rt")]
    for r in rows:
        for k in ("clean_ids", "corrupt_ids"):
            if isinstance(r[k], str):
                r[k] = json.loads(r[k])
    return {r["pair_id"]: r for r in rows}


def boot(x, n=2000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    v = x[idx].mean(1)
    return x.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


res = []
for exp in ["E3", "E2", "E2r"]:
    for d in sorted((SP / "ana_runs" / exp).glob("*/manifest.json")):
        man = json.loads(d.read_text())
        if man.get("status") != "complete":
            continue
        pp = pd.read_parquet(d.parent / "per_pair.parquet", columns=["pair_id", "task", "family", "fit_families", "condition",
                                                                      "estimator", "ans_A_id", "ans_B_id", "top1_id",
                                                                      "logit_A", "logit_B", "donor_status", "layer"])
        pp = pp[(pp.task == "add_chain") & (pp.estimator.fillna("") == "")]
        if pp.empty or not pp.condition.str.startswith("feat_A_resampC").any():
            continue
        seed = int(man["config"].get("seed", 0))
        P = {}
        for f in ("add_chain_pilot", "add_chain_e3"):
            P.update(prompts(f))
        pp = pp.assign(evalgrp=np.where(pp.family.fillna("pilot").str.startswith("pilot"), "pilot", "fresh"))
        for (layer, fit, evg), g in pp.groupby(["layer", "fit_families", "evalgrp"]):  # one harness cell = one donor pool
            ids = list(g.pair_id.unique())
            rows = [P[i] for i in ids]
            for order_name in ("harness_family_rank", "pair_id_sorted"):
                if order_name == "harness_family_rank":
                    rs = sorted(rows, key=lambda r: (r.get("family", "pilot"), r["rank"]))
                else:
                    rs = sorted(rows, key=lambda r: r["pair_id"])
                meta = {"T": np.array([r["T"] for r in rs]), "first_diff": np.array([r["first_diff"] for r in rs]),
                        "ans_A": np.array([r["ans_A"] for r in rs]), "ans_B": np.array([r["ans_B"] for r in rs]),
                        "family": np.array([r.get("family", "pilot") for r in rs], dtype=object)}
                dn = {k: select_donors(meta, k == "same", seed) for k in ("same", "diff")}
                stored = {k: g[g.condition == f"feat_A_resampC_{k}Ans"].drop_duplicates("pair_id").set_index("pair_id").donor_status
                          for k in ("same", "diff")}
                mism = {k: int(sum(stored[k].get(r["pair_id"]) != dn[k][i]["status"] for i, r in enumerate(rs))) for k in dn}
                ncls = {k: pd.Series([x["status"] for x in dn[k]]).value_counts().to_dict() for k in dn}
                print(exp, d.parent.name, layer, fit, evg, order_name, "status mismatches", mism, ncls)
                if order_name != "harness_family_rank":
                    continue
                # token agreement donor vs A on i's patched span, and on the full prompt
                ov = {}
                for k in dn:
                    o_span, o_full, o_num = [], [], []
                    for i, r in enumerate(rs):
                        x = dn[k][i]
                        if x["j"] < 0:
                            o_span.append(np.nan), o_full.append(np.nan), o_num.append(np.nan)
                            continue
                        c = rs[x["j"]]
                        toks = c["clean_ids"] if x["prompt"] == "A" else c["corrupt_ids"]
                        a = r["clean_ids"]
                        fd, T = r["first_diff"], r["T"]
                        o_span.append(float(np.mean([toks[t] == a[t] for t in range(fd, T)])))
                        o_full.append(float(np.mean([toks[t] == a[t] for t in range(T)])))
                        # digit tokens on the span (Gemma digits 235276..235285)
                        dig = [t for t in range(fd, T) if 235276 <= a[t] <= 235285]
                        o_num.append(float(np.mean([toks[t] == a[t] for t in dig])) if dig else np.nan)
                    ov[k] = pd.DataFrame({"span": o_span, "full": o_full, "digits": o_num}, index=[r["pair_id"] for r in rs])
                # outcomes (as deep/scripts/mech_resample.py)
                gg = g.drop_duplicates(["pair_id", "condition"]).assign(D=lambda z: z.logit_A - z.logit_B,
                                                                       isA=lambda z: z.top1_id == z.ans_A_id,
                                                                       isB=lambda z: z.top1_id == z.ans_B_id)
                D = gg.pivot(index="pair_id", columns="condition", values="D")
                A = gg.pivot(index="pair_id", columns="condition", values="isA")
                B = gg.pivot(index="pair_id", columns="condition", values="isB")
                el = D.index[(A.clean_A == 1) & (B.clean_B == 1)]
                ok = [p for p in el if stored["same"].get(p) == "ok" and stored["diff"].get(p) == "ok"]
                den = (D.full_A - D.clean_B).loc[ok].mean()
                dnm = ((D.loc[ok, "feat_A_resampC_sameAns"] - D.loc[ok, "feat_A_resampC_diffAns"]) / den)
                dfl = A.loc[ok, "feat_A_resampC_sameAns"].astype(float) - A.loc[ok, "feat_A_resampC_diffAns"].astype(float)
                dov = (ov["same"].loc[ok] - ov["diff"].loc[ok])
                m, lo, hi = boot(dnm)
                row = {"exp": exp, "run": d.parent.name, "layer": layer, "fit": fit, "eval": evg, "n": len(ok), "dnldr": m, "dnldr_lo": lo,
                       "dnldr_hi": hi, "dflip": dfl.mean(),
                       "ov_span_same": ov["same"].loc[ok, "span"].mean(), "ov_span_diff": ov["diff"].loc[ok, "span"].mean(),
                       "ov_digit_same": ov["same"].loc[ok, "digits"].mean(), "ov_digit_diff": ov["diff"].loc[ok, "digits"].mean(),
                       "ov_full_same": ov["same"].loc[ok, "full"].mean(), "ov_full_diff": ov["diff"].loc[ok, "full"].mean()}
                # regression dnm ~ 1 + d_overlap(span) ; intercept = same-vs-diff effect at equal span overlap
                for key in ("span", "digits"):
                    xx = dov[key].to_numpy()
                    mk = np.isfinite(xx)
                    X = np.c_[np.ones(mk.sum()), xx[mk]]
                    y = dnm.to_numpy()[mk]
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    bs = []
                    for _ in range(1000):
                        ii = rng.integers(0, mk.sum(), mk.sum())
                        bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
                    bs = np.array(bs)
                    row.update({f"icpt_{key}": beta[0], f"icpt_{key}_lo": np.percentile(bs[:, 0], 2.5),
                                f"icpt_{key}_hi": np.percentile(bs[:, 0], 97.5), f"slope_{key}": beta[1],
                                f"slope_{key}_lo": np.percentile(bs[:, 1], 2.5), f"slope_{key}_hi": np.percentile(bs[:, 1], 97.5)})
                # stratum with equal digit overlap
                eq = dov["digits"].abs() < 1e-9
                if eq.sum() > 20:
                    m2, l2, h2 = boot(dnm[eq])
                    row.update({"n_equal_digit_overlap": int(eq.sum()), "dnldr_equal_ov": m2, "dnldr_equal_ov_lo": l2,
                                "dnldr_equal_ov_hi": h2})
                res.append(row)
T_ = pd.DataFrame(res)
T_.to_csv(OUT / "redteam_resample.csv", index=False)
pd.set_option("display.width", 250)
print(T_.round(3).T.to_string())

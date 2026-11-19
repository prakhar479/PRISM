"""EXPLORATORY, post hoc (2026-11-19). E22: does a DAS / PCA subspace fitted on English transfer to French / Hindi / Hinglish?
RS L18 (L18_16k_l074) and PC L12 (L12_16k_l082), test split, run $SP/ana_runs/E22/1c85291e335f.

Part 1 (behaviour): per evaluation family x filter x condition (full_along_U, feat_A_along_U) x estimator (das4, das8, pca_pooled
r=4/8): flip rate and NLDR (harness.e8.metric_samples definitions: flip among pairs whose clean-B top-1 is B's answer; NLDR as
ratio of means) with the subspace fitted natively vs on English, on the SAME pairs; paired bootstrap (2000 pair resamples) of
English-fit minus native-fit. Random-U null: p95 over the random seeds (identical U for both fits). E9 effectiveness rule
(flip >= p95 + 0.10 and lower CI > p95); DAS fit validity = train interchange acc >= 0.5 (json meta).
Part 2 (geometry): principal cosines between the English-fit and native-fit subspaces (pca_pooled first r columns = "error
directions"; DAS raw and with the shared initial rotation projected out = learned component, make_final.das_learned), vs Haar p99.
Part 3: E13 behavioural gates per family (hw/tables/final/E13_behaviour_lang.csv, test split).
BH over the paired transfer-minus-native tests (f_A_correct, both conditions, 4 estimators, 6 non-English families = 48 tests).
Outputs: $SP/team/multilingual_out/E22_*.csv
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path("/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad")
sys.path.insert(0, str(SP / "hw"))
sys.path.insert(0, str(SP / "hw" / "scripts"))
from harness import e8 as E  # noqa: E402
from harness.directions import principal_cosines  # noqa: E402

RUN = SP / "ana_runs" / "E22" / "1c85291e335f"
OUT = SP / "team" / "multilingual_out"
B = 2000
HAAR = {4: 0.097, 8: 0.131}
COLS = ["pair_id", "family", "fit_families", "condition", "estimator", "r", "seed", "ans_A_id", "ans_B_id", "f_none",
        "f_A_correct", "logit_A", "logit_B", "lse", "top1_id"]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def pval(bs, null=0.0):
    bs = bs[np.isfinite(bs)]
    if not len(bs):
        return np.nan
    if np.all(bs == null):
        return 1.0
    return max(2 * min(np.mean(bs <= null), np.mean(bs >= null)), 1 / B)


def acc_meta():
    acc = {}
    for j in (RUN / "directions").glob("*.json"):
        fit = j.stem.split("__fit-")[1]
        d = json.loads(j.read_text())
        for est in ("das4", "das8"):
            acc[(fit, est)] = d[est]["meta"]["train_interchange_acc"]
    return acc


def behaviour(pp):
    acc = acc_meta()
    rows = []
    rng_seed = 0
    for fam in sorted(pp.family.unique()):
        lang_fit = fam
        eng_fit = fam.split("_")[0] + "_en"
        for flt in ("f_A_correct", "f_none"):
            g = pp[(pp.family == fam) & pp[flt]]
            fr = {f: g[g.fit_families == f] for f in {lang_fit, eng_fit}}
            base = fr[lang_fit]
            ca = base[base.condition == "clean_A"].set_index("pair_id").sort_index()
            cb = base[base.condition == "clean_B"].set_index("pair_id").loc[ca.index]
            pairs = ca.index
            n = len(pairs)
            ansA, ansB = ca.ans_A_id.to_numpy(), ca.ans_B_id.to_numpy()
            A, Bc = E.Arr(ca, ansA), E.Arr(cb, ansA)
            elig = (cb.top1_id.to_numpy() == ansB).astype(float)
            rs = np.random.default_rng(rng_seed)
            idx = [np.arange(n)] + [rs.integers(0, n, n) for _ in range(B)]
            W = E.weights(idx, n)

            def met(f):
                f = f.set_index("pair_id").reindex(pairs)
                assert not f.top1_id.isna().any()
                return E.metric_samples(E.Arr(f, ansA), A, Bc, elig, W)
            for cond in ("full_along_U", "feat_A_along_U"):
                # random null (U independent of fit; use the native-fit rows)
                rnd = base[(base.condition == cond) & (base.estimator == "random")]
                null = {r: [met(x)["flip_rate"][0] for _, x in rnd[rnd.r == r].groupby("seed")] for r in (4, 8)}
                for est, r in (("das4", 4), ("das8", 8), ("pca_pooled", 4), ("pca_pooled", 8)):
                    res = {}
                    for src, fit in (("native", lang_fit), ("english", eng_fit)):
                        x = fr[fit]
                        x = x[(x.condition == cond) & (x.estimator == est) & (x.r == r)]
                        res[src] = met(x)
                    p95 = float(np.percentile(null[r], 95))
                    row = {"family": fam, "filter": flt, "condition": cond, "estimator": est, "r": r, "n_pairs": n,
                           "n_eligible": int(elig.sum()), "random_p95": p95, "random_n": len(null[r])}
                    for src in ("native", "english"):
                        for m in ("flip_rate", "nldr"):
                            v = res[src][m]
                            row[f"{m}_{src}"] = v[0]
                            row[f"{m}_{src}_lo"], row[f"{m}_{src}_hi"] = np.nanpercentile(v[1:], [2.5, 97.5])
                        row[f"effective_{src}"] = bool(row[f"flip_rate_{src}"] >= p95 + 0.10 and row[f"flip_rate_{src}_lo"] > p95)
                        fit = lang_fit if src == "native" else eng_fit
                        row[f"das_acc_{src}"] = acc.get((fit, est), np.nan)
                    if fam.endswith("_en"):
                        for k in [c for c in row if c.endswith("_english") or "_english_" in c]:
                            row[k] = np.nan
                    else:
                        for m in ("flip_rate", "nldr"):
                            d = res["english"][m] - res["native"][m]
                            row[f"{m}_eng_minus_nat"] = d[0]
                            row[f"{m}_eng_minus_nat_lo"], row[f"{m}_eng_minus_nat_hi"] = np.nanpercentile(d[1:], [2.5, 97.5])
                            row[f"{m}_eng_minus_nat_p"] = pval(d[1:])
                    rows.append(row)
    T = pd.DataFrame(rows)
    m = (T["filter"] == "f_A_correct") & T.flip_rate_eng_minus_nat_p.notna()
    T.loc[m, "flip_eng_minus_nat_q"] = bh(T.loc[m, "flip_rate_eng_minus_nat_p"])
    T.loc[m, "nldr_eng_minus_nat_q"] = bh(T.loc[m, "nldr_eng_minus_nat_p"])
    return T


def geometry():
    from make_final import das_learned
    rows = []
    for task, L, sae, pre in (("add_chain", 18, "L18_16k_l074", "rs"), ("people_cities", 12, "L12_16k_l082", "pc")):
        Z = {f: np.load(RUN / "directions" / f"{task}_L{L}_{sae}__fit-{pre}_{f}.npz") for f in ("en", "fr", "hi", "hinglish")}
        langs = list(Z)
        for i, a in enumerate(langs):
            for b in langs[i + 1:]:
                for est, r in (("pca_pooled", 4), ("pca_pooled", 8), ("das4", 4), ("das8", 8)):
                    Ua, Ub = Z[a][est][:, :r], Z[b][est][:, :r]
                    c = principal_cosines(Ua, Ub)
                    row = {"task": pre, "a": a, "b": b, "estimator": est, "r": r, "cos_top": c[0], "cos_mean": c.mean(),
                           "haar_p99": HAAR[r]}
                    if est.startswith("das"):
                        (la, ca), (lb, cb) = das_learned(Ua, est, 0), das_learned(Ub, est, 0)
                        cl = principal_cosines(la, lb)
                        row.update({"cos_top_learned": cl[0], "cos_mean_learned": cl.mean(), "init_capture_a": ca,
                                    "init_capture_b": cb})
                    if est == "pca_pooled":
                        row["cos_top1_vec"] = abs(float(Ua[:, 0] @ Ub[:, 0]))       # rank-1 error direction
                    rows.append(row)
        # DAS (learned) vs own-language PCA r=8 (how much of the causal subspace lies in the error PCA subspace)
        for f in langs:
            for est in ("das4", "das8"):
                lrn, cap = das_learned(Z[f][est], est, 0)
                P = np.linalg.qr(Z[f]["pca_pooled"][:, :8])[0]
                rows.append({"task": pre, "a": f, "b": f, "estimator": f"{est}_learned_in_pca8",
                             "r": lrn.shape[1], "cos_top": float(principal_cosines(lrn, P)[0]),
                             "energy_in_pca8": float(np.sum((P.T @ lrn) ** 2) / lrn.shape[1]), "init_capture_a": cap})
    # crude null for learned-component cosines: RS L18 fit vs PC L12 fit (different task AND layer; same init rotation)
    for f in ("en", "fr", "hi", "hinglish"):
        za = np.load(RUN / "directions" / f"add_chain_L18_L18_16k_l074__fit-rs_{f}.npz")
        zb = np.load(RUN / "directions" / f"people_cities_L12_L12_16k_l082__fit-pc_{f}.npz")
        for est, r in (("das4", 4), ("das8", 8)):
            c = principal_cosines(za[est], zb[est])
            (la, ca), (lb, cb) = das_learned(za[est], est, 0), das_learned(zb[est], est, 0)
            cl = principal_cosines(la, lb)
            rows.append({"task": "rsL18_vs_pcL12", "a": f, "b": f, "estimator": est, "r": r, "cos_top": c[0],
                         "cos_mean": c.mean(), "haar_p99": HAAR[r], "cos_top_learned": cl[0], "cos_mean_learned": cl.mean(),
                         "init_capture_a": ca, "init_capture_b": cb})
    return pd.DataFrame(rows)


def main():
    pp = pd.read_parquet(RUN / "per_pair.parquet", columns=COLS)
    T = behaviour(pp)
    T.to_csv(OUT / "E22_transfer_paired.csv", index=False)
    G = geometry()
    G.to_csv(OUT / "E22_geometry.csv", index=False)
    e13 = pd.read_csv(SP / "hw" / "tables" / "final" / "E13_behaviour_lang.csv")
    e13 = e13[(e13.split == "test") & e13.task.isin(["add_chain", "people_cities"])][
        ["family", "acc_clean", "acc_clean_lo", "acc_clean_hi", "keep_pilot_rule", "median_D_clean", "gate_A_viable",
         "gate_B_within_10pts"]]
    e13.to_csv(OUT / "E22_e13_gates.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 50, "display.max_rows", 300)
    t = T[T["filter"] == "f_A_correct"]
    print(t[["family", "condition", "estimator", "r", "n_eligible", "random_p95", "das_acc_native", "das_acc_english",
             "flip_rate_native", "flip_rate_native_lo", "effective_native", "flip_rate_english", "flip_rate_english_lo",
             "effective_english", "flip_rate_eng_minus_nat", "flip_rate_eng_minus_nat_lo", "flip_rate_eng_minus_nat_hi",
             "flip_eng_minus_nat_q", "nldr_native", "nldr_english", "nldr_eng_minus_nat_lo", "nldr_eng_minus_nat_hi"]
            ].round(3).to_string(index=False))
    print(G.round(3).to_string(index=False))
    print(e13.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

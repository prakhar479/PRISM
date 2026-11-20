"""E18 per-pair response profiles: SVD modes, split-half stability, clustering, prompt-feature associations.
Exploratory/post hoc. Output: patterns_out/profiles_*.csv and printed summary."""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans2

import patterns_common as C

rng = np.random.default_rng(0)
d = pd.read_parquet(C.OUT / "e18_long.parquet") if (C.OUT / "e18_long.parquet").exists() else C.add_refs(C.load("E18"))
d = d[d.A_ok & d.B_ok]
d["key"] = C.ckey(d)
rand = d[d.estimator == "random"].groupby(["task", "layer", "pair_id", "condition", "r"]).nldr.mean().reset_index()
rand["key"] = rand.condition + "|random_mean|r" + rand.r.astype(str)
nr = d[(d.estimator != "random") & ~d.condition.isin(["clean_A", "clean_B", "full_A", "full_B"])]
allp = pd.concat([nr[["task", "layer", "pair_id", "key", "nldr"]], rand[["task", "layer", "pair_id", "key", "nldr"]]])
den = d[d.condition == "clean_A"].drop_duplicates(["task", "layer", "pair_id"]).set_index(["task", "layer", "pair_id"])

rows, assoc = [], []
for (t, L), g in allp.groupby(["task", "layer"]):
    M = g.pivot_table(index="pair_id", columns="key", values="nldr")
    M = M.dropna(axis=1, thresh=int(0.9 * len(M))).dropna()
    X = M.clip(-1, 2).values
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ve = S ** 2 / (S ** 2).sum()
    # split-half stability of the first 3 right singular vectors (20 random splits)
    cos = []
    for s in range(20):
        idx = rng.permutation(len(X)); h1, h2 = idx[: len(X) // 2], idx[len(X) // 2:]
        V1 = np.linalg.svd(X[h1] - X[h1].mean(0), full_matrices=False)[2]
        V2 = np.linalg.svd(X[h2] - X[h2].mean(0), full_matrices=False)[2]
        cos.append([abs(V1[k] @ V2[k]) for k in range(3)])
    cos = np.array(cos)
    # random-sign null for the split-half cosine: column-permute within halves breaks pair structure
    nullc = []
    for s in range(20):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        idx = rng.permutation(len(X)); h1, h2 = idx[: len(X) // 2], idx[len(X) // 2:]
        V1 = np.linalg.svd(Xp[h1] - Xp[h1].mean(0), full_matrices=False)[2]
        V2 = np.linalg.svd(Xp[h2] - Xp[h2].mean(0), full_matrices=False)[2]
        nullc.append(abs(V1[0] @ V2[0]))
    # permutation null for variance explained by PC1
    nullve = []
    for s in range(50):
        Xp = np.column_stack([rng.permutation(Xc[:, j]) for j in range(X.shape[1])])
        sp = np.linalg.svd(Xp, compute_uv=False); nullve.append(sp[0] ** 2 / (sp ** 2).sum())
    rows.append(dict(task=t, layer=L, n_pairs=len(X), n_conds=X.shape[1], ve1=ve[0], ve2=ve[1], ve3=ve[2],
                     ve1_null_p95=np.percentile(nullve, 95),
                     splitcos1=np.median(cos[:, 0]), splitcos2=np.median(cos[:, 1]), splitcos3=np.median(cos[:, 2]),
                     splitcos1_null_med=np.median(nullc)))
    # loadings of modes
    load = pd.DataFrame({"key": M.columns, "mean": X.mean(0), "v1": Vt[0], "v2": Vt[1], "v3": Vt[2]})
    load.to_csv(C.OUT / f"profiles_loadings_{t}_L{L}.csv", index=False)
    sc = pd.DataFrame({"pair_id": M.index, "pc1": U[:, 0] * S[0], "pc2": U[:, 1] * S[1], "pc3": U[:, 2] * S[2],
                       "mean_resp": X.mean(1)})
    # pair-level covariates
    dd = den.loc[(t, L)].reindex(M.index)
    sc["den"] = dd.den.values; sc["D_A"] = dd.D_A.values; sc["D_B"] = dd.D_B.values
    sn = C.set_name_for(C.run_dirs("E18")[0], t)
    F = C.features(t, list(M.index), sn)
    sc = sc.merge(F, on="pair_id")
    sc.to_csv(C.OUT / f"profiles_scores_{t}_L{L}.csv", index=False)
    print(f"\n=== {t} L{L}: n={len(X)} conds={X.shape[1]} VE={np.round(ve[:4],3)} null95={np.percentile(nullve,95):.3f} "
          f"splitcos={np.round(np.median(cos,0),3)} null={np.median(nullc):.3f}")
    for k in ["v1", "v2"]:
        o = load.reindex(load[k].abs().sort_values(ascending=False).index).head(8)
        print(k, "top loadings:", "; ".join(f"{a}:{b:+.2f}" for a, b in zip(o.key, o[k])))
    # associations: spearman of pc1/pc2 with numeric covariates; kruskal for categorical
    num = [c for c in ["den", "D_A", "D_B", "n_patched", "corrupt_step", "start", "inc_A", "inc_B", "ans_A", "ans_B", "delta",
                       "q_moved_in_A", "q_init_pos", "q_pos"] if c in sc and sc[c].nunique() > 1]
    for pc in ["pc1", "pc2", "mean_resp"]:
        for c in num:
            r, p = stats.spearmanr(sc[pc], sc[c])
            # split-half: sign agreement in both halves
            idx = rng.permutation(len(sc)); h1, h2 = idx[: len(sc) // 2], idx[len(sc) // 2:]
            r1 = stats.spearmanr(sc[pc].values[h1], sc[c].values[h1])[0]; r2 = stats.spearmanr(sc[pc].values[h2], sc[c].values[h2])[0]
            assoc.append(dict(task=t, layer=L, score=pc, covariate=c, rho=r, p=p, rho_h1=r1, rho_h2=r2))
        for c in ["template", "ans_A_tok", "ans_B_tok", "query", "A_is_init", "cityA", "cityB"]:
            if c in sc and sc[c].nunique() > 1:
                grp = [v.values for _, v in sc.groupby(c)[pc] if len(v) >= 3]
                if len(grp) > 1:
                    h, p = stats.kruskal(*grp)
                    assoc.append(dict(task=t, layer=L, score=pc, covariate=c, rho=np.nan, p=p, rho_h1=np.nan, rho_h2=np.nan))
    # k-means on standardised profiles, k=2..4 ; report silhouette-like ratio and cluster stability between halves
    Z = Xc / (Xc.std(0) + 1e-9)
    for k in [2, 3]:
        best = None
        for s in range(10):
            cen, lab = kmeans2(Z, k, seed=int(s), minit="++")
            w = sum(((Z[lab == j] - cen[j]) ** 2).sum() for j in range(k))
            if best is None or w < best[0]:
                best = (w, cen, lab)
        lab = best[2]
        sc[f"km{k}"] = lab
        desc = sc.groupby(f"km{k}").agg(n=("pair_id", "size"), mean_resp=("mean_resp", "mean"), den=("den", "median"),
                                         n_patched=("n_patched", "median"), corrupt_step=("corrupt_step", "mean"))
        print(f"kmeans k={k}:\n", desc.round(3).to_string())
    sc.to_csv(C.OUT / f"profiles_scores_{t}_L{L}.csv", index=False)

R = pd.DataFrame(rows); R.to_csv(C.OUT / "profiles_svd.csv", index=False)
A = pd.DataFrame(assoc); A["q_bh"] = C.bh(A.p.fillna(1).values); A.to_csv(C.OUT / "profiles_assoc.csv", index=False)
print(R.round(3).to_string())
print(f"\n{len(A)} association tests (BH over all):")
print(A[A.q_bh < 0.01].sort_values("p").round(4).head(40).to_string())

# cost vs patchability: per-pair feature-only NLDR (feat_A_errB) vs DAS4 full_along_U NLDR, Spearman with split halves
print("\ncost (feat_A_errB) vs patchability (full_along_U das4):")
for (t, L), g in allp.groupby(["task", "layer"]):
    M = g.pivot_table(index="pair_id", columns="key", values="nldr")
    a, b = M["feat_A_errB||r-1"], M["full_along_U|das4|r4"]
    ok = a.notna() & b.notna(); a, b = a[ok], b[ok]
    idx = rng.permutation(len(a)); h = idx[: len(a) // 2]
    r, p = stats.spearmanr(a, b)
    print(t, L, "n", len(a), "rho", round(r, 3), "p", f"{p:.2g}", "halves", round(stats.spearmanr(a.iloc[h], b.iloc[h])[0], 3),
          round(stats.spearmanr(a.drop(a.index[h]), b.drop(b.index[h]))[0], 3))

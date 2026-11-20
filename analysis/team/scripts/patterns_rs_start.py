"""RS L18 (E18): does the start number predict per-pair responses beyond the clean margin (den) and template?
OLS with HC-free plain SEs + split-half replication. Exploratory/post hoc."""
import numpy as np, pandas as pd
from scipy import stats
import patterns_common as C

d = pd.read_parquet(C.OUT / "e18_long.parquet")
d = d[d.A_ok & d.B_ok & (d.task == "add_chain")]
d["key"] = C.ckey(d)
keys = ["feat_A_errB||r-1", "feat_A_meanPos||r-1", "full_along_U|das1|r1", "full_along_U|das2|r2", "full_along_U|das4|r4",
        "feat_A_along_U|das4|r4", "full_along_U|pca_pooled|r8", "full_along_U|answer_sep|r8"]
W = d[d.key.isin(keys)].pivot_table(index="pair_id", columns="key", values="nldr")
S = pd.read_csv(C.OUT / "profiles_scores_add_chain_L18.csv").set_index("pair_id")
W = W.join(S[["start", "den", "template", "corrupt_step", "ans_A", "ans_B", "delta", "pc1"]])
W["logden"] = np.log(W.den); W["calc"] = (W.template == "calc").astype(float)
rng = np.random.default_rng(3)
rows = []
for k in keys + ["pc1"]:
    for half in ["all", "h1", "h2"]:
        Z = W.dropna(subset=[k])
        if half != "all":
            idx = rng.permutation(len(Z)); Z = Z.iloc[idx[: len(Z) // 2]] if half == "h1" else Z.iloc[idx[len(Z) // 2:]]
        X = np.column_stack([np.ones(len(Z)), Z.logden, Z.calc, Z.corrupt_step, Z.start])
        y = Z[k].clip(-1, 3).values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b; s2 = res @ res / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        t = b / se; p = 2 * stats.t.sf(abs(t), len(y) - X.shape[1])
        rows.append(dict(key=k, half=half, n=len(y), b_start=b[4], se_start=se[4], p_start=p[4], b_logden=b[1], p_logden=p[1]))
R = pd.DataFrame(rows); R.to_csv(C.OUT / "rs_start_regression.csv", index=False)
print(R.round(4).to_string())
print(W.groupby("start")[keys].mean().round(3).assign(n=W.groupby("start").size()).to_string())

# robustness: add ans_A, ans_B, |delta| and corrupted-increment size as covariates
print("\nwith ans_A, ans_B, |delta|, inc_A, inc_B added:")
F = C.features("add_chain", list(W.index), "add_chain_pilot").set_index("pair_id")
W = W.join(F[["inc_A", "inc_B"]])
for k in ["full_along_U|das4|r4", "full_along_U|pca_pooled|r8", "feat_A_errB||r-1", "pc1"]:
    Z = W.dropna(subset=[k])
    X = np.column_stack([np.ones(len(Z)), Z.logden, Z.calc, Z.corrupt_step, Z.ans_A, Z.ans_B, Z.delta.abs(), Z.inc_A, Z.inc_B, Z.start])
    y = Z[k].clip(-1, 3).values
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; s2 = res @ res / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); p = 2 * stats.t.sf(abs(b / se), len(y) - X.shape[1])
    print(f"{k}: b_start={b[-1]:.4f} se={se[-1]:.4f} p={p[-1]:.2g}  n={len(y)}")

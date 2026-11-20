"""Task 2: where does per-pair variation in task-state gaps come from (pair/family, condition, layer, seed, machine)? And
between-run reliability of per-pair outcomes. EXPLORATORY. Outputs team/stats_out/var_*.csv + printed summary.

Per-pair outcomes: nldr_i = (D_p - D_B)/(D_A - D_B) (winsorized to [-1, 2] for the variance decompositions; D = logit A - logit B),
flip_i = top1 == A's answer on pairs whose clean B run predicts B (eligible). Test split; RS = add_chain L18 16k, OT = entity_track
L12 16k, pilot families.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".layout/team/scripts")
from stats_common import OUT, RUNS  # noqa: E402
from stats_pp import host, load_cell, run_dirs  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
CELLS = {"RS": ("add_chain", 18, "L18_16k_l074"), "OT": ("entity_track", 12, "L12_16k_l082")}
EXPS = ["E2", "E2r", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E17", "E18", "E19"]


def wins(x):
    return np.clip(x, -1, 2)


def seq_anova(df, y, terms):
    """Sequential (type I) SS shares. terms: list of column-name tuples; each tuple = interaction of those columns."""
    yv = df[y].to_numpy(float)
    sst = ((yv - yv.mean()) ** 2).sum()
    X = np.ones((len(df), 1))
    prev = 0.0
    out = []
    for t in terms:
        lab = df[list(t)].astype(str).agg("|".join, axis=1)
        D = pd.get_dummies(lab, drop_first=True).to_numpy(float)
        X = np.c_[X, D]
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
        ss = ((X @ b - yv.mean()) ** 2).sum()
        out.append({"term": ":".join(t), "share": (ss - prev) / sst, "df": D.shape[1]})
        prev = ss
    out.append({"term": "residual", "share": 1 - prev / sst, "df": len(df) - X.shape[1]})
    return pd.DataFrame(out)


def icc1(M):
    """M [n_subjects, k_raters] complete: one-way random ICC(1,1)."""
    n, k = M.shape
    gm = M.mean()
    msb = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((M - M.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if msb + (k - 1) * msw > 0 else np.nan


def fleiss_kappa_binary(M):
    """M [n, k] in {0,1}."""
    n, k = M.shape
    p1 = M.mean()
    P_i = (M.sum(1) * (M.sum(1) - 1) + (k - M.sum(1)) * (k - M.sum(1) - 1)) / (k * (k - 1))
    Pe = p1 ** 2 + (1 - p1) ** 2
    return (P_i.mean() - Pe) / (1 - Pe) if Pe < 1 else np.nan


# ------------------------------------------------------------------ load all runs for the two confirmatory cells
data = {c: {} for c in CELLS}
for e in EXPS:
    for d in run_dirs(e):
        for c, (task, L, sae) in CELLS.items():
            t = load_cell(d, task, L, sae, fit="pilot", family="pilot")
            if t.pair_id.nunique() >= 100:
                tag = f"{e}:{d.rstrip('/').split('/')[-1][:6]}"
                t["run"], t["host"] = tag, host(d)
                data[c][tag] = t
for c in CELLS:
    print(c, {k: (v.host.iloc[0], v.pair_id.nunique()) for k, v in data[c].items()})

# ------------------------------------------------------------------ A. cross-run reproducibility of baseline conditions
print("\n[A] cross-run reproducibility (same pairs, same condition, different runs / machines)")
A = []
for c in CELLS:
    for cond in ("clean_B", "full_A", "feat_A_errB", "feat_A_meanPos", "feat_A_zero"):
        frames = {k: v[(v.condition == cond) & (v.estimator == "")].set_index("pair_id") for k, v in data[c].items()}
        frames = {k: v for k, v in frames.items() if len(v)}
        if len(frames) < 2:
            continue
        common = sorted(set.intersection(*[set(v.index) for v in frames.values()]))
        Dm = np.c_[[frames[k].loc[common, "D"].to_numpy() for k in frames]].T
        Nm = np.c_[[wins(frames[k].loc[common, "nldr_i"].to_numpy()) for k in frames]].T
        el = np.c_[[frames[k].loc[common, "elig"].to_numpy() for k in frames]].T
        elig_all = el.min(1) > 0
        Hm = np.c_[[frames[k].loc[common, "hitA"].to_numpy() for k in frames]].T[elig_all]
        hosts = [frames[k].host.iloc[0] for k in frames]
        fr = Hm.mean(0)
        A.append({"cell": c, "condition": cond, "n_runs": len(frames), "hosts": len(set(hosts)), "n_pairs": len(common),
                  "runs": ";".join(frames), "max_abs_dD_any": float(np.abs(Dm - Dm[:, :1]).max()),
                  "mean_sd_D_across_runs": float(Dm.std(1, ddof=1).mean()), "sd_D_across_pairs": float(Dm[:, 0].std(ddof=1)),
                  "icc_D": icc1(Dm), "icc_nldr_i": icc1(Nm) if cond not in ("clean_B", "full_A") else np.nan,
                  "n_elig_all": int(elig_all.sum()), "elig_disagree": int(((el.min(1) > 0) != (el.max(1) > 0)).sum()),
                  "flip_unanimous_share": float(np.mean((Hm.min(1) == Hm.max(1)))) if len(Hm) else np.nan,
                  "fleiss_kappa_flip": fleiss_kappa_binary(Hm) if len(Hm) and 0 < Hm.mean() < 1 else np.nan,
                  "flip_rate_min": float(fr.min()), "flip_rate_max": float(fr.max()),
                  "nldr_rom_min": float(((Dm.mean(0) - np.c_[[frames[k].loc[common, 'DB'] for k in frames]].T.mean(0)) /
                                         (np.c_[[frames[k].loc[common, 'DA'] for k in frames]].T.mean(0) -
                                          np.c_[[frames[k].loc[common, 'DB'] for k in frames]].T.mean(0))).min()),
                  "nldr_rom_max": float(((Dm.mean(0) - np.c_[[frames[k].loc[common, 'DB'] for k in frames]].T.mean(0)) /
                                         (np.c_[[frames[k].loc[common, 'DA'] for k in frames]].T.mean(0) -
                                          np.c_[[frames[k].loc[common, 'DB'] for k in frames]].T.mean(0))).max())})
        # bitwise-identical run groups
        grp = {}
        for j, k in enumerate(frames):
            key = None
            for g0 in grp:
                if np.array_equal(Dm[:, j], Dm[:, list(frames).index(g0)]):
                    key = g0
                    break
            grp.setdefault(key or k, []).append(f"{k}@{hosts[j]}")
        A[-1]["identical_groups"] = " || ".join(",".join(v) for v in grp.values())
A = pd.DataFrame(A)
A.to_csv(f"{OUT}/var_crossrun.csv", index=False)
print(A.drop(columns=["runs", "identical_groups"]).round(4).to_string(index=False))
for _, r in A.iterrows():
    print(r.cell, r.condition, "identical groups:", r.identical_groups)

# ------------------------------------------------------------------ B. seeds: DAS (E17) and random U (E9)
print("\n[B] seed variation: E17 DAS seeds (RS L18) — per-pair agreement and decomposition pair x seed")
B = []
e17 = [v for k, v in data["RS"].items() if k.startswith("E17")][0]
fams = {1: [("das1", "das1_s1", "das1_s2"), ("das1_e10", "das1_e10_s1", "das1_e10_s2")],
        2: [("das2", "das2_s1", "das2_s2"), ("das2_e10", "das2_e10_s1", "das2_e10_s2")],
        4: [("das4", "das4_s1")], 8: [("das8", "das8_s1")]}
for r, groups in fams.items():
    for ests in groups:
        for cond in ("full_along_U", "feat_A_along_U", "repair"):
            sub = e17[(e17.condition == cond) & e17.estimator.isin(ests)]
            if sub.estimator.nunique() < 2:
                continue
            piv_h = sub.pivot_table(index="pair_id", columns="estimator", values="hitA")
            piv_n = sub.pivot_table(index="pair_id", columns="estimator", values="nldr_i").clip(-1, 2)
            el = sub.groupby("pair_id").elig.first().reindex(piv_h.index) > 0
            H = piv_h[el].to_numpy()
            fr = H.mean(0)
            nb = 2000
            rng = np.random.default_rng(0)
            se = np.std([H[rng.integers(0, len(H), len(H)), 0].mean() for _ in range(nb)])
            long = sub[sub.elig > 0][["pair_id", "estimator", "hitA", "family"]]
            sa = seq_anova(long, "hitA", [("estimator",), ("pair_id",)])
            longn = sub[["pair_id", "estimator", "nldr_i"]].assign(nldr_i=lambda x: wins(x.nldr_i))
            san = seq_anova(longn, "nldr_i", [("estimator",), ("pair_id",)])
            B.append({"rank": r, "estimators": "/".join(ests), "condition": cond, "n_elig": int(el.sum()),
                      "flip_rates": "/".join(f"{x:.3f}" for x in fr), "flip_range": float(fr.max() - fr.min()),
                      "sd_flip_across_seeds": float(fr.std(ddof=1)), "boot_se_flip_one_seed": float(se),
                      "seed_sd_over_sampling_se": float(fr.std(ddof=1) / se) if se > 0 else np.nan,
                      "flip_unanimous_share": float(np.mean(H.min(1) == H.max(1))),
                      "fleiss_kappa_flip": fleiss_kappa_binary(H) if 0 < H.mean() < 1 else np.nan,
                      "icc_nldr_i": icc1(piv_n.to_numpy()),
                      "share_seed_flip": float(sa.share[0]), "share_pair_flip": float(sa.share[1]),
                      "share_resid_flip": float(sa.share[2]),
                      "share_seed_nldr": float(san.share[0]), "share_pair_nldr": float(san.share[1]),
                      "share_resid_nldr": float(san.share[2])})
B = pd.DataFrame(B)
B.to_csv(f"{OUT}/var_seed_das.csv", index=False)
print(B.round(3).to_string(index=False))

print("\n[B2] random-U seeds (20 Haar draws): spread of flip rate across draws vs sampling SE (E9 RS / OT, full_along_U)")
B2 = []
for c in CELLS:
    e9 = [v for k, v in data[c].items() if k.startswith("E9")][0]
    for cond in ("full_along_U", "repair"):
        for r, g in e9[(e9.condition == cond) & (e9.estimator == "random")].groupby("r"):
            fr = g[g.elig > 0].groupby("seed").hitA.mean()
            nr = g.groupby("seed").apply(lambda x: (x.D.mean() - x.DB.mean()) / (x.DA.mean() - x.DB.mean()))
            B2.append({"cell": c, "condition": cond, "r": r, "n_seeds": len(fr), "flip_mean": fr.mean(), "flip_sd": fr.std(),
                       "flip_p95": np.percentile(fr, 95), "nldr_mean": nr.mean(), "nldr_sd": nr.std()})
B2 = pd.DataFrame(B2)
print(B2.round(4).to_string(index=False))
B2.to_csv(f"{OUT}/var_seed_random.csv", index=False)

# ------------------------------------------------------------------ C. DAS across runs/machines (same nominal settings)
print("\n[C] DAS with identical nominal settings in different runs (E9/E10 node-C vs E17/E18/E19 node-D)")
C = []
for c in CELLS:
    for est in ("das1", "das2", "das4", "das8"):
        for cond in ("full_along_U", "feat_A_along_U", "repair"):
            frames = {}
            for k, v in data[c].items():
                s = v[(v.condition == cond) & (v.estimator == est) & (v.source.isin(["", "same_task"]))]
                if len(s):
                    frames[f"{k}@{v.host.iloc[0]}"] = s.drop_duplicates("pair_id").set_index("pair_id")
            if len(frames) < 2:
                continue
            common = sorted(set.intersection(*[set(v.index) for v in frames.values()]))
            el = np.c_[[frames[k].loc[common, "elig"] for k in frames]].T.min(1) > 0
            H = np.c_[[frames[k].loc[common, "hitA"] for k in frames]].T[el]
            N = np.c_[[wins(frames[k].loc[common, "nldr_i"]) for k in frames]].T
            C.append({"cell": c, "estimator": est, "condition": cond, "runs": " ".join(frames), "n_elig": int(el.sum()),
                      "flip_rates": " ".join(f"{x:.3f}" for x in H.mean(0)),
                      "flip_unanimous_share": float(np.mean(H.min(1) == H.max(1))),
                      "fleiss_kappa_flip": fleiss_kappa_binary(H) if 0 < H.mean() < 1 else np.nan,
                      "icc_nldr_i": icc1(N), "max_abs_dD": float(np.abs(np.c_[[frames[k].loc[common, "D"] for k in frames]].T -
                                                                      frames[list(frames)[0]].loc[common, "D"].to_numpy()[:, None]).max())})
C = pd.DataFrame(C)
C.to_csv(f"{OUT}/var_das_crossrun.csv", index=False)
print(C.round(3).to_string(index=False))

# U-matrix principal cosines for the same DAS name in different runs
print("\n[C2] principal cosines between DAS rotations with the same name in different runs")
C2 = []
for c, (task, L, sae) in CELLS.items():
    U = {}
    for e in ("E9", "E10", "E17", "E18", "E19", "E16"):
        for d in run_dirs(e):
            f = f"{d}directions/{task}_L{L}_{sae}.npz"
            try:
                z = np.load(f)
            except FileNotFoundError:
                continue
            for k in ("das1", "das2", "das4", "das8"):
                if k in z.files:
                    U[(f"{e}@{host(d)}", k)] = z[k]
    for k in ("das1", "das2", "das4", "das8"):
        keys = [x for x in U if x[1] == k]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = U[keys[i]], U[keys[j]]
                a = a if a.shape[0] == 2304 else a.T
                b = b if b.shape[0] == 2304 else b.T
                qa, qb = np.linalg.qr(a)[0], np.linalg.qr(b)[0]
                s = np.linalg.svd(qa.T @ qb, compute_uv=False)
                C2.append({"cell": c, "das": k, "a": keys[i][0], "b": keys[j][0], "cos_top": s[0], "cos_min": s[-1],
                           "max_abs_elem_diff": float(np.abs(a - b).max()) if a.shape == b.shape else np.nan})
C2 = pd.DataFrame(C2)
C2.to_csv(f"{OUT}/var_das_Ucos.csv", index=False)
print(C2.round(4).to_string(index=False))

# ------------------------------------------------------------------ D. layer: E8 atlas feat_A_errB across layers
print("\n[D] E8 atlas: per-pair feat_A_errB outcome decomposed into pair / layer / pair x layer")
D = []
d8 = run_dirs("E8")[0]
for task, layers in (("add_chain", [4, 8, 10, 12, 14, 16, 18, 20]), ("entity_track", [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20])):
    frames = []
    for L in layers:
        import pyarrow.dataset as pds
        saes = pds.dataset(d8 + "per_pair.parquet").to_table(columns=["sae"], filter=(pds.field("task") == task) & (
            pds.field("layer") == L)).to_pandas().sae.unique()
        s16 = [s for s in saes if "_16k_" in s]
        if not s16:
            continue
        t = load_cell(d8, task, L, s16[0], fit="pilot")
        t = t[(t.condition == "feat_A_errB") & (t.estimator == "")]
        t["L"] = L
        frames.append(t)
    t = pd.concat(frames)
    common = set.intersection(*[set(g.pair_id) for _, g in t.groupby("L")])
    t = t[t.pair_id.isin(common)].copy()
    t["nldr_w"] = wins(t.nldr_i)
    t["family"] = t.family.fillna("pilot").astype(str)
    sa = seq_anova(t, "nldr_w", [("L",), ("family",), ("family", "L"), ("pair_id",)])
    sa["task"], sa["outcome"], sa["n_pairs"], sa["n_layers"] = task, "nldr_i(wins)", len(common), t.L.nunique()
    D.append(sa)
    te = t[t.elig > 0]
    sb = seq_anova(te, "hitA", [("L",), ("family",), ("family", "L"), ("pair_id",)])
    sb["task"], sb["outcome"], sb["n_pairs"], sb["n_layers"] = task, "flip", te.pair_id.nunique(), te.L.nunique()
    D.append(sb)
D = pd.concat(D)
D.to_csv(f"{OUT}/var_layer_E8.csv", index=False)
print(D.round(4).to_string(index=False))

# ------------------------------------------------------------------ E. pair / family / condition / run decomposition at the cell
print("\n[E] RS L18 & OT L12: nldr_i(wins) and flip decomposed: family, pair|family, condition, family:cond, pair:cond, run(+machine)")
E = []
for c in CELLS:
    rows = []
    for k, v in data[c].items():
        s = v[v.condition.isin(["feat_A_errB", "feat_A_meanPos"]) & (v.estimator == "")]
        if s.condition.nunique() == 2:
            rows.append(s)
    t = pd.concat(rows)
    runs_ok = t.groupby("run").pair_id.nunique()
    common = set.intersection(*[set(g.pair_id) for _, g in t.groupby("run")])
    t = t[t.pair_id.isin(common)].copy()
    t["family"] = t.family.fillna("pilot").astype(str)
    t["nldr_w"] = wins(t.nldr_i)
    for yname, tt in (("nldr_i(wins)", t), ("flip", t[t.elig > 0])):
        y = "nldr_w" if yname.startswith("nldr") else "hitA"
        sa = seq_anova(tt, y, [("family",), ("pair_id",), ("condition",), ("family", "condition"), ("pair_id", "condition"),
                               ("host",), ("run",)])
        sa["cell"], sa["outcome"], sa["n_pairs"], sa["n_runs"] = c, yname, tt.pair_id.nunique(), tt.run.nunique()
        sa["hosts"] = ",".join(sorted(tt.host.unique()))
        E.append(sa)
E = pd.concat(E)
E.to_csv(f"{OUT}/var_decomp_cell.csv", index=False)
print(E.round(5).to_string(index=False))

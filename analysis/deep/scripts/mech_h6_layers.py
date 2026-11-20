"""H6 (exploratory, post hoc): layer profiles with 95% CIs. RS (E8): flip gap 1 - P(A | x_hat_A + eps_B) and its paired
L16 -> L18 change; OT (E1): P(B | x_hat_B + eps_A) (does B's move survive with A's error?) per layer; IOI and var_binding the same."""
from mech_common import *
pd.set_option("display.width", 250)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
rows = []
pp = load("E1", cols=["pair_id", "task", "layer", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
pp = wide(pp[pp.estimator.fillna("") == ""]).drop_duplicates(["pair_id", "layer", "condition"])
flipA = {}
for (task, layer), g in pp.groupby(["task", "layer"]):
    A = g.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="first")
    B = g.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="first")
    el = A.index[(A.clean_A == 1) & (B.clean_B == 1)]
    kB = int(B.loc[el, "feat_B_errA"].sum()); kA = int(A.loc[el, "feat_A_errB"].sum()); n = len(el)
    lo, hi = wilson(kB, n); lo2, hi2 = wilson(n - kA, n)
    rows.append(dict(task=task, layer=layer, n=n, P_B_given_xB_eA=kB / n, lo=lo, hi=hi, flipgap_xA_eB=(n - kA) / n, fg_lo=lo2, fg_hi=hi2))
    flipA[(task, layer)] = A.loc[el, "feat_A_errB"]
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "h6_layers.csv", index=False)
print(t.round(3).to_string(index=False))
a, b = flipA[("add_chain", 16)], flipA[("add_chain", 18)]
ix = a.index.intersection(b.index)
m, lo, hi = boot_ci((a.loc[ix].astype(float) - b.loc[ix].astype(float)).values)
print(f"RS paired flip(L16) - flip(L18) under x_hat_A + eps_B: {m:.3f} [{lo:.3f}, {hi:.3f}] n={len(ix)}")

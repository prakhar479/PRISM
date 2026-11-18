"""H3 (exploratory, post hoc): OT L12, does the error-PCA rank-1 direction carry B's latest binding update?
Joins E9 (along-U conditions) with E8 (feat_B_noise at L12) per pair, and splits by A's role (stay = never moved, moved)."""
from mech_common import *
from mech_ot_parse import table
pd.set_option("display.width", 250)
e9 = load("E9"); e9 = wide(e9[(e9.task == "entity_track")]).drop_duplicates(["pair_id", "key"])
e8 = load("E8", cols=["pair_id", "task", "layer", "condition", "estimator", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
e8 = wide(e8[(e8.task == "entity_track") & (e8.layer == 12) & (e8.estimator.fillna("") == "")]).drop_duplicates(["pair_id", "condition"])
par = table("entity_track_pilot").set_index("pair_id")
A9 = e9.pivot_table(index="pair_id", columns="key", values="isA", aggfunc="first")
B9 = e9.pivot_table(index="pair_id", columns="key", values="isB", aggfunc="first")
A8 = e8.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="first")
B8 = e8.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="first")
k = lambda c, e, r: f"{c}|{e}|same_task|{r}|-1"
el = A9.index[(A9["clean_A||nan|-1|-1".replace("nan", "")] == 1) if False else A9.index.isin(A9.index)]
cA = e9[e9.condition == "clean_A"].set_index("pair_id").isA; cB = e9[e9.condition == "clean_B"].set_index("pair_id").isB
el = cA.index[(cA == 1) & (cB.reindex(cA.index) == 1)].intersection(A8.index)
print("eligible", len(el))
fb = A8.loc[el, "feat_B_noise"]          # fallback: x_hat_B + noise -> A
rows = []
for est in ["pca_pooled", "answer_sep", "das1", "das4"]:
    for r in [1, 2, 4, 8]:
        ck = k("full_along_U", est, r); fk = k("feat_A_along_U", est, r)
        if ck not in A9: continue
        fu = A9.loc[el, ck]; fe = A9.loc[el, fk]
        for role in ["stay", "moved", "all"]:
            s = el if role == "all" else el[par.loc[el, "roleA"].values == role]
            rows.append(dict(est=est, r=r, roleA=role, n=len(s), full_along_U=fu.loc[s].mean(), feat_A_along_U=fe.loc[s].mean(),
                             fallback_featBnoise=fb.loc[s].mean(),
                             P_full_given_fallback=fu.loc[s][fb.loc[s] == 1].mean(), P_full_given_nofallback=fu.loc[s][fb.loc[s] == 0].mean(),
                             n_nofallback=int((fb.loc[s] == 0).sum())))
t = pd.DataFrame(rows); t.to_csv(SP / "deep" / "ot_alongU_fallback.csv", index=False)
print(t.round(3).to_string(index=False))
# role split for the 2x2 swaps at L12 (E8 has feat_A_errB, feat_B_noise; E1 has feat_B_errA)
print(pd.DataFrame({"n": par.loc[el].groupby("roleA").size(), "A|featBnoise": fb.groupby(par.loc[el, "roleA"]).mean(),
                    "B|featBnoise": B8.loc[el, "feat_B_noise"].groupby(par.loc[el, "roleA"]).mean()}).round(3))

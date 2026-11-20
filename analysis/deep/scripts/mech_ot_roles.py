"""OT (exploratory, post hoc): eligibility by A's role (stay / moved), and the swap outcomes per family and role, E3 fresh + E1/E8."""
from mech_common import *
from mech_ot_parse import table
pd.set_option("display.width", 250)
par = pd.concat([table("entity_track_pilot"), table("entity_track_e3")]).drop_duplicates("pair_id").set_index("pair_id")
out = []
for exp, filt in [("E3", "fresh"), ("E1", None), ("E14", None)]:
    pp = load(exp, cols=["pair_id", "task", "layer", "family", "fit_families", "condition", "estimator", "seed", "ans_A_id", "ans_B_id", "top1_id", "logit_A", "logit_B"])
    pp = pp[(pp.task == "entity_track") & (pp.estimator.fillna("") == "")]
    if filt: pp = pp[pp.fit_families == filt]
    if exp == "E14": pp = pp[(pp.family == "ot_en") & (pp.fit_families == "ot_en")]
    pp = wide(pp)
    for (layer, fam), g in pp.groupby(["layer", "family"]):
        A = g.pivot_table(index="pair_id", columns="condition", values="isA", aggfunc="mean")
        B = g.pivot_table(index="pair_id", columns="condition", values="isB", aggfunc="mean")
        D = g.pivot_table(index="pair_id", columns="condition", values="D", aggfunc="mean")
        role = par.reindex(A.index).roleA
        if role.isna().all():
            continue
        for rl in ["stay", "moved"]:
            s = A.index[role == rl]
            el = s[(A.loc[s, "clean_A"] == 1) & (B.loc[s, "clean_B"] == 1)]
            r = dict(exp=exp, layer=layer, family=fam, roleA=rl, n_all=len(s), clean_A_acc=A.loc[s, "clean_A"].mean(), clean_B_acc=B.loc[s, "clean_B"].mean(), n_elig=len(el))
            if len(el) >= 10:
                for c, M in [("feat_B_noise", A), ("feat_B_errA", A), ("feat_A_errB", A), ("feat_A_zero", A)]:
                    if c in M: r[f"A|{c}"] = M.loc[el, c].mean()
                if "feat_A_errB" in D:
                    r["nldr_featA_errB"] = ((D.loc[el, "feat_A_errB"] - D.loc[el, "clean_B"]).mean() / (D.loc[el, "full_A"] - D.loc[el, "clean_B"]).mean())
            out.append(r)
t = pd.DataFrame(out); t.to_csv(SP / "deep" / "ot_roles.csv", index=False)
print(t.round(3).to_string(index=False))

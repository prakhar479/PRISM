"""Exploratory: all-candidate margin. D* = logit_A - max over the other answer candidates (all task answers incl. B; cand_logits).
NLDR* = (mean D*_cond - mean D*_cleanB) / (mean D*_fullA - mean D*_cleanB), gap* = 1 - NLDR*; next to the pairwise gap (A vs B).
Also distractor_loss = mean over pairs of [max_other_but_B(cond) - max_other_but_B(fullA)] relative to... reported as the
change in the best non-B distractor logit relative to A: mean(D_dis_fullA) - mean(D_dis_cond), D_dis = logit_A - max non-A,non-B.
Bootstrap 95% CIs over pairs; difference to English (independent bootstrap). Native fit, f_A_correct, E14."""
import numpy as np, pandas as pd, glob
SP="/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
pp=pd.read_parquet(glob.glob(f"{SP}/ana_runs/E14/*/per_pair.parquet")[0])
pp=pp[(pp.fit_families==pp.family)&pp.f_A_correct&(pp.estimator.fillna("")=="")]
def margins(x):
    C=np.vstack(x.cand_logits.to_numpy()); la,lb=x.logit_A.to_numpy(),x.logit_B.to_numpy()
    isA=np.isclose(C,la[:,None]); isB=np.isclose(C,lb[:,None])
    oth=np.where(isA,-np.inf,C).max(1); dis=np.where(isA|isB,-np.inf,C).max(1)
    return la-lb, la-oth, la-dis
rng=np.random.default_rng(0); rows=[]; boots={}
for (task,fam),g in pp.groupby(["task","family"]):
    f={c:x.drop_duplicates("pair_id").set_index("pair_id") for c,x in g.groupby("condition")}
    ids=f["clean_B"].index.intersection(f["full_A"].index)
    M={c:np.vstack(margins(f[c].loc[ids])) for c in ["clean_B","full_A","feat_A_errB","feat_A_meanPos"] if c in f}
    n=len(ids); idx=[np.arange(n)]+[rng.integers(0,n,n) for _ in range(3000)]
    for c in ["feat_A_errB","feat_A_meanPos"]:
        def st(ix):
            m=lambda k,j:M[k][j,ix].mean()
            return [1-(m(c,j)-m("clean_B",j))/(m("full_A",j)-m("clean_B",j)) for j in range(3)]+[m("full_A",2)-m(c,2)]
        s=np.array([st(ix) for ix in idx]); boots[(task,fam,c)]=s[1:]
        lo,hi=np.percentile(s[1:],[2.5,97.5],axis=0)
        rows.append({"task":task,"family":fam,"cond":c,"n":n,"gap_pair":s[0,0],"gap_all":s[0,1],"gap_all_lo":lo[1],"gap_all_hi":hi[1],
                     "gap_distractor":s[0,2],"gap_dis_lo":lo[2],"gap_dis_hi":hi[2],"dis_logit_loss":s[0,3]})
t=pd.DataFrame(rows)
for i,r in t.iterrows():
    en=r.family.split("_")[0]+"_en"
    if r.family!=en and (r.task,en,r.cond) in boots:
        d=boots[(r.task,r.family,r.cond)][:,1]-boots[(r.task,en,r.cond)][:,1]
        t.loc[i,"gap_all_minus_en"]=r.gap_all-t[(t.task==r.task)&(t.family==en)&(t.cond==r.cond)].gap_all.iloc[0]
        t.loc[i,"lo_d"],t.loc[i,"hi_d"]=np.percentile(d,[2.5,97.5])
t.to_csv(f"{SP}/deep/ml_candidate_margin.csv",index=False)
pd.set_option("display.width",250); print(t.round(3).to_string(index=False))

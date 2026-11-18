"""Exploratory: headroom = logit_A - best other candidate under full_A (and clean_A), per family (E14, native fit, f_A_correct,
eligible pairs). A flip under feat_A_errB requires the feature-only run to keep D* > 0; with an (approximately) language-invariant
fractional loss, the flip rate is set by headroom. Check: predicted flips if every pair lost the family's mean fractional
all-candidate loss vs observed flips; and the median headroom."""
import numpy as np, pandas as pd, glob
SP="/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
pp=pd.read_parquet(glob.glob(f"{SP}/ana_runs/E14/*/per_pair.parquet")[0])
pp=pp[(pp.fit_families==pp.family)&pp.f_A_correct&(pp.estimator.fillna("")=="")]
def dstar(x):
    C=np.vstack(x.cand_logits.to_numpy()); la=x.logit_A.to_numpy(); isA=np.isclose(C,la[:,None]); return la-np.where(isA,-np.inf,C).max(1)
rows=[]
for (task,fam),g in pp.groupby(["task","family"]):
    f={c:x.drop_duplicates("pair_id").set_index("pair_id") for c,x in g.groupby("condition")}
    cb=f["clean_B"]; elig=cb.index[cb.top1_id==cb.ans_B_id]
    ids=elig.intersection(f["full_A"].index).intersection(f["feat_A_errB"].index)
    hA=dstar(f["full_A"].loc[ids]); hB=dstar(f["clean_B"].loc[ids]); hF=dstar(f["feat_A_errB"].loc[ids])
    frac=np.median((hA-hF)/(hA-hB))
    pred=((hA-frac*(hA-hB))>0).mean()
    rows.append({"task":task,"family":fam,"n":len(ids),"headroom_fullA_median":np.median(hA),"headroom_q25":np.percentile(hA,25),
                 "swing_median(hA-hB)":np.median(hA-hB),"headroom/swing":np.median(hA/(hA-hB)),"median_frac_loss":frac,
                 "flip_obs":(hF>0).mean(),"flip_pred_const_frac":pred})
t=pd.DataFrame(rows); t.to_csv(f"{SP}/deep/ml_headroom.csv",index=False)
pd.set_option("display.width",250); print(t.round(3).to_string(index=False))

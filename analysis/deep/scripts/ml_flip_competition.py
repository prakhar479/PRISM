"""Exploratory: why flip-rate gaps differ by language while NLDR gaps do not. For feat_A_errB on eligible pairs (clean B top-1 =
B's answer; f_A_correct), classify the top-1 token: A's answer (flip), B's answer, or a third token; and the fraction with
logit_A > logit_B (A wins the A-vs-B contest) among non-flips. Also full_A for reference. Token strings from the local tokenizer."""
import numpy as np, pandas as pd, glob, os
from tokenizers import Tokenizer
SP = str(__import__("layout_shim").root())
tok=Tokenizer.from_file(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/tokenizer.json"))[0])
pp=pd.read_parquet(glob.glob(f"{SP}/ana_runs/E14/*/per_pair.parquet")[0]); pp=pp[(pp.fit_families==pp.family)&pp.f_A_correct&(pp.estimator.fillna("")=="")]
rows=[]; ex={}
for (task,fam),g in pp.groupby(["task","family"]):
    cb=g[g.condition=="clean_B"].drop_duplicates("pair_id").set_index("pair_id"); elig=cb.index[cb.top1_id==cb.ans_B_id]
    for c in ["full_A","feat_A_errB","feat_A_meanPos"]:
        x=g[g.condition==c].drop_duplicates("pair_id").set_index("pair_id").reindex(elig).dropna(subset=["top1_id"])
        isA=x.top1_id==x.ans_A_id; isB=x.top1_id==x.ans_B_id; third=~isA&~isB
        rows.append({"task":task,"family":fam,"cond":c,"n_elig":len(x),"flip":isA.mean(),"top1_B":isB.mean(),"top1_third":third.mean(),
                     "A_beats_B":(x.logit_A>x.logit_B).mean(),"A_beats_B_but_third_top1":((x.logit_A>x.logit_B)&third).mean()})
        if c=="feat_A_errB" and third.any(): ex[(task,fam)]=pd.Series([tok.id_to_token(int(i)) for i in x.top1_id[third]]).value_counts().head(5).to_dict()
t=pd.DataFrame(rows); t.to_csv(f"{SP}/deep/ml_flip_competition.csv",index=False)
pd.set_option("display.width",250); print(t.round(3).to_string(index=False))
for k,v in ex.items(): print(k,v)

"""Exploratory: set-level aggregates (ratio of sums, as harness/fidelity.summarize) + decomposition of recovered."""
import pandas as pd, numpy as np
SP = str(__import__("layout_shim").root())
w=pd.read_parquet(f"{SP}/deep/ml_windows.parquet")
def agg(g):
    p=g.pred.sum(); c,s,m,z=(g[k].sum()/p for k in ("nll_clean","nll_splice","nll_mean","nll_zero"))
    return pd.Series({"ce_clean":c,"ce_delta":s-c,"abl_range_mean":m-c,"abl_range_zero":z-c,"rec_mean":(m-s)/(m-c),"rec_zero":(z-s)/(z-c),
        "fvu":g.fvu_num.sum()/g.fvu_den.sum(),"eps_x":g.eps_ratio_sum.sum()/g.pos.sum(),"l0":g.l0_sum.sum()/g.pos.sum(),
        "byte_frac":g.byte_frac.mean(),"piece_chars":g.piece_chars.mean(),"tpw":g.tpw_set.iloc[0],"n":len(g)})
t=w.groupby(["layer","set"]).apply(agg).reset_index()
t.to_csv(f"{SP}/deep/ml_setlevel.csv",index=False)
pd.set_option("display.width",250)
print(t.round(3).to_string(index=False))

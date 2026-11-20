"""Exploratory: window-level determinants of the splice cost. Per layer, OLS of ce_delta_w on fvu_w, ce_clean_w,
word_start_frac_w (share of SentencePiece word-initial tokens), with set fixed effects; cluster-robust SE by set is not possible
with few sets, so we report (a) within-set Spearman correlations and (b) a cross-set test: native-Indic indicator after
controlling for window covariates, with a set-level bootstrap (resample windows within each set)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SP = str(__import__("layout_shim").root())
w=pd.read_parquet(f"{SP}/deep/ml_windows.parquet"); w=w[w.set!="en_pilot100"].copy()
NATIVE={"hi_deva_dakshina","hi_deva_web","ta_taml_dakshina","ta_web","bn_beng_dakshina","bn_web"}
w["native_indic"]=w.set.isin(NATIVE).astype(float)
pd.set_option("display.width",250)
print("== within-set Spearman of ce_delta_w with window covariates (L12 and L18)")
rows=[]
for (L,s),g in w[w.layer.isin([12,18])].groupby(["layer","set"]):
    r={"layer":L,"set":s,"n":len(g)}
    for c in ["fvu","eps_x","ce_clean","word_start_frac","piece_chars","l0"]:
        r[c]=spearmanr(g.ce_delta,g[c])[0]
    rows.append(r)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("== pooled OLS (all sets), ce_delta ~ fvu + ce_clean + word_start_frac + native_indic, per layer; then + set FE")
import statsmodels.formula.api as smf
for L in [6,12,18,24]:
    g=w[w.layer==L]
    m=smf.ols("ce_delta ~ fvu + ce_clean + word_start_frac + native_indic",g).fit()
    m2=smf.ols("ce_delta ~ fvu + ce_clean + word_start_frac + C(set)",g).fit()
    print(L, "n",len(g),"R2",round(m.rsquared,3),{k:(round(v,3),round(m.bse[k],3)) for k,v in m.params.items()}, "| FE R2",round(m2.rsquared,3),
          "fvu",round(m2.params["fvu"],3),"ce_clean",round(m2.params["ce_clean"],3),"wsf",round(m2.params["word_start_frac"],3))

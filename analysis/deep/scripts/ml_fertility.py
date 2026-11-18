"""Exploratory: set-level fertility vs script. 12 sets (en_pile + 11 non-English), per layer: Spearman of tokens-per-word (MANIFEST)
and chars-per-token (tokenizer pieces) with rec_mean, ce_delta, fvu, and sensitivity; then within-group (native-script Indic only,
n=6; Latin-script n=6) correlations. Dakshina pairs: fertility ratio vs ce_delta ratio."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SP="/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
s=pd.read_csv(f"{SP}/deep/ml_setlevel.csv"); r=pd.read_csv(f"{SP}/deep/ml_ratios.csv")
s=s.merge(r[["layer","set","sens"]],on=["layer","set"]); s=s[s.set!="en_pilot100"]
NATIVE={"hi_deva_dakshina","hi_deva_web","ta_taml_dakshina","ta_web","bn_beng_dakshina","bn_web"}
s["native"]=s.set.isin(NATIVE)
for L in [12,18]:
    g=s[s.layer==L]
    for x in ["tpw","piece_chars"]:
        out={y:round(spearmanr(g[x],g[y])[0],2) for y in ["rec_mean","ce_delta","fvu","sens"]}
        nat={y:round(spearmanr(g[g.native][x],g[g.native][y])[0],2) for y in ["ce_delta","sens"]}
        lat={y:round(spearmanr(g[~g.native][x],g[~g.native][y])[0],2) for y in ["ce_delta","sens"]}
        print(L,x,"all n=",len(g),out,"| native n=",g.native.sum(),nat,"| latin n=",(~g.native).sum(),lat)
    print(g[["set","tpw","piece_chars","ce_delta","fvu","sens","rec_mean"]].sort_values("tpw").round(3).to_string(index=False))

"""Answer-value selectivity of low-rank directions: is the per-pair effect of full_along_U / feat_A_along_U explained by
WHICH answer the pair has (city, digit, box letter)? eta^2 of answer identity, permutation p, split-half replication of
per-answer means, and cross-cell (layer / language / run) correlation of per-answer means. Exploratory/post hoc."""
import numpy as np
import pandas as pd
from scipy import stats

import patterns_common as C

rng = np.random.default_rng(1)


def eta2(y, g):
    y = np.asarray(y, float); m = y.mean()
    ss_b = sum(len(v) * (v.mean() - m) ** 2 for v in (y[g == k] for k in np.unique(g)))
    return ss_b / ((y - m) ** 2).sum()


def analyse(df, label):
    """df: one cell, eligible pairs, cols pair_id, key, nldr, ansA, ansB."""
    out = []
    for key, g in df.groupby("key"):
        y = g.nldr.clip(-1, 3).values
        if np.std(y) < 1e-6 or len(y) < 40:
            continue
        for fac in ["ansA", "ansB"]:
            gr = g[fac].values
            e = eta2(y, gr)
            null = [eta2(y, rng.permutation(gr)) for _ in range(500)]
            # split-half: per-level means in two halves (levels with >=2 pairs in both halves)
            rs = []
            for s in range(20):
                h = rng.permutation(len(y)) % 2
                m1 = pd.Series(y[h == 0]).groupby(gr[h == 0]).agg(["mean", "size"])
                m2 = pd.Series(y[h == 1]).groupby(gr[h == 1]).agg(["mean", "size"])
                j = m1.join(m2, lsuffix="1", rsuffix="2", how="inner")
                j = j[(j.size1 >= 2) & (j.size2 >= 2)]
                if len(j) >= 4:
                    rs.append(stats.spearmanr(j.mean1, j.mean2)[0])
            out.append(dict(cell=label, key=key, factor=fac, n=len(y), n_levels=len(np.unique(gr)), mean=y.mean(),
                            eta2=e, eta2_null_p95=np.percentile(null, 95), p_perm=(np.sum(np.array(null) >= e) + 1) / 501,
                            splithalf_rho_med=np.median(rs) if rs else np.nan))
    return out


def ans_maps(d, task):
    from tokenizers import Tokenizer
    import glob
    tok = Tokenizer.from_file(glob.glob("/home/user/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/tokenizer.json")[0])
    ids = set(d.ans_A_id.unique()) | set(d.ans_B_id.unique())
    return {i: tok.decode([int(i)]).strip() for i in ids}


res, means = [], []
SEL = ["full_along_U", "feat_A_along_U"]
for exp in ["E18", "E9", "E17", "E20", "E22", "E16"]:
    d = C.add_refs(C.load(exp, filt=lambda x: x.condition.isin(SEL + ["clean_A", "clean_B"]) & (x.estimator != "random")))
    d = d[d.A_ok & d.B_ok & d.condition.isin(SEL)]
    d["key"] = C.ckey(d)
    for (t, L, s, fam), g in d.groupby(["task", "layer", "sae", "family"]):
        mp = ans_maps(g, t)
        g = g.assign(ansA=g.ans_A_id.map(mp), ansB=g.ans_B_id.map(mp))
        lab = f"{exp}:{t}:L{L}:{s}:{fam}"
        res += analyse(g, lab)
        m = g.groupby(["key", "ansA"]).nldr.agg(["mean", "size"]).reset_index()
        means.append(m.assign(cell=lab, exp=exp, task=t, layer=L, sae=s, family=fam))
    print(exp, "done", flush=True)
R = pd.DataFrame(res); R["q_bh"] = C.bh(R.p_perm.values)
R.to_csv(C.OUT / "answer_selectivity.csv", index=False)
M = pd.concat(means); M.to_csv(C.OUT / "answer_selectivity_means.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
R["est"] = R.key.str.split("|").str[1]
R["kind"] = np.where(R.est.str.startswith("das"), "das", R.est)
print(f"{len(R)} tests (BH): share q<.05 by cell x kind x factor; median eta2 and split-half rho")
print(R.groupby(["cell", "kind", "factor"]).agg(n_keys=("eta2", "size"), eta2_med=("eta2", "median"), null95_med=("eta2_null_p95", "median"),
                                                 sig=("q_bh", lambda q: (q < .05).mean()), split_rho=("splithalf_rho_med", "median"),
                                                 mean_nldr=("mean", "median")).round(3).to_string())

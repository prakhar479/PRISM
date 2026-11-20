"""E19 steering, candidate level: what do pairs answer along the steering path, is the attractor pair-independent, and
does it match the train-set answer-frequency imbalance n(ans_A=c) - n(ans_B=c)? Exploratory/post hoc."""
import glob
import gzip
import json

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

import patterns_common as C

tok = Tokenizer.from_file(glob.glob("/home/user/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/tokenizer.json")[0])
dec = lambda i: tok.decode([int(i)])
p = C.OUT / "e19_nr.parquet"
d = pd.read_parquet(p) if p.exists() else C.add_refs(C.load("E19", cand=True, filt=lambda x: x.estimator != "random"))
d = d[d.A_ok & d.B_ok]
st = d[d.condition.str.startswith("steer")].copy()
st["s"] = st.condition.str.replace("steer_x", "").astype(float)
rng = np.random.default_rng(0)

# --- train-set answer imbalance per candidate (prompt files used by E19)
imb = {}
for t in ["add_chain", "entity_track", "people_cities"]:
    sn = C.set_name_for(C.run_dirs("E19")[0], t)
    rows = [json.loads(l) for l in gzip.open(C.PS / f"{sn}.jsonl.gz", "rt")]
    tr = [r for r in rows if r["split"] == "train"]
    a = pd.Series([r["ans_A"] for r in tr]).value_counts(); b = pd.Series([r["ans_B"] for r in tr]).value_counts()
    imb[t] = a.sub(b, fill_value=0).sort_values(ascending=False)
    print(f"{t} ({sn}, n_train={len(tr)}) imbalance n(A=c)-n(B=c):",
          ", ".join(f"{dec(k)!r}:{int(v):+d}" for k, v in imb[t].items()))

# --- attractor table: modal top1 per (task, est, r, s), share, in each half of pairs
out = []
for t, g in st.groupby("task"):
    pids = np.array(sorted(g.pair_id.unique())); rng.shuffle(pids)
    half = dict(zip(pids, np.arange(len(pids)) % 2))
    g = g.assign(half=g.pair_id.map(half))
    base = d[(d.task == t) & (d.condition == "clean_B")].top1_id.value_counts(normalize=True)
    for (e, r, s), h in g.groupby(["estimator", "r", "s"]):
        vc = h.top1_id.value_counts(normalize=True)
        m = vc.index[0]
        sh = [(hh.top1_id == m).mean() for _, hh in h.groupby("half")]
        oth = h[~h.isA & ~h.isB]
        out.append(dict(task=t, est=e, r=r, s=s, n=len(h), flipA=h.isA.mean(), stayB=h.isB.mean(), other=1 - h.isA.mean() - h.isB.mean(),
                        modal=dec(m), modal_share=vc.iloc[0], modal_share_h1=sh[0], modal_share_h2=sh[1],
                        modal_base_share=base.get(m, 0.0), modal_is_candidate=bool(np.isin(m, h.ans_A_id.unique()) | np.isin(m, h.ans_B_id.unique())),
                        imb_rank=(list(imb[t].index).index(m) + 1) if m in imb[t].index else np.nan,
                        modal_other=dec(oth.top1_id.value_counts().index[0]) if len(oth) else "",
                        modal_other_share=oth.top1_id.value_counts(normalize=True).iloc[0] if len(oth) else np.nan))
A = pd.DataFrame(out); A.to_csv(C.OUT / "steer_attractors.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(A[A.s >= 4].round(3).to_string())

# --- dose shape: per pair, scale at which D peaks (interior peak = non-monotone); share of pairs whose D at s=16 < D at s=1
rows = []
for (t, e, r), g in st.groupby(["task", "estimator", "r"]):
    P = g.pivot_table(index="pair_id", columns="s", values="D")
    pk = P.columns.values[np.argmax(P.values, 1)]
    rows.append(dict(task=t, est=e, r=r, n=len(P), peak_s_median=np.median(pk), share_peak_interior=np.mean((pk > P.columns.min()) & (pk < P.columns.max())),
                     share_D16_below_D1=np.mean(P[16.0] < P[1.0]), flip_s1=g[g.s == 1].isA.mean(), flip_max=g.groupby("s").isA.mean().max(),
                     s_at_flip_max=g.groupby("s").isA.mean().idxmax(), flip_s16=g[g.s == 16].isA.mean()))
Dd = pd.DataFrame(rows); Dd.to_csv(C.OUT / "steer_dose_shape.csv", index=False)
print(Dd.round(3).to_string())

# --- RS: is steering pair-specific at all? flip by sign of A-B and digit shift of the top-1
rs = st[st.task == "add_chain"].copy()
digit = {int(i): int(dec(i)) for i in rs.top1_id.unique() if dec(i).strip().isdigit()}
F = C.features("add_chain", rs.pair_id.unique(), C.set_name_for(C.run_dirs("E19")[0], "add_chain")).set_index("pair_id")
rs["delta"] = rs.pair_id.map(F.delta); rs["ansB"] = rs.pair_id.map(F.ans_B)
rs["top_digit"] = rs.top1_id.map(digit)
rs["shift"] = rs.top_digit - rs.ansB
t2 = rs.groupby(["estimator", "r", "s"]).apply(lambda g: pd.Series(dict(
    flip_deltapos=g[g.delta > 0].isA.mean(), flip_deltaneg=g[g.delta < 0].isA.mean(),
    n_pos=(g.delta > 0).sum(), n_neg=(g.delta < 0).sum(),
    mean_shift=g["shift"].mean(), share_digit=g.top_digit.notna().mean(), modal_digit=g.top_digit.mode().iloc[0] if g.top_digit.notna().any() else np.nan)))
t2.to_csv(C.OUT / "steer_rs_sign.csv")
print(t2.round(3).to_string())

# --- RS number line: per-pair digit trajectory along s ("1" coded as 10 = first token of a two-digit number)
from scipy.stats import fisher_exact
rs["dig10"] = rs.top_digit.where(rs.top_digit != 1, 10)
print("\nRS per-pair digit trajectories (digit as fn of s; '1' -> 10):")
res = []
for (e, r), g in rs.groupby(["estimator", "r"]):
    P = g.pivot_table(index="pair_id", columns="s", values="dig10")
    dif = np.diff(P.values, axis=1)
    up = (dif >= 0).all(1); down = (dif <= 0).all(1); moved = (dif != 0).any(1)
    smax = g.groupby("s").isA.mean().idxmax(); h = g[g.s == smax]
    tab = [[(h[h.delta > 0].isA).sum(), (~h[h.delta > 0].isA).sum()], [(h[h.delta < 0].isA).sum(), (~h[h.delta < 0].isA).sum()]]
    orr, pf = fisher_exact(tab)
    # split-half replication of sign asymmetry
    ids = np.array(sorted(h.pair_id.unique())); rng.shuffle(ids); H = set(ids[: len(ids) // 2])
    asym = [(hh[hh.delta > 0].isA.mean() - hh[hh.delta < 0].isA.mean()) for hh in (h[h.pair_id.isin(H)], h[~h.pair_id.isin(H)])]
    res.append(dict(est=e, r=r, n=len(P), share_moved=moved.mean(), share_monotone_up_of_moved=(up & moved).sum() / max(moved.sum(), 1),
                    share_monotone_down_of_moved=(down & moved).sum() / max(moved.sum(), 1), s_flipmax=smax,
                    flip_pos=h[h.delta > 0].isA.mean(), flip_neg=h[h.delta < 0].isA.mean(), fisher_p=pf, asym_h1=asym[0], asym_h2=asym[1],
                    mean_digit_s16_minus_s0=(P[16.0] - P.iloc[:, 0]).mean()))
R = pd.DataFrame(res); R["fisher_q_bh"] = C.bh(R.fisher_p.values); R.to_csv(C.OUT / "steer_rs_numberline.csv", index=False)
print(R.round(4).to_string())

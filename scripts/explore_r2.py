#!/usr/bin/env python
"""EXPLORATORY analyses of release 2 (not preregistered; no multiplicity correction; 95% pair-bootstrap CIs, 1000 resamples).

    python scripts/explore_r2.py --runs <runs_root> [--sections A,B,...] [--out explore_r2.md]

Prints compact tables only. Metrics use the same code as E8/the gate (harness/e8.py: flip rate on pairs whose clean B run
predicts B's answer; NLDR against clean_A / clean_B). Sections:
  A  E3 gate quantity per evaluation family (fit fresh, eval fresh, filter f_pilot)
  B  B-run fragility: share of pairs whose clean B run already answers A, and the noise control, by family and corrupt step
  C  error-condition ladder in the two confirmatory cells (E3, fresh, f_pilot): every baseline's flip rate and NLDR
  D  where the running-sum shortfall sits: by corrupt step, by answer distance |A-B|, by answer value (E3 fresh, f_pilot)
  E  E4: gap vs width and realized L0 across all OT L12 SAEs (NLDR, prob recovery), with a log-log fit
  F  E5: share of the features-only and meanPos gaps closed by subthreshold add-back, by alpha, slot, greedy k
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from harness import e8 as E
from harness import stats as St
from harness import tasks as T

NB = 1000
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def table(df, cols=None, r=3):
    say((df[cols] if cols else df).round(r).to_string(index=False))
    say()


def load(runs, exp, cols=None):
    f = glob.glob(f"{runs}/{exp}/*/per_pair.parquet")
    return pd.read_parquet(f[0], columns=cols) if f else None


class Cell:
    """Per-pair arrays of one cell/filter, aligned on the pairs present in every condition used."""

    def __init__(self, g, flt, keys=("condition",)):
        g = g[g[flt]]
        self.frames = {k if len(keys) > 1 else k[0] if isinstance(k, tuple) else k: f.set_index("pair_id")
                       for k, f in g.groupby(list(keys), dropna=False)}
        base = [f for k, f in self.frames.items() if (k if isinstance(k, str) else k[0]) in ("clean_A", "clean_B", "full_A")]
        pairs = base[0].index
        for f in base[1:]:
            pairs = pairs.intersection(f.index)
        self.pairs = pairs.sort_values()
        ca = self.get("clean_A")
        self.ansA, self.ansB = ca.ans_A_id.to_numpy(), ca.ans_B_id.to_numpy()
        self.A, self.B = E.Arr(ca, self.ansA), E.Arr(self.get("clean_B"), self.ansA)
        self.elig = (self.get("clean_B").top1_id.to_numpy() == self.ansB).astype(float)
        self.n = len(self.pairs)
        self.W = E.weights([np.arange(self.n)] + list(St.boot_indices(self.n, NB, 0)), self.n)

    def key(self, cond):
        return next(k for k in self.frames if (k if isinstance(k, str) else k[0]) == cond)

    def get(self, cond_or_key):
        k = cond_or_key if cond_or_key in self.frames else self.key(cond_or_key)
        f = self.frames[k]
        return f.loc[f.index.intersection(self.pairs)].reindex(self.pairs)

    def m(self, k, W=None):
        return E.metric_samples(E.Arr(self.get(k), self.ansA), self.A, self.B, self.elig, self.W if W is None else W)


def ci(s):
    lo, hi = E._ci(s[1:])
    return float(s[0]), lo, hi


def sec_A(runs):
    pp = load(runs, "E3")
    say("## A. E3 gate quantity per evaluation family (fit fresh, eval fresh, f_pilot)")
    rows = []
    for (task, layer), g in pp[(pp.fit_families == "fresh")].groupby(["task", "layer"]):
        for fam, gf in g.groupby("family"):
            if fam.startswith("pilot"):
                continue
            c = Cell(gf[~gf.condition.isin(["repair", "repair_mean"])], "f_pilot", ("condition", "seed"))
            fa, mp = c.m(c.key("full_A"))["flip_rate"], c.m(c.key("feat_A_meanPos"))["flip_rate"]
            eb = c.m(c.key("feat_A_errB"))["flip_rate"]
            nzk = [k for k in c.frames if k[0] == "feat_B_noise"]
            nz = float(np.nanmean([c.m(k)["flip_rate"][0] for k in nzk])) if nzk else np.nan
            s, lo, hi = ci(fa - mp)
            rows.append({"task": task, "family": fam, "n": c.n, "elig": int(c.elig.sum()), "full_A": fa[0], "meanPos": mp[0],
                         "errB": eb[0], "noise_ctrl": nz, "shortfall": s, "lo": lo, "hi": hi})
    table(pd.DataFrame(rows))


def sec_B(runs):
    say("## B. B-run fragility: clean B already answers A (top-1), by family and corrupt step (E3, all candidates in f_none)")
    pp = load(runs, "E3", ["pair_id", "task", "family", "condition", "top1_id", "ans_A_id", "ans_B_id", "f_none", "f_pilot", "fit_families"])
    pp = pp[(pp.condition == "clean_B") & (pp.fit_families == "fresh")].drop_duplicates("pair_id")
    meta = {}
    for t in ("entity_track", "add_chain"):
        rows, _ = T.load_set(Path(runs).parent / "hw_data" / f"{t}_e3.jsonl.gz") if False else T.load_set(DATA / f"{t}_e3.jsonl.gz")
        meta.update({r["pair_id"]: r["meta"].get("corrupt_step") for r in rows})
    pp["corrupt_step"] = pp.pair_id.map(meta)
    pp["B_says_A"] = pp.top1_id == pp.ans_A_id
    pp["B_says_B"] = pp.top1_id == pp.ans_B_id
    g = pp.groupby(["task", "family"]).agg(n=("pair_id", "size"), B_says_A=("B_says_A", "mean"), B_says_B=("B_says_B", "mean"),
                                           in_f_pilot=("f_pilot", "mean")).reset_index()
    table(g)
    g2 = pp.groupby(["task", "corrupt_step"]).agg(n=("pair_id", "size"), B_says_A=("B_says_A", "mean"), B_says_B=("B_says_B", "mean")).reset_index()
    table(g2)


def sec_C(runs):
    pp = load(runs, "E3")
    say("## C. Error-condition ladder in the confirmatory cells (E3, fit fresh, eval fresh families, f_pilot)")
    rows = []
    fresh = pp[(pp.fit_families == "fresh") & ~pp.family.str.startswith("pilot")]
    for (task, layer), g in fresh.groupby(["task", "layer"]):
        c = Cell(g[~g.condition.isin(["repair", "repair_mean"])], "f_pilot", ("condition", "seed"))
        for k in sorted(c.frames, key=str):
            if k[0] in ("clean_A", "clean_B"):
                continue
            mm = c.m(k)
            f, flo, fhi = ci(mm["flip_rate"])
            nl, nlo, nhi = ci(mm["nldr"])
            rows.append({"task": task, "condition": k[0], "seed": k[1], "flip": f, "flip_lo": flo, "flip_hi": fhi, "nldr": nl, "nldr_lo": nlo, "nldr_hi": nhi})
    table(pd.DataFrame(rows))


def sec_D(runs):
    pp = load(runs, "E3")
    say("## D. Running-sum L18 shortfall (full_A - meanPos flip) by pair features (E3, fit fresh, eval fresh, f_pilot)")
    g = pp[(pp.task == "add_chain") & (pp.fit_families == "fresh") & ~pp.family.str.startswith("pilot") & pp.f_pilot]
    rows, _ = T.load_set(DATA / "add_chain_e3.jsonl.gz")
    meta = {r["pair_id"]: (r["meta"].get("corrupt_step"), int(r["clean_target"]), int(r["corrupt_target"])) for r in rows}
    c = Cell(g[g.condition.isin(["clean_A", "clean_B", "full_A", "feat_A_meanPos", "feat_A_errB"])], "f_pilot")
    hit = lambda k: E.Arr(c.get(k), c.ansA).hitA
    d = pd.DataFrame({"pair_id": c.pairs, "elig": c.elig, "full": hit("full_A"), "mp": hit("feat_A_meanPos"), "eb": hit("feat_A_errB")})
    d[["step", "A", "B"]] = pd.DataFrame([meta[p] for p in d.pair_id], index=d.index)
    d["dist"] = (d.A - d.B).abs()
    d = d[d.elig > 0]
    for by in ("step", "dist", "A"):
        t = d.groupby(by).agg(n=("pair_id", "size"), full=("full", "mean"), meanPos=("mp", "mean"), errB=("eb", "mean")).reset_index()
        t["shortfall_meanPos"] = t.full - t.meanPos
        table(t)


def sec_E(runs):
    say("## E. E4: gap vs width and realized L0, OT L12, filter f_pilot (gap = M(full_A) - M(feat_A_errB))")
    f = glob.glob(f"{runs}/E4/*/per_pair.parquet")[0]
    pp = pd.read_parquet(f)
    summ = json.loads((Path(f).parent / "summary.json").read_text())
    l0 = {c["cell"]: (float(np.mean(c["mean_l0_test"])), (c.get("generic") or {}).get("mean_l0"), (c.get("generic") or {}).get("err_energy_ratio"), c.get("rel_err_norm_test")) for c in summ["cells"]}
    rows = []
    for sae, g in pp[(pp.task == "entity_track")].groupby("sae"):
        c = Cell(g[g.condition.isin(["clean_A", "clean_B", "full_A", "feat_A_errB", "feat_A_meanPos"])], "f_pilot")
        fa, eb, mp = c.m("full_A"), c.m("feat_A_errB"), c.m("feat_A_meanPos")
        lt, lg, er, rn = l0.get(f"entity_track_{sae}", (np.nan,) * 4)
        w = sae.split("_")[1]
        rows.append({"sae": sae, "width_k": int(w[:-1]) * (1024 if w.endswith("m") else 1), "l0_test": lt, "l0_generic": lg,
                     "generic_err_energy": er, "test_rel_err_norm": rn,
                     "gap_nldr": ci(fa["nldr"] - eb["nldr"])[0], "gap_nldr_lo": ci(fa["nldr"] - eb["nldr"])[1], "gap_nldr_hi": ci(fa["nldr"] - eb["nldr"])[2],
                     "gap_prob": (fa["prob_recovery"] - eb["prob_recovery"])[0], "gap_flip": (fa["flip_rate"] - eb["flip_rate"])[0],
                     "meanPos_gap_nldr": (fa["nldr"] - mp["nldr"])[0]})
    d = pd.DataFrame(rows).sort_values(["width_k", "l0_test"])
    table(d)
    X = np.column_stack([np.ones(len(d)), np.log2(d.width_k), np.log2(d.l0_test)])
    for y in ("gap_nldr", "gap_prob"):
        b, *_ = np.linalg.lstsq(X, d[y].to_numpy(), rcond=None)
        pred = X @ b
        r2 = 1 - ((d[y] - pred) ** 2).sum() / ((d[y] - d[y].mean()) ** 2).sum()
        say(f"OLS {y} ~ 1 + log2(width) + log2(L0_test): intercept {b[0]:.3f}, per doubling width {b[1]:+.4f}, per doubling L0 {b[2]:+.4f}, R^2 {r2:.2f} (n={len(d)}; descriptive)")
    say(f"corr(gap_nldr, generic_err_energy) = {d.gap_nldr.corr(d.generic_err_energy):.2f}; corr(gap_nldr, test_rel_err_norm) = {d.gap_nldr.corr(d.test_rel_err_norm):.2f}")
    say()


def sec_F(runs):
    pp = load(runs, "E5")
    say("## F. E5: subthreshold add-back, share of gap closed (f_pilot): (M(cond) - M(baseline)) / (M(full_A) - M(baseline))")
    rows = []
    for (task, sae), g in pp.groupby(["task", "sae"]):
        c = Cell(g, "f_pilot", ("condition", "estimator", "source", "r"))
        fa = c.m(c.key("full_A"))
        for k in c.frames:
            if k[0] != "subthresh":
                continue
            slot = k[2].split("slot=")[-1]
            base = c.m(c.key("feat_A_errB" if slot == "B" else "feat_A_meanPos"))
            mm = c.m(k)
            for met in ("flip_rate", "nldr"):
                den = fa[met] - base[met]
                with np.errstate(invalid="ignore", divide="ignore"):
                    sh = np.where(np.abs(den) > 1e-9, (mm[met] - base[met]) / den, np.nan)
                s, lo, hi = ci(sh)
                rows.append({"task": task, "sae": sae, "metric": met, "slot": slot, "alpha": k[2].split("|")[0].split("=")[1],
                             "est": k[1], "k": int(k[3]), "gap": den[0], "share": s, "lo": lo, "hi": hi})
    d = pd.DataFrame(rows)
    d = d[(d.metric == "nldr") | (d.gap.abs() > 0.02)]
    for (task, sae, met), g in d.groupby(["task", "sae", "metric"]):
        say(f"{task} {sae} {met}: gap {g.gap.iloc[0]:.3f}")
        piv = g.pivot_table(index=["slot", "alpha"], columns=["est", "k"], values="share").round(2)
        say(piv.to_string())
        say()


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "prompt_sets"))
    ap.add_argument("--sections", default="A,B,C,D,E,F")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    DATA = Path(a.data)
    say("# Release 2: EXPLORATORY analyses (not preregistered; no multiplicity correction)")
    say()
    for s in a.sections.split(","):
        globals()[f"sec_{s}"](a.runs)
    if a.out:
        Path(a.out).write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())

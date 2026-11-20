"""AUDIT (claims auditor, 2026-11-19): independent recomputation of the multilingual text-fidelity claims (C7 and the Corrections
table) of hw/analysis/team_analysis_2026-11-19.md from the raw E12 (16k) / E21 (65k) per-window npz files.
New code: no imports from team/, deep/ or hw/analysis scripts (their docstrings were read only to learn definitions).

Per window: cost = (nll_splice - nll_clean)/pred; nllc = nll_clean/pred; eps = eps_ratio_sum/pos (mean |eps|/|x|);
fvu = fvu_num/fvu_den. Native = {hi_deva_dakshina, hi_deva_web, ta_taml_dakshina, ta_web, bn_beng_dakshina, bn_web};
Latin = {hi/ta/bn_latn_dakshina, en_pile, fr_web, hinglish_phinc}. en_pilot100 excluded.
CEM: shared grid, bins nllc 0.25 nats, log eps 0.05; reference windows reweighted to the target's stratum counts; ratio =
mean cost(T matched) / weighted mean cost(R). "supported" = >= 50 % of target windows matched and >= 100 matched.
Bootstraps (windows within set): B = 500 for CEM, 1000 for the overlap and width contrasts. Seeds fixed below.
Usage: OMP_NUM_THREADS=2 python audit_ml.py [part]   part in {match, overlap, width, regress, all}
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
OUT = f"{SP}/team/audit_out"
os.makedirs(OUT, exist_ok=True)
NATIVE = ["hi_deva_dakshina", "hi_deva_web", "ta_taml_dakshina", "ta_web", "bn_beng_dakshina", "bn_web"]
LATIN = ["hi_latn_dakshina", "ta_latn_dakshina", "bn_latn_dakshina", "en_pile", "fr_web", "hinglish_phinc"]
LANG = {"hi_deva_dakshina": "hi", "hi_deva_web": "hi", "hi_latn_dakshina": "hi", "hinglish_phinc": "hi",
        "ta_taml_dakshina": "ta", "ta_web": "ta", "ta_latn_dakshina": "ta", "bn_beng_dakshina": "bn", "bn_web": "bn",
        "bn_latn_dakshina": "bn", "en_pile": "en", "fr_web": "fr"}
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def load():
    fr = []
    for d in sorted(glob.glob(f"{SP}/ana_runs/E12/*/")):
        m = json.load(open(d + "manifest.json"))
        if m.get("status") != "complete":
            continue
        for f in sorted(glob.glob(d + "per_window_*.npz")):
            s, L = os.path.basename(f)[11:-4].rsplit("_L", 1)
            if s == "en_pilot100":
                continue
            z = np.load(f)
            df = pd.DataFrame({k: z[k] for k in ("nll_clean", "nll_splice", "pred", "fvu_num", "fvu_den", "eps_ratio_sum", "pos")})
            df["set"], df["layer"], df["width"] = s, int(L), m["config"]["width"]
            df["win"] = np.arange(len(df))
            fr.append(df)
    w = pd.concat(fr, ignore_index=True)
    w["cost"] = (w.nll_splice - w.nll_clean) / w.pred
    w["nllc"] = w.nll_clean / w.pred
    w["eps"] = w.eps_ratio_sum / w.pos
    w["fvu"] = w.fvu_num / w.fvu_den
    w["le"], w["lf"] = np.log(w.eps), np.log(w.fvu)
    w["native"] = w.set.isin(NATIVE).astype(int)
    return w


BIN = {"nllc": 0.25, "le": 0.05, "lf": 0.10}


def strata(df, cols):
    key = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        b = np.floor(df[c].to_numpy() / BIN[c]).astype(np.int64)
        key = key * 1_000_003 + (b - b.min())
    return np.unique(key, return_inverse=True)[1]


def cem(st, yt, sr, yr):
    S = max(st.max(), sr.max()) + 1
    nt, nr = np.bincount(st, minlength=S), np.bincount(sr, minlength=S)
    ok = (nt > 0) & (nr > 0)
    if not ok.any():
        return np.nan, 0, 0.0
    mt = np.bincount(st, yt, minlength=S)[ok].sum() / nt[ok].sum()
    mr = (nt[ok] * np.bincount(sr, yr, minlength=S)[ok] / nr[ok]).sum() / nt[ok].sum()
    return mt / mr, int(nt[ok].sum()), nt[ok].sum() / len(st)


def part_match(w, B=500):
    rng = np.random.default_rng(101)
    rows = []
    for (L, wd), g in w.groupby(["layer", "width"]):
        sets = set(g.set)
        for T in sorted(sets):
            refs = {"en_pile": ["en_pile"], "latin_other": [s for s in LATIN if s in sets and s != T]}
            for rn, rs in refs.items():
                if T in rs or not rs:
                    continue
                gt, gr = g[g.set == T], g[g.set.isin(rs)]
                grp = [np.flatnonzero(gr.set.to_numpy() == s) for s in rs]
                for spec, cols in (("P", ["nllc"]), ("PE", ["nllc", "le"])):
                    s = strata(pd.concat([gt[cols], gr[cols]]), cols)
                    st, sr = s[:len(gt)], s[len(gt):]
                    yt, yr = gt.cost.to_numpy(), gr.cost.to_numpy()
                    r, nm, sh = cem(st, yt, sr, yr)
                    bs = []
                    for _ in range(B):
                        it = rng.integers(0, len(gt), len(gt))
                        ir = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in grp])
                        bs.append(cem(st[it], yt[it], sr[ir], yr[ir])[0])
                    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                    rows.append(dict(layer=L, width=wd, target=T, native=int(T in NATIVE), ref=rn, spec=spec, ratio=r, lo=lo,
                                     hi=hi, n_matched=nm, share=sh, supported=bool(sh >= 0.5 and nm >= 100)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/ml_matched.csv", index=False)
    main = R[R.layer.isin([12, 18])]

    def summ(d, lab):
        print(f"{lab}: n={len(d)} median {d.ratio.median():.3f} range [{d.ratio.min():.3f}, {d.ratio.max():.3f}] "
              f"point>1 {int((d.ratio > 1).sum())}/{len(d)} CI>1 {int((d.lo > 1).sum())}/{len(d)}")
    nat, lat = main[main.native == 1], main[main.native == 0]
    summ(nat[(nat.ref == "en_pile") & (nat.spec == "P")], "(b) native/en, nllc CEM, all 24")
    x = nat[(nat.ref == "en_pile") & (nat.spec == "PE")]
    summ(x, "(c) native/en, nllc+eps CEM, all"); summ(x[x.supported], "(c) native/en, nllc+eps CEM, supported")
    x = nat[(nat.ref == "latin_other") & (nat.spec == "PE")]
    summ(x, "(c') native/pooled Latin, PE, all"); summ(x[x.supported], "(c') native/pooled Latin, PE, supported")
    for sp in ("P", "PE"):
        x = lat[(lat.ref == "latin_other") & (lat.spec == sp)]
        summ(x, f"(d) Latin placebo/pooled other Latin {sp}, all"); summ(x[x.supported], f"(d) placebo {sp}, supported")
    x = R[(R.native == 1) & (R.ref == "en_pile") & (R.spec == "PE")]
    print("(e) per layer x width, native/en PE (supported only in brackets):")
    for (L, wd), d in x.groupby(["layer", "width"]):
        ds = d[d.supported]
        print(f"   L{L} {wd}: all {d.ratio.min():.2f}-{d.ratio.max():.2f} median {d.ratio.median():.2f}; supported n={len(ds)} "
              f"{ds.ratio.min() if len(ds) else np.nan:.2f}-{ds.ratio.max() if len(ds) else np.nan:.2f}")
    print(x[x.target.str.startswith("hi_deva") | (x.layer.isin([6, 24]))][["layer", "width", "target", "ratio", "lo", "hi", "share",
                                                                             "n_matched", "supported"]].round(3).to_string(index=False))
    print("(c) individual cells, native/en PE:")
    sel = x[x.layer.isin([12, 18])].sort_values(["layer", "width", "target"])
    print(sel[["layer", "width", "target", "ratio", "lo", "hi", "share", "n_matched", "supported"]].round(3).to_string(index=False))
    # raw (unmatched) ratios
    raw = []
    for (L, wd), g in w[w.layer.isin([12, 18])].groupby(["layer", "width"]):
        en = g[g.set == "en_pile"].cost.mean()
        for T in NATIVE:
            raw.append(g[g.set == T].cost.mean() / en)
    raw = np.array(raw)
    print(f"(a) raw native/en mean-cost ratio over 24 cells: geometric mean {np.exp(np.log(raw).mean()):.3f}, "
          f"median {np.median(raw):.3f}, range {raw.min():.2f}-{raw.max():.2f}")
    return R


def part_overlap(w, B=1000):
    rng = np.random.default_rng(202)
    rows = []
    for (L, wd), g in w.groupby(["layer", "width"]):
        t, r = g[g.set == "hi_deva_dakshina"], g[g.set == "hi_latn_dakshina"]
        if not len(t) or not len(r):
            continue
        a, b = max(t.nllc.min(), r.nllc.min()), min(t.nllc.max(), r.nllc.max())
        ti, ri = t[(t.nllc >= a) & (t.nllc <= b)], r[(r.nllc >= a) & (r.nllc <= b)]
        yt, yr = ti.cost.to_numpy(), ri.cost.to_numpy()
        for lab, cols in (("raw", []), ("adj_nllc", ["nllc"]), ("adj_nllc_le", ["nllc", "le"])):
            Xt, Xr = ti[cols].to_numpy(), ri[cols].to_numpy()

            def f(it, ir):
                if not cols:
                    return yt[it].mean() / yr[ir].mean()
                xb = np.vstack([Xt[it], Xr[ir]]).mean(0)
                ct = np.linalg.lstsq(np.c_[np.ones(len(it)), Xt[it]], yt[it], rcond=None)[0]
                cr = np.linalg.lstsq(np.c_[np.ones(len(ir)), Xr[ir]], yr[ir], rcond=None)[0]
                return (ct[0] + xb @ ct[1:]) / (cr[0] + xb @ cr[1:])
            p = f(np.arange(len(yt)), np.arange(len(yr)))
            bs = [f(rng.integers(0, len(yt), len(yt)), rng.integers(0, len(yr), len(yr))) for _ in range(B)]
            rows.append(dict(layer=L, width=wd, lo_nllc=a, hi_nllc=b, n_T=len(ti), n_R=len(ri), est=lab, ratio=p,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    O = pd.DataFrame(rows); O.to_csv(f"{OUT}/ml_overlap.csv", index=False)
    print("(i) hi_deva_dakshina / hi_latn_dakshina on the overlap of clean NLL:")
    print(O.round(3).to_string(index=False))


def part_width(w, B=1000):
    rng = np.random.default_rng(303)
    x = w[w.layer.isin([12, 18])]
    rows, boots = [], {}
    for (L, s), g in x.groupby(["layer", "set"]):
        a = g[g.width == "16k"].sort_values("win"); b = g[g.width == "65k"].sort_values("win")
        assert len(a) == len(b) and np.array_equal(a.nll_clean.to_numpy(), b.nll_clean.to_numpy())
        ca, cb = (a.nll_splice - a.nll_clean).to_numpy(), (b.nll_splice - b.nll_clean).to_numpy()
        fa, fb, den = a.fvu_num.to_numpy(), b.fvu_num.to_numpy(), a.fvu_den.to_numpy()
        cr = cb.sum() / ca.sum(); lf = np.log(fb.sum() / den.sum()) - np.log(fa.sum() / den.sum())
        bc, bl = np.empty(B), np.empty(B)
        for i in range(B):
            ix = rng.integers(0, len(a), len(a))
            bc[i] = cb[ix].sum() / ca[ix].sum(); bl[i] = np.log(fb[ix].sum() / fa[ix].sum())
        boots[(L, s)] = (bc, bl)
        rows.append(dict(layer=L, set=s, cost_ratio=cr, cost_lo=np.percentile(bc, 2.5), cost_hi=np.percentile(bc, 97.5),
                         dlogfvu=lf, dlogfvu_lo=np.percentile(bl, 2.5), dlogfvu_hi=np.percentile(bl, 97.5)))
    W = pd.DataFrame(rows); W.to_csv(f"{OUT}/ml_width.csv", index=False)
    print("(h) 65k/16k pooled splice-cost ratio and change in log pooled FVU, paired windows:")
    print(W.round(3).to_string(index=False))
    n_ex = 0; dd = []
    for L in (12, 18):
        eb = boots[(L, "en_pile")][1]; e0 = W[(W.layer == L) & (W.set == "en_pile")].dlogfvu.iloc[0]
        for s in NATIVE:
            nb = boots[(L, s)][1]; n0 = W[(W.layer == L) & (W.set == s)].dlogfvu.iloc[0]
            d = nb - eb; lo, hi = np.percentile(d, [2.5, 97.5]); n_ex += int(lo > 0 or hi < 0)
            dd.append((L, s, n0 - e0, lo, hi))
    print("native - English dlogFVU:", [(L, s, round(v, 3), round(lo, 3), round(hi, 3)) for L, s, v, lo, hi in dd])
    print(f"CI excludes 0 in {n_ex}/12")
    for L in (12, 18):
        n = W[(W.layer == L) & W.set.isin(NATIVE)].cost_ratio
        print(f"L{L}: English cost change {100 * (W[(W.layer == L) & (W.set == 'en_pile')].cost_ratio.iloc[0] - 1):.1f}%, "
              f"native {100 * (n.max() - 1):.1f}% to {100 * (n.min() - 1):.1f}%")


def rcs(x, k):
    K = len(k); p = lambda u: np.maximum(u, 0) ** 3; d = (k[-1] - k[0]) ** 2
    out = [x]
    for j in range(K - 2):
        out.append((p(x - k[j]) - p(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                    + p(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / d)
    return np.column_stack(out)


def design(w, P, E, langfe):
    cols = []
    for c in sorted(w.cell.unique()):
        m = (w.cell == c).to_numpy().astype(float); cols.append(m)
        for e in ([] if E == "none" else E.split("+")):
            cols.append(m * w[e].to_numpy())
        x = w.nllc.to_numpy()
        if P == "lin":
            cols.append(m * x)
        elif P.startswith("rcs"):
            K = int(P[3:]); kn = np.quantile(x[m > 0], np.linspace(0.05, 0.95, K)); Bm = rcs(x, kn)
            cols += [m * Bm[:, j] for j in range(Bm.shape[1])]
        elif P == "bins":
            b = np.floor(x / 0.25).astype(int)
            cols += [m * (b == v) for v in np.unique(b[m > 0])[1:]]
        if langfe:
            cols += [m * (w.lang == lg).to_numpy() for lg in ("ta", "bn", "en", "fr")]
    X = np.column_stack(cols + [w.native.to_numpy().astype(float)])
    return X[:, np.abs(X).sum(0) > 0]   # drops all-zero columns (e.g. a language absent from the sample); keeps intercepts


def ols_cr1(X, y, cl, G):
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ (X.T @ y); e = y - X @ b
    S = np.zeros((G, X.shape[1])); np.add.at(S, cl, X * e[:, None]); n, k = X.shape
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V))


def wild_p(X, y, cl, G):
    """restricted wild cluster bootstrap, Rademacher, all 2^G sign vectors; coefficient = last column (native)."""
    b, se = ols_cr1(X, y, cl, G); t0 = b[-1] / se[-1]
    Xr = X[:, :-1]; br = np.linalg.lstsq(Xr, y, rcond=None)[0]; fr = Xr @ br; er = y - fr
    XtXi = np.linalg.pinv(X.T @ X); H = XtXi @ X.T            # k x n
    n, k = X.shape; corr = (G / (G - 1)) * ((n - 1) / (n - k))
    signs = (((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2 - 1).astype(float)
    Hy0, Her = H @ fr, None
    # b(s) = H (fr + er*s[cl]) = Hy0 + sum_g s_g * H[:, cl==g] er[cl==g]
    Hg = np.column_stack([H[:, cl == g] @ er[cl == g] for g in range(G)])          # k x G
    ts = np.empty(len(signs))
    # residual-score per cluster for CR1: S_g = X_g' e(s)_g, e(s) = y(s) - X b(s)
    XgX = [X[cl == g].T @ X[cl == g] for g in range(G)]
    Xgfr = [X[cl == g].T @ fr[cl == g] for g in range(G)]
    Xger = [X[cl == g].T @ er[cl == g] for g in range(G)]
    for i, s in enumerate(signs):
        bs = Hy0 + Hg @ s
        S = np.stack([Xgfr[g] + s[g] * Xger[g] - XgX[g] @ bs for g in range(G)])
        V = XtXi[-1] @ (S.T @ S) @ XtXi[-1] * corr
        ts[i] = bs[-1] / np.sqrt(V)
    return float(np.mean(np.abs(ts) >= abs(t0)))


def part_regress(w, do_p=True):
    x = w[w.layer.isin([12, 18])].copy()
    x["cell"] = x.layer.astype(str) + "_" + x.width
    x["lang"] = x.set.map(LANG)
    x["stratum"] = x.cell + "|" + np.floor(x.nllc / 0.25).astype(int).astype(str) + "|" + np.floor(x.lf / 0.1).astype(int).astype(str)
    x["support"] = x.stratum.map(x.groupby("stratum").native.nunique() == 2)
    x = x[x.cost > 0].reset_index(drop=True)
    x["y"] = np.log(x.cost)
    rows = []
    specs = [("none", "none")] + [(P, E) for P in ("lin", "rcs4", "rcs6", "bins") for E in ("lf", "le", "lr", "lf+le")]
    if "lr" not in x:
        x["lr"] = np.log(x.fvu_num / x.pos)
    for P, E in specs:
        for lfe in (False, True):
            for samp in ("all", "support"):
                if samp == "support" and (P, E) == ("none", "none"):
                    continue
                d = x if samp == "all" else x[x.support].reset_index(drop=True)
                cl = pd.factorize(d.set)[0]; G = cl.max() + 1
                X = design(d, P, E, lfe); b, se = ols_cr1(X, d.y.to_numpy(), cl, G)
                p = wild_p(X, d.y.to_numpy(), cl, G) if (do_p and samp == "all") else np.nan
                rows.append(dict(P=P, E=E, langFE=lfe, sample=samp, ratio=np.exp(b[-1]), cr1_lo=np.exp(b[-1] - 2.201 * se[-1]),
                                 cr1_hi=np.exp(b[-1] + 2.201 * se[-1]), wild_p=p, n=len(d)))
                print(rows[-1], flush=True)
    # leave English + French out, rcs4 + lf, no language FE
    for samp in ("all", "support"):
        d = x[~x.set.isin(["en_pile", "fr_web"])]
        d = (d if samp == "all" else d[d.support]).reset_index(drop=True)
        cl = pd.factorize(d.set)[0]; G = cl.max() + 1
        b, se = ols_cr1(design(d, "rcs4", "lf", False), d.y.to_numpy(), cl, G)
        rows.append(dict(P="rcs4", E="lf", langFE=False, sample=samp + "_drop_en_fr", ratio=np.exp(b[-1]), n=len(d)))
        print(rows[-1], flush=True)
    # per cell (separate fits), rcs4 + le, with and without language FE: is the matched excess larger at L12?
    for c, d in x.groupby("cell"):
        d = d.reset_index(drop=True); cl = pd.factorize(d.set)[0]; G = cl.max() + 1
        for lfe in (False, True):
            b, se = ols_cr1(design(d, "rcs4", "le", lfe), d.y.to_numpy(), cl, G)
            rows.append(dict(P="rcs4", E="le", langFE=lfe, sample=f"cell_{c}", ratio=np.exp(b[-1]), n=len(d)))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/ml_regress.csv", index=False)
    g = R[(R.P != "none") & (R["sample"] == "all")]
    for lfe in (False, True):
        h = g[g.langFE == lfe]
        print(f"langFE={lfe}: 16 specs ratio {h.ratio.min():.2f}-{h.ratio.max():.2f}, wild p<0.05 in {(h.wild_p < 0.05).sum()}/{len(h)}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    w = load()
    print("windows per (layer,width):", w.groupby(["layer", "width"]).size().to_dict())
    if part in ("match", "all"):
        part_match(w)
    if part in ("overlap", "all"):
        part_overlap(w)
    if part in ("width", "all"):
        part_width(w)
    if part in ("regress", "all"):
        part_regress(w)

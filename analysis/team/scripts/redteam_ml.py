"""RED TEAM (exploratory, post hoc, 2026-11-19). Attack on claim A: "script, not error size, sets the cost of SAE error".

Reads raw E12/E21 per-window arrays (16k and 65k), recomputes the native/romanized Dakshina ratios under alternative
normalisations, checks the "same sentences" alignment, and does predictability-matched comparisons.
Writes team/scripts/out/redteam_ml_*.csv. Run: OMP_NUM_THREADS=2 python redteam_ml.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path("/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad")
LANG = SP / "hw" / "data" / "lang"
OUT = SP / "team" / "scripts" / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
NB = 2000


def tokenizer():
    from tokenizers import Tokenizer
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/tokenizer.json"))[0]
    return Tokenizer.from_file(f)


def load():
    man = json.loads((LANG / "MANIFEST.json").read_text())
    tok = tokenizer()
    feats = {}
    for name in man:
        d = json.loads((LANG / man[name]["file"]).read_text())
        skip = int(d["skip_positions"])
        rows = []
        for ids in d["ids"]:
            sc = ids[skip:]
            txt = tok.decode(list(sc), skip_special_tokens=True)
            pieces = [tok.id_to_token(i) or "" for i in sc]
            rows.append({"words": len(txt.split()), "chars": len(txt),
                         "byte_frac": float(np.mean([p.startswith("<0x") for p in pieces]))})
        feats[name] = pd.DataFrame(rows)
    out = []
    for f in sorted(SP.glob("ana_runs/E12/*/per_window_*_L*.npz")):
        stem = f.stem[len("per_window_"):]
        name, L = stem.rsplit("_L", 1)
        m = json.loads((f.parent / "manifest.json").read_text())
        z = np.load(f)
        df = pd.DataFrame({k: z[k] for k in z.files if z[k].ndim == 1})
        df["set"], df["layer"], df["width"], df["window"] = name, int(L), m["config"]["width"], np.arange(len(df))
        df = pd.concat([df, feats[name]], axis=1)
        out.append(df)
    w = pd.concat(out, ignore_index=True)
    p = w.pred
    for k in ("clean", "splice", "mean", "zero"):
        w["ce_" + k] = w["nll_" + k] / p
    w["ce_delta"] = w.ce_splice - w.ce_clean
    return w, man


COLS = ["pred", "nll_clean", "nll_splice", "nll_mean", "nll_zero", "words", "chars", "fvu_num", "fvu_den", "pos",
        "eps_ratio_sum", "l0_sum"]


def agg(g):
    """Set-level statistics under alternative normalisations (ratio of sums, like harness/fidelity.summarize).
    g: DataFrame or dict of column sums."""
    t = {k: float(np.sum(g[k])) for k in COLS} if not isinstance(g, dict) else g
    P = t["pred"]
    c, s, m, z = (t["nll_" + k] / P for k in ("clean", "splice", "mean", "zero"))
    dl = s - c
    return {"ce_clean": c, "ce_delta": dl, "delta_per_word": dl * P / t["words"], "delta_per_char": dl * P / t["chars"],
            "rel_clean": dl / c, "one_minus_rec_mean": dl / (m - c), "one_minus_rec_zero": dl / (z - c),
            "fvu": t["fvu_num"] / t["fvu_den"], "eps2_per_pos": t["fvu_num"] / t["pos"],
            "var_per_pos": t["fvu_den"] / t["pos"], "eps_x": t["eps_ratio_sum"] / t["pos"], "l0": t["l0_sum"] / t["pos"]}


def boot_ratios(a, b, keys, nb=NB):
    A, B = a[COLS].to_numpy(float), b[COLS].to_numpy(float)
    pa, pb = agg(a), agg(b)
    res = {k: [] for k in keys}
    for _ in range(nb):
        ta = dict(zip(COLS, A[rng.integers(0, len(A), len(A))].sum(0)))
        tb = dict(zip(COLS, B[rng.integers(0, len(B), len(B))].sum(0)))
        qa, qb = agg(ta), agg(tb)
        for k in keys:
            res[k].append(qa[k] / qb[k])
    return {k: (pa[k] / pb[k], np.percentile(res[k], 2.5), np.percentile(res[k], 97.5)) for k in keys}


def main():
    w, man = load()
    w.to_parquet(OUT / "redteam_windows.parquet")
    pairs = {"hi": ("hi_deva_dakshina", "hi_latn_dakshina"), "ta": ("ta_taml_dakshina", "ta_latn_dakshina"),
             "bn": ("bn_beng_dakshina", "bn_latn_dakshina")}
    # 1. ratios under alternative normalisations; also restricted to romanized windows that cover (about) the same
    #    sentences as the native set (both are prefixes of the same shuffled sentence order; the set with fewer words per
    #    248k tokens covers a prefix of the other's sentences).
    rows = []
    keys = ["ce_delta", "delta_per_word", "delta_per_char", "rel_clean", "one_minus_rec_mean", "one_minus_rec_zero",
            "eps2_per_pos", "var_per_pos", "fvu", "eps_x"]
    for lang, (nat, rom) in pairs.items():
        wn, wr = man[nat]["words"], man[rom]["words"]
        for (L, width), g in w.groupby(["layer", "width"]):
            a, b = g[g.set == nat], g[g.set == rom]
            if not len(a) or not len(b):
                continue
            for restrict in (False, True):
                bb, aa = b, a
                if restrict:          # keep the shorter-coverage set whole, cut the other to the same word count
                    if wn < wr:
                        bb = b[b.words.cumsum() <= wn]
                    else:
                        aa = a[a.words.cumsum() <= wr]
                row = {"lang": lang, "layer": L, "width": width, "restrict_same_sentences": restrict,
                       "n_nat": len(aa), "n_rom": len(bb)}
                br = boot_ratios(aa, bb, keys)
                for k in keys:
                    row[k], row[k + "_lo"], row[k + "_hi"] = br[k]
                rows.append(row)
                print(lang, L, width, restrict, {k: round(row[k], 3) for k in keys})
    pd.DataFrame(rows).to_csv(OUT / "redteam_ml_ratios.csv", index=False)

    # 2. predictability-matched comparison: windows binned by clean CE (0.5-nat bins); native vs romanized splice cost
    #    in bins where both have >= 20 windows; also within-set slope of ce_delta on ce_clean.
    mrows = []
    for lang, (nat, rom) in pairs.items():
        for (L, width), g in w.groupby(["layer", "width"]):
            a, b = g[g.set == nat], g[g.set == rom]
            if not len(a) or not len(b):
                continue
            sl = {}
            for nm, x in (("nat", a), ("rom", b)):
                X = np.c_[np.ones(len(x)), x.ce_clean]
                sl[nm] = np.linalg.lstsq(X, x.ce_delta, rcond=None)[0][1]
            bins = np.arange(0, 12.01, 0.5)
            a_b, b_b = pd.cut(a.ce_clean, bins), pd.cut(b.ce_clean, bins)
            diffs, ws = [], []
            for iv in a_b.cat.categories:
                na, nb_ = (a_b == iv).sum(), (b_b == iv).sum()
                if na >= 20 and nb_ >= 20:
                    da, db = a.ce_delta[a_b == iv].mean(), b.ce_delta[b_b == iv].mean()
                    mrows.append({"lang": lang, "layer": L, "width": width, "bin": str(iv), "n_nat": na, "n_rom": nb_,
                                  "delta_nat": da, "delta_rom": db, "ratio": da / db,
                                  "fvu_nat": a.fvu_num[a_b == iv].sum() / a.fvu_den[a_b == iv].sum(),
                                  "fvu_rom": b.fvu_num[b_b == iv].sum() / b.fvu_den[b_b == iv].sum()})
            ov = ((a.ce_clean > b.ce_clean.quantile(0.05)) & (a.ce_clean < b.ce_clean.quantile(0.95))).mean()
            print(lang, L, width, "slope nat %.3f rom %.3f" % (sl["nat"], sl["rom"]),
                  "frac native windows inside romanized 5-95%% clean range: %.3f" % ov,
                  "q nat", a.ce_clean.quantile([.05, .5, .95]).round(2).tolist(), "q rom", b.ce_clean.quantile([.05, .5, .95]).round(2).tolist())
            mrows.append({"lang": lang, "layer": L, "width": width, "bin": "SLOPES", "slope_nat": sl["nat"], "slope_rom": sl["rom"],
                          "frac_nat_in_rom_range": ov})
    pd.DataFrame(mrows).to_csv(OUT / "redteam_ml_matched.csv", index=False)
    print(pd.DataFrame([r for r in mrows if r["bin"] != "SLOPES"]).round(3).to_string())

    # 3. width (E21): does 65k remove the script gap? native/romanized ratio 16k vs 65k, and the native-script excess
    #    vs English per unit FVU.
    srows = []
    for (s, L, width), g in w.groupby(["set", "layer", "width"]):
        srows.append({"set": s, "layer": L, "width": width, **agg(g)})
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "redteam_ml_setlevel.csv", index=False)
    S["sens"] = S.ce_delta / S.fvu
    piv = S[S.layer.isin([12, 18])].pivot_table(index="set", columns=["layer", "width"], values=["ce_delta", "sens"])
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()

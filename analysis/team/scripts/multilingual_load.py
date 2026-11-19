"""EXPLORATORY, post hoc (2026-11-19). Multilingual role. Per-window E12/E21 table for L12 / L18 at both SAE widths.

Reads every $SP/ana_runs/E12/<hash>/per_window_<set>_L<layer>.npz; width and nominal L0 come from manifest config (width, l0).
Writes $SP/team/multilingual_out/windows.parquet: one row per (set, layer, width, window). Checks that the 16k and 65k runs score
identical windows (nll_clean, nll_mean, nll_zero, pred, fvu_den equal), so 16k-vs-65k contrasts can be paired per window.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path("/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad")
OUT = SP / "team" / "multilingual_out"
OUT.mkdir(parents=True, exist_ok=True)

NATIVE = {"hi_deva_dakshina", "hi_deva_web", "ta_taml_dakshina", "ta_web", "bn_beng_dakshina", "bn_web"}
ROMAN = {"hi_latn_dakshina", "ta_latn_dakshina", "bn_latn_dakshina"}
PAIRS = {"hi": ("hi_deva_dakshina", "hi_latn_dakshina"), "ta": ("ta_taml_dakshina", "ta_latn_dakshina"),
         "bn": ("bn_beng_dakshina", "bn_latn_dakshina")}


def group(s):
    return ("native" if s in NATIVE else "roman" if s in ROMAN else "hinglish" if s == "hinglish_phinc"
            else "fr" if s == "fr_web" else "en")


def main():
    rows = []
    for f in sorted(SP.glob("ana_runs/E12/*/per_window_*_L*.npz")):
        name, L = f.stem[len("per_window_"):].rsplit("_L", 1)
        if name == "en_pilot100" or int(L) not in (12, 18):
            continue
        cfg = json.loads((f.parent / "manifest.json").read_text())["config"]
        z = np.load(f)
        d = pd.DataFrame({k: z[k] for k in z.files})
        d["set"], d["layer"], d["width"], d["l0_nominal"] = name, int(L), cfg["width"], int(cfg["l0"][str(L)])
        d["run"], d["window"] = f.parent.name, np.arange(len(d))
        rows.append(d)
    w = pd.concat(rows, ignore_index=True)
    w["grp"] = w.set.map(group)
    p = w.pred
    w["ce_clean"] = w.nll_clean / p
    w["ce_delta"] = (w.nll_splice - w.nll_clean) / p
    w["zero_range"] = (w.nll_zero - w.nll_clean) / p
    w["mean_range"] = (w.nll_mean - w.nll_clean) / p
    w["fvu"] = w.fvu_num / w.fvu_den
    w["eps_x"] = w.eps_ratio_sum / w.pos
    w["l0"] = w.l0_sum / w.pos
    # pairing check: same windows at both widths
    bad = 0
    for (s, L), g in w.groupby(["set", "layer"]):
        a, b = g[g.width == "16k"].sort_values("window"), g[g.width == "65k"].sort_values("window")
        assert len(a) == len(b), (s, L, len(a), len(b))
        for k in ("nll_clean", "nll_mean", "nll_zero", "pred", "fvu_den"):
            if not np.allclose(a[k].to_numpy(), b[k].to_numpy(), rtol=1e-6):
                bad += 1
                print("MISMATCH", s, L, k)
    print("pairing mismatches:", bad)
    w.to_parquet(OUT / "windows.parquet")
    print(w.groupby(["set", "layer", "width"]).size().unstack())
    print(w.groupby(["layer", "width"]).l0_nominal.first())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()

"""STATS (exploratory). Diagnostic for the within-language overlap: what are the native-script windows that are as unpredictable
as romanized text, and the romanized windows as predictable as native text? Script composition per window (share of letters in
the set's native script vs Latin letters, digits share) from the stored token ids, decoded with the Gemma tokenizer (offline).
Also: does window-level splice cost depend on the Latin-letter share WITHIN native sets at fixed predictability?
Output: team/stats_out/ms_script_mix.csv (+ printed summary)
"""
import json
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, ".layout/team/scripts")
from stats_common import OUT, SP, e12_windows  # noqa: E402

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", local_files_only=True)
SKIP = 4
SCRIPT = {"hi": "DEVANAGARI", "ta": "TAMIL", "bn": "BENGALI"}


def comp(text, native):
    nat = lat = dig = let = 0
    for ch in text:
        if ch.isdigit():
            dig += 1
        if not ch.isalpha():
            continue
        let += 1
        nm = unicodedata.name(ch, "")
        if native and nm.startswith(native):
            nat += 1
        elif nm.startswith("LATIN"):
            lat += 1
    return nat / max(let, 1), lat / max(let, 1), dig / max(len(text), 1)


rows = []
for s in ["hi_deva_dakshina", "hi_deva_web", "hi_latn_dakshina", "hinglish_phinc", "ta_taml_dakshina", "ta_web", "bn_beng_dakshina",
          "bn_web"]:
    ids = json.load(open(f"{SP}/hw/data/lang/{s}.json"))["ids"]
    native = SCRIPT.get(s[:2]) if ("deva" in s or "taml" in s or "beng" in s or s.endswith("_web")) else None
    for i, x in enumerate(ids):
        txt = tok.decode(list(x[SKIP:]), skip_special_tokens=True)
        a, b, c = comp(txt, native)
        rows.append({"set": s, "win": i, "native_letter_share": a, "latin_letter_share": b, "digit_share": c})
M = pd.DataFrame(rows)
w = e12_windows()
w = w[(w.width == "16k") & (w.layer.isin([12, 18]))]
D = w.merge(M, on=["set", "win"], how="inner")
D.to_csv(f"{OUT}/ms_script_mix.csv", index=False)
pd.set_option("display.width", 250)
# overlap region for Hindi Dakshina
lo, hi = 3.579, 4.445
D["in_overlap"] = (D.nllc >= lo) & (D.nllc <= hi)
print("Script composition by set (L12 rows), all windows vs windows in the hi overlap interval [3.579, 4.445] nats/token:")
print(D[D.layer == 12].groupby(["set", "in_overlap"])[["native_letter_share", "latin_letter_share", "digit_share", "nllc"]]
      .mean().round(3).to_string())
# within native sets: cost ~ latin share at fixed nllc (OLS, log cost), per layer
from numpy.linalg import lstsq  # noqa: E402
print("\nWithin native-script sets: OLS log(cost) ~ nllc + log eps + latin_letter_share (per set, layer); coefficient on latin share")
for (s, L), g in D[D.set.isin(["hi_deva_dakshina", "hi_deva_web", "ta_taml_dakshina", "ta_web", "bn_beng_dakshina", "bn_web"])
                   & (D.cost > 0)].groupby(["set", "layer"]):
    X = np.column_stack([np.ones(len(g)), g.nllc, np.log(g.eps), g.latin_letter_share])
    y = np.log(g.cost.to_numpy())
    b = lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0)
    bs = [lstsq(X[ix], y[ix], rcond=None)[0][3] for ix in (rng.integers(0, len(g), len(g)) for _ in range(500))]
    print(f"  {s:18s} L{L}: latin share mean {g.latin_letter_share.mean():.3f}, sd {g.latin_letter_share.std():.3f}; "
          f"coef {b[3]:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (log cost per unit share)")

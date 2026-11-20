"""Cross-template (E3 fit-fresh vs fit-pilot) and cross-language (E14 native vs English fit) stability of error directions."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geo_lib import *
print("== E3: fit on pilot templates vs fit on fresh template families (same layer, same SAE)")
for t, L, s in [("add_chain", 18, "16k_l074"), ("entity_track", 12, "16k_l082")]:
    a = load("E3", f"{t}_L{L}_L{L}_{s}__fit-pilot")[0]; b = load("E3", f"{t}_L{L}_L{L}_{s}__fit-fresh")[0]
    for est in ["pca_pooled", "answer_sep"]:
        print(t, est, {r: round(top(a[est][:, :r], b[est][:, :r]), 3) for r in (1, 2, 4, 8)},
              "capture r8", round(capture(a[est][:, :8], b[est][:, :8]), 3))
print("== E14: native-fit vs English-fit error directions (same layer); Haar top-cos p99 r1 .052 r4 .097; capture of random 4 in 4: 4/2304=.0017")
for t, L, s, en, langs in [("add_chain", 18, "16k_l074", "rs_en", ["rs_fr", "rs_hi", "rs_hinglish"]),
                            ("entity_track", 12, "16k_l082", "ot_en", ["ot_hi_latn"]),
                            ("people_cities", 12, "16k_l082", "pc_en", ["pc_fr", "pc_hi", "pc_hi_latn", "pc_hinglish"])]:
    E = load("E14", f"{t}_L{L}_L{L}_{s}__fit-{en}")[0]
    for lg in langs:
        X = load("E14", f"{t}_L{L}_L{L}_{s}__fit-{lg}")
        if X is None: print(t, lg, "missing"); continue
        X = X[0]
        for est in ["pca_pooled", "answer_sep"]:
            print(f"{t:14s} {lg:12s} {est:10s}", {r: round(top(E[est][:, :r], X[est][:, :r]), 3) for r in (1, 2, 4, 8)},
                  "cap r4", round(capture(E[est][:, :4], X[est][:, :4]), 3))
# English pilot vs English e13 templates (different template set) as the reference for 'template change only'
print("== English: E14 (e13 templates) vs pilot templates (E9/E10/E11)")
for t, L, s, en, src in [("add_chain", 18, "16k_l074", "rs_en", ("E9", "add_chain_L18_L18_16k_l074")),
                          ("entity_track", 12, "16k_l082", "ot_en", ("E9", "entity_track_L12_L12_16k_l082")),
                          ("people_cities", 12, "16k_l082", "pc_en", ("E10", "people_cities_L12_L12_16k_l082__fit-pc_en"))]:
    E = load("E14", f"{t}_L{L}_L{L}_{s}__fit-{en}")[0]; P = load(*src)[0]
    for est in ["pca_pooled", "answer_sep"]:
        print(f"{t:14s} {est:10s}", {r: round(top(E[est][:, :r], P[est][:, :r]), 3) for r in (1, 2, 4, 8)})

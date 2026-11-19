"""E21 standard readout: M1 (addendum B) at 65k vs 16k. M1 supported for (set, layer, SAE) iff recovered_mean upper CI < en_pile's
lower CI at the same layer AND the same SAE (non-overlapping 95% bootstrap CIs over windows); reported for mean (primary) and zero
ablation. Also ce_delta (splice cost, nats/token) and recovered at 16k vs 65k. Input: r5/tables/E12_fidelity_lang.csv.
CIs only (no p-values are reported, so no BH)."""
from standard_common import *

t = pd.read_csv(R5 / "E12_fidelity_lang.csv")
t = t[t.layer.isin([12, 18])]
t["width"] = t.sae.str.split("_").str[1]
rows = []
for (layer, sae), g in t.groupby(["layer", "sae"]):
    en = g[g.set == "en_pile"].iloc[0]
    for _, r in g.iterrows():
        rows.append({"set": r.set, "layer": layer, "sae": sae, "width": r.width, "n_windows": r.n_windows, "l0": r.l0,
                     "rec_mean": r.recovered_mean, "rec_mean_lo": r.recovered_mean_lo, "rec_mean_hi": r.recovered_mean_hi,
                     "M1_mean": (r.recovered_mean_hi < en.recovered_mean_lo) if r.set != "en_pile" else np.nan,
                     "rec_zero": r.recovered_zero, "M1_zero": (r.recovered_zero_hi < en.recovered_zero_lo) if r.set != "en_pile" else np.nan,
                     "minus_en_mean": r.recovered_mean - en.recovered_mean,
                     "ce_delta": r.ce_delta, "ce_delta_lo": r.ce_delta_lo, "ce_delta_hi": r.ce_delta_hi,
                     "ce_delta_above_en": (r.ce_delta_lo > en.ce_delta_hi) if r.set != "en_pile" else np.nan,
                     "fvu": r.fvu, "eps_over_x": r.eps_over_x, "shipped_flag": r.recovered_mean_below_en_ci})
R = pd.DataFrame(rows)
chk = R[(R.set != "en_pile") & R.shipped_flag.notna()]
print("recheck vs shipped recovered_mean_below_en_ci:", int((chk.M1_mean.astype(bool) != chk.shipped_flag.astype(str).eq("True")).sum()), "disagreements of", len(chk))
R.to_csv(OUT / "E21_m1.csv", index=False)

w = R.pivot_table(index=["set", "layer"], columns="width",
                  values=["rec_mean", "M1_mean", "M1_zero", "ce_delta", "minus_en_mean", "fvu", "eps_over_x", "l0"], aggfunc="first")
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index()
w["d_rec_65k_minus_16k"] = w.rec_mean_65k - w.rec_mean_16k
w["d_ce_delta_65k_minus_16k"] = w.ce_delta_65k - w.ce_delta_16k
w["d_minus_en_65k_vs_16k"] = w.minus_en_mean_65k - w.minus_en_mean_16k
w.to_csv(OUT / "E21_16k_vs_65k.csv", index=False)
print(w[["set", "layer", "l0_16k", "l0_65k", "rec_mean_16k", "rec_mean_65k", "d_rec_65k_minus_16k", "M1_mean_16k", "M1_mean_65k",
         "M1_zero_16k", "M1_zero_65k", "minus_en_mean_16k", "minus_en_mean_65k", "ce_delta_16k", "ce_delta_65k",
         "d_ce_delta_65k_minus_16k", "fvu_16k", "fvu_65k"]].round(4).to_string())
for layer in (12, 18):
    s = w[(w.layer == layer) & ~w.set.isin(["en_pile", "en_pilot100"])]   # the 11 non-English language sets
    print(f"L{layer}: M1(mean) supported 16k {int(s.M1_mean_16k.astype(bool).sum())}/{len(s)}, 65k {int(s.M1_mean_65k.astype(bool).sum())}/{len(s)};"
          f" M1(zero) 16k {int(s.M1_zero_16k.astype(bool).sum())}, 65k {int(s.M1_zero_65k.astype(bool).sum())};"
          f" rec_mean higher at 65k for {int((s.d_rec_65k_minus_16k > 0).sum())}/{len(s)} non-English sets;"
          f" ce_delta lower at 65k for {int((s.d_ce_delta_65k_minus_16k < 0).sum())}/{len(s)}; gap to English smaller (less negative) at 65k for"
          f" {int((s.d_minus_en_65k_vs_16k > 0).sum())}/{len(s)}")
    e = w[(w.layer == layer) & (w.set == "en_pile")].iloc[0]
    print(f"   en_pile: rec 16k {e.rec_mean_16k:.4f} 65k {e.rec_mean_65k:.4f}; ce_delta 16k {e.ce_delta_16k:.4f} 65k {e.ce_delta_65k:.4f}")
# mixed-layer outcomes per set (M1 mean)
for width in ("16k", "65k"):
    m = w.pivot_table(index="set", columns="layer", values=f"M1_mean_{width}", aggfunc="first")
    mixed = m[m.nunique(axis=1) > 1]
    print(f"{width} sets mixed across layers (M1 mean): {list(mixed.index)}")

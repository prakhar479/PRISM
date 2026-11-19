"""E19 standard readout: steering dose-response (s = 0.25..16) per cell and estimator vs the matched-norm random p95, E10's Q2
rule (flip >= .50 AND random p95 <= .05), and the post hoc question "any scale where U beats random by >= 10 pts" (reported with
the effective-style rule: flip >= p95 + .10 AND flip_lo > p95, plus an exact binomial test H0 flip <= p95 + .10, BH within E19).
Inputs: r5/tables/E19_steering.csv, hw/tables/final/E10_steering.csv. Outputs: team/standard_out/E19_*.csv."""
from standard_common import *

t = pd.read_csv(R5 / "E19_steering.csv")
t["cell"] = t.task.str.replace("add_chain", "RS").str.replace("entity_track", "OT").str.replace("people_cities", "PC") + " L" + t.layer.astype(str)
t["s"] = t.condition.str.replace("steer_x", "").astype(float)
t["est"] = t.estimator + np.where(t.estimator.str.startswith("das"), "", "_r" + t.r.astype(str))
t = add_p_eff(t)
t["q_bh"] = bh(t.p_eff)
print(f"E19 tests: {t.p_eff.notna().sum()} binomial tests (BH over all); q<.05: {(t.q_bh < .05).sum()}")
t["beats10"] = (t.flip_rate >= t.random_p95 + EFFECT_PTS) & (t.flip_rate_lo > t.random_p95)
# Q2 recheck
q2 = (t.flip_rate >= .5) & (t.random_p95 <= .05)
print("Q2 recheck disagreements:", int((q2 != t.q2_both.astype(bool)).sum()), "of", len(t), "; q2_both true:", int(t.q2_both.astype(bool).sum()))
t.to_csv(OUT / "E19_rows.csv", index=False)

for flt in FILTERS:
    print(f"\n==== {flt}: flip [lo] / random p95 ('+' = beats random by >=10 pts with lo > p95; 'Q' = Q2 both)")
    s = t[t["filter"] == flt].copy()
    s["v"] = [f"{f:.3f}[{lo:.2f}]/{p:.3f}{'+' if b else ''}{'Q' if q else ''}" for f, lo, p, b, q in
              zip(s.flip_rate, s.flip_rate_lo, s.random_p95, s.beats10, s.q2_both.astype(bool))]
    print(s.pivot_table(index=["cell", "est"], columns="s", values="v", aggfunc="first").to_string())

print("\n==== rows that beat random by >= 10 pts (all filters)")
b = t[t.beats10].sort_values(["cell", "filter", "est", "s"])
print(b[["cell", "filter", "est", "s", "flip_n_eligible", "flip_rate", "flip_rate_lo", "flip_rate_hi", "random_p95", "q_bh", "q2_flip_ge_50", "q2_random_p95_le_5", "q2_both"]].round(3).to_string())

print("\n==== Q2 columns: count of rows with flip >= .5 and with random p95 <= .05, by cell (f_pilot)")
s = t[t["filter"] == "f_pilot"]
print(s.groupby("cell").agg(rows=("s", "size"), flip_ge_50=("q2_flip_ge_50", "sum"), p95_le_5=("q2_random_p95_le_5", "sum"), both=("q2_both", "sum")).to_string())
print(s.groupby(["cell", "s"]).random_p95.agg(["min", "max"]).round(3).unstack(0).to_string())

# filter disagreement
piv = t.pivot_table(index=["cell", "est", "s"], columns="filter", values="beats10", aggfunc="first")
print("\n==== filter disagreements on beats10:")
print(piv[piv.nunique(axis=1) > 1].to_string())

# replication of E10 at shared settings
e10 = pd.read_csv(FINAL / "E10_steering.csv")
k = ["task", "layer", "filter", "condition", "estimator", "r"]
m = e10.merge(t, on=k, suffixes=("_E10", "_E19"))
print(f"\n==== E10 vs E19 shared rows: {len(m)}; max|flip diff| = {np.abs(m.flip_rate_E10 - m.flip_rate_E19).max():.3g}; "
      f"max|random p95 diff| = {np.abs(m.random_p95_E10 - m.random_p95_E19).max():.3g}")
dd = m[np.abs(m.flip_rate_E10 - m.flip_rate_E19) > 1e-9]
print(dd[k + ["flip_rate_E10", "flip_rate_E19", "random_p95_E10", "random_p95_E19"]].round(3).to_string())

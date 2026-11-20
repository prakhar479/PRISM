# PRISM E17–E22: the standard-methods readout (role `standard`, 2026-11-19)

Everything here is **exploratory and post hoc**. E17–E22 were designed after the final-pass results were known. The decision
rules are the frozen ones from the addendum (section E, constants in `harness/final.py`), applied the same way the addendum applies
them to E9/E10. Nothing here is confirmatory. E23 (norm-matched noise controls) was **not run**: the server ran the older tag. So
nothing below rules out "any perturbation of this norm would do the same" except where a random-U null already does that.

Provenance: the new runs are on node-D, commit 240e5f6 (E9 ran on node-C, commit 14e7e1f). Tables come from `$SP/r5/tables`
(E17–E22) and `$SP/hw/tables/final` (E9, E10, E13, E16 baselines). Paired tests use the per-pair parquet files in
`$SP/ana_runs/E*/`. Scripts: `$SP/team/scripts/standard_{common,e17,e18,e19,e20,e21,e22}.py`. Outputs (CSV) are in
`$SP/team/standard_out/`, and the printed logs are in `$SP/team/standard_out_e1{7,8,9}.log` and `standard_out_e2{0,1,2}.log`.

## 0. Method (what "by the rules" means here)

* **Rules**, per (cell, filter, estimator, rank), with flip rate as the metric:
  * **effective**: flip ≥ random-U p95 + 10 pts AND lower 95% CI > p95.
  * **illusion**: full_along_U is effective AND full − feat ≥ 10 pts with a paired CI lower bound > 0.
  * **feature-mediated**: feat_A_along_U is effective.
  * **DAS fit valid**: train interchange accuracy ≥ 0.5. For an invalid fit every rule outcome is NA ("failed fit, not
    evidence").
  * **aligned**: top principal cosine ≥ 0.5. **unrelated**: at or below the Haar p99.
  * **Q2**: steer flip ≥ 50% AND matched-norm random p95 ≤ 5%.
  * **M1**: the language's CE-recovered (mean ablation) upper CI is below en_pile's lower CI, at the same layer and with the
    same SAE.
* **Sanity checks.** I recomputed every shipped `effective` flag from its own columns: 0 disagreements in E17 (60 evaluable
  rows), E18 (594), E20 (270) and E22 (234). E19 Q2: 0 of 567. E21 M1: 0 of 48. I recomputed the flip rates from the per-pair
  files and they match the tables to within 1e-16 (E17 156 rows, E18 456, E22 144).
* **Rule outcomes are not mutually exclusive.** A row can be both "illusion" and "feature-mediated", and every valid RS DAS row
  is. I report the flags as the rules define them, not as categories.
* **p-values (post hoc, added for multiplicity; not part of the frozen rules).**
  * `p_eff`: one-sided exact binomial test, H0: flip ≤ p95 + 0.10, using flips among eligible pairs.
  * `p_ill`: one-sided exact McNemar on discordant pairs, full_along_U vs feat_A_along_U.
  * E22 also gets a two-sided exact McNemar for English-fit vs native-fit flips on the same pairs.
  * BH (q = 0.05) is applied within each experiment over every test that experiment's script computes, across all three
    filters:

  | experiment | tests | BH rejections (q < .05) |
  |---|---|---|
  | E17 | 234 | 175 |
  | E18 | 912 | 246 |
  | E19 | 567 | 82 |
  | E20 | 360 | 145 |
  | E22 | 648 | 192 |

  E21 reports CIs only (no p-values).

  The rule "effective" is weaker than the margin test: it needs only the point estimate ≥ p95 + 10 and lo > p95. Several
  rule-effective rows do not survive the margin test after BH, and I flag them where it matters.
* **Filters.** f_pilot (the fit filter) is reported first, then f_A_correct, then f_none. Unless a line says otherwise, a rule
  outcome holds under all three filters. Every filter disagreement is listed.

## 1. Findings ranked by strength

1. **[FACT, strongest] E9's DAS rotations reproduce bitwise across machines.**
   * E17 retrained das1/2/4/8 (seed 0, 3 epochs, lr 1e-2) on node-D; the logs show retraining, ~6 min per fit.
   * The retrained U matrices are bitwise equal to E9's (max |diff| = 0 for das1/2/4/8 and pca_pooled).
   * Train accuracy and losses are identical. All 39,560 shared per-pair rows have identical logit_A/logit_B (max |diff| 0) and
     100% top-1 agreement.
   * All 54 shared table rows have identical flip, CI, NLDR, p95 and illusion diff, with 0 rule-flag disagreements.
   * Both manifests record device = cpu and torch 2.14.1 (cu130 vs cu126 builds).
   * E18 also reproduces E9 exactly at the 252 shared rows (flip |diff| = 0). E19 reproduces E10 at 135 shared rows (flip
     |diff| = 0).
   * Caveat: the random-U p95 is *not* identical between runs. The differences are E18 vs E9 ≤ 0.023 (OT) and E19 vs E10 ≤ 0.19
     (OT). The random draws differ, but no rule flag changed.
2. **[FACT] Longer training does not make rank-1/2 DAS valid on RS L18.** None of the 16 rank-1/2 variants reaches 0.5 train
   interchange accuracy (best: das2_e30, 0.473). Every rank-4/8 variant (6 of 6) is valid, effective, feature-mediated and an
   illusion under all three filters.
3. **[FACT] DAS rotations are not seed-stable as subspaces, although their effect is qualitatively stable.**
   * Different-seed pairs at rank 4/8: mean principal cosine 0.28–0.32, top cosine 0.645–0.729.
   * Same-seed pairs (different epochs): top cosine 0.99.
   * The seed-1 fits flip fewer pairs (das4_s1 0.593 vs das4 0.760; das8_s1 0.733 vs 0.867).
4. **[FACT] Error-derived U (pca_pooled, answer_sep) on RS L18 becomes "effective" on the full residual only at r ≥ 32.** The
   flip rates are small (0.15–0.28), and under f_pilot pca r = 32 does not survive the margin test after BH (q = .094). By rule,
   the r = 32/64 rows are "illusion" (feat flips ≤ 0.067). They are never feature-mediated at any rank up to 64.
5. **[FACT] On OT L12, every estimator is effective on the full residual from r = 1.** This includes error-U: pca r1 0.910,
   p95 .045. Error-U is an illusion at every rank and becomes feature-mediated at r ≥ 4 (pca) or r ≥ 8 (answer_sep). DAS on OT
   is valid, effective and feature-mediated, but never an illusion.
6. **[FACT] People-cities (L12, L16):**
   * Error-U does nothing: full_along_U and feat_A_along_U flip 0.000 at every rank up to 64.
   * DAS fits are valid only at rank 16 (train acc 0.81 / 0.78). There they are effective, feature-mediated and an illusion,
     with large diffs (0.373 / 0.553).
   * Ranks 1–8 are failed fits (acc ≤ 0.437).
7. **[FACT] E19: Q2 is met by 0 of 567 rows.** RS never reaches 50% (max 0.220, das8 at s = 16). OT reaches ≥ 50% in 41 of 63
   f_pilot rows, but its random p95 is never ≤ 5% (0.112 at s = 0.25, up to 0.956). PC never exceeds 0.027. Only RS DAS rows
   ever beat random by ≥ 10 pts by rule (s = 4–16), and just one survives the margin test after BH (das4, s = 4, f_pilot,
   q = .034).
8. **[FACT] E20: wider SAEs at matched L0 change no effective or illusion outcome** (0 changes in 30 rows per SAE). Six
   feature-mediated flags at OT flip, all at the rule's margin. RS L18 65k: the DAS feat flips rise (das4 0.547 → 0.620) and the
   illusion diff shrinks (0.213 → 0.140), but it still passes.
9. **[FACT] E21: M1 is supported for all 11 non-English sets at both layers, at 16k and at 65k.** The 65k SAE raises CE-recovered
   for every set and lowers the splice cost (ce_delta) for every set, English included.
10. **[FACT] E22: no row is both readable (E13 gates) and evidential (valid fit).**
    * pc_fr is the only language that passes both gates, and every PC DAS fit is invalid (native and English).
    * On RS, the English-fit DAS transfers to fr/hi/hinglish at full_along_U flips of 0.52–0.61, effective by rule. Native
      das4 fits fail in all three languages.
    * The one native fit that beats English is rs_hi das8: 0.840 vs 0.620, q < .001. It is a language that fails Gate A.

## 2. E17 — DAS robustness, RS L18 16k (run 5a4a5fedea5f)

**Question.** Were E9's failed rank-1/2 DAS fits an optimisation failure? Are rotations seed-stable? Does E9 reproduce on a
different node?

**Table E17-1 (f_pilot, n = 150 eligible pairs).** All p95 values are random-U p95 (20 Haar draws). The p95 is 0.000 for every
row except full_along_U at rank 8 (0.000333).
* "ungated eff" is what the effective rule would say if the validity gate were ignored. It is reported for transparency and is
  **not evidence** by rule.
* q-values are BH within E17. q(eff) is the margin test for full / feat; q(ill) is the McNemar test.
* A q of 0.000 means q < 0.0005.

| variant | k | ep | lr | seed | train acc | valid | full [95% CI] | feat [95% CI] | full − feat [CI] | eff / feat-med / illusion | ungated eff full / feat | q(eff) full / feat | q(ill) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| das1 | 1 | 3 | 1e-2 | 0 | 0.250 | F | 0.273 [0.200, 0.347] | 0.120 [0.073, 0.173] | 0.153 [0.093, 0.213] | NA NA NA | T / T | 0.000 / 0.309 | 0.000 |
| das1_s1 | 1 | 3 | 1e-2 | 1 | 0.080 | F | 0.087 [0.047, 0.133] | 0.053 [0.020, 0.093] | 0.033 [0.007, 0.067] | NA NA NA | F / F | 0.899 / 1.000 | 0.042 |
| das1_s2 | 1 | 3 | 1e-2 | 2 | 0.333 | F | 0.307 [0.233, 0.380] | 0.187 [0.127, 0.253] | 0.120 [0.047, 0.193] | NA NA NA | T / T | 0.000 / 0.001 | 0.002 |
| das1_e10 | 1 | 10 | 1e-2 | 0 | 0.297 | F | 0.347 [0.273, 0.427] | 0.167 [0.107, 0.227] | 0.180 [0.113, 0.253] | NA NA NA | T / T | 0.000 / 0.011 | 0.000 |
| das1_e10_s1 | 1 | 10 | 1e-2 | 1 | 0.227 | F | 0.147 [0.093, 0.207] | 0.080 [0.040, 0.127] | 0.067 [0.020, 0.113] | NA NA NA | T / F | 0.058 / 0.995 | 0.009 |
| das1_e10_s2 | 1 | 10 | 1e-2 | 2 | 0.340 | F | 0.320 [0.247, 0.400] | 0.200 [0.140, 0.267] | 0.120 [0.047, 0.193] | NA NA NA | T / T | 0.000 / 0.000 | 0.002 |
| das1_e10_lr3e-2 | 1 | 10 | 3e-2 | 0 | 0.000 | F | 0.000 | 0.000 | 0.000 | NA NA NA | F / F | 1 / 1 | 1 |
| das1_e30 | 1 | 30 | 1e-2 | 0 | 0.300 | F | 0.300 [0.227, 0.373] | 0.100 [0.053, 0.153] | 0.200 [0.127, 0.273] | NA NA NA | T / T | 0.000 / 0.665 | 0.000 |
| das2 | 2 | 3 | 1e-2 | 0 | 0.333 | F | 0.347 [0.273, 0.420] | 0.200 [0.140, 0.267] | 0.147 [0.080, 0.220] | NA NA NA | T / T | 0.000 / 0.000 | 0.000 |
| das2_s1 | 2 | 3 | 1e-2 | 1 | 0.453 | F | 0.440 [0.360, 0.520] | 0.240 [0.173, 0.313] | 0.200 [0.140, 0.267] | NA NA NA | T / T | 0.000 / 0.000 | 0.000 |
| das2_s2 | 2 | 3 | 1e-2 | 2 | 0.343 | F | 0.313 [0.240, 0.393] | 0.160 [0.107, 0.220] | 0.153 [0.080, 0.233] | NA NA NA | T / T | 0.000 / 0.020 | 0.000 |
| das2_e10 | 2 | 10 | 1e-2 | 0 | 0.393 | F | 0.387 [0.307, 0.467] | 0.247 [0.180, 0.320] | 0.140 [0.067, 0.213] | NA NA NA | T / T | 0.000 / 0.000 | 0.000 |
| das2_e10_s1 | 2 | 10 | 1e-2 | 1 | 0.363 | F | 0.373 [0.300, 0.453] | 0.147 [0.093, 0.207] | 0.227 [0.160, 0.300] | NA NA NA | T / T | 0.000 / 0.058 | 0.000 |
| das2_e10_s2 | 2 | 10 | 1e-2 | 2 | 0.367 | F | 0.327 [0.253, 0.400] | 0.200 [0.140, 0.267] | 0.127 [0.060, 0.200] | NA NA NA | T / T | 0.000 / 0.000 | 0.001 |
| das2_e10_lr3e-2 | 2 | 10 | 3e-2 | 0 | 0.437 | F | 0.440 [0.360, 0.520] | 0.227 [0.160, 0.293] | 0.213 [0.133, 0.293] | NA NA NA | T / T | 0.000 / 0.000 | 0.000 |
| das2_e30 | 2 | 30 | 1e-2 | 0 | 0.473 | F | 0.500 [0.420, 0.580] | 0.287 [0.220, 0.360] | 0.213 [0.133, 0.293] | NA NA NA | T / T | 0.000 / 0.000 | 0.000 |
| das4 | 4 | 3 | 1e-2 | 0 | 0.733 | T | 0.760 [0.687, 0.827] | 0.547 [0.467, 0.627] | 0.213 [0.147, 0.287] | **T T T** | T / T | 0.000 / 0.000 | 0.000 |
| das4_s1 | 4 | 3 | 1e-2 | 1 | 0.647 | T | 0.593 [0.513, 0.673] | 0.307 [0.233, 0.380] | 0.287 [0.213, 0.360] | **T T T** | T / T | 0.000 / 0.000 | 0.000 |
| das4_e10 | 4 | 10 | 1e-2 | 0 | 0.840 | T | 0.733 [0.660, 0.807] | 0.540 [0.460, 0.620] | 0.193 [0.127, 0.267] | **T T T** | T / T | 0.000 / 0.000 | 0.000 |
| das8 | 8 | 3 | 1e-2 | 0 | 0.890 | T | 0.867 [0.807, 0.920] | 0.680 [0.600, 0.753] | 0.187 [0.113, 0.260] | **T T T** | T / T | 0.000 / 0.000 | 0.000 |
| das8_s1 | 8 | 3 | 1e-2 | 1 | 0.780 | T | 0.733 [0.660, 0.807] | 0.467 [0.387, 0.547] | 0.267 [0.193, 0.347] | **T T T** | T / T | 0.000 / 0.000 | 0.000 |
| das8_e10 | 8 | 10 | 1e-2 | 0 | 0.980 | T | 0.900 [0.847, 0.947] | 0.700 [0.627, 0.773] | 0.200 [0.127, 0.273] | **T T T** | T / T | 0.000 / 0.000 | 0.000 |
| pca_pooled r1/2/4/8 | – | – | – | – | – | – | 0.007 / 0.007 / 0.007 / 0.060 | 0 / 0 / 0 / 0 | ≤ 0.060 | F F F (all) | F | 1 | – |

**Does longer training make rank 1 or rank 2 valid? No (FACT).**
* Seed-0 train accuracy rises with epochs: das1 0.250 (3 epochs) → 0.297 (10) → 0.300 (30); das2 0.333 → 0.393 → 0.473.
* lr 3e-2 diverges at rank 1: acc 0.000, last-epoch loss 3.77 vs first 3.12. At rank 2 it gives 0.437.
* The seed spread at 3 epochs is about as large as the epoch effect: das1 0.080–0.333; das2 0.333–0.453.
* The best rank-2 fit (das2_e30) has acc 0.473 and full flip 0.500 [0.420, 0.580]. It sits just under the validity bar.
* INTERPRETATION: the rank-1/2 failure is a soft optimisation ceiling that rises slowly with training. It is not a hard
  impossibility, but no tested budget crosses the bar.
* Caveat: across the 22 DAS variants, the validity gate is almost a threshold on the outcome itself. Spearman(train acc,
  full_along_U flip on test) = 0.961 (p = 1.3e-12) and Spearman(train acc, feat flip) = 0.955. A variant with acc 0.473 (invalid)
  and one with 0.647 (valid) differ in full flip by 0.50 vs 0.59.

**The ungated view (not evidence by rule).** Every rank-1/2 fit with acc ≥ 0.25 moves 27–50% of pairs on the full residual,
against p95 = 0, with a positive paired full − feat difference. If the gate were ignored, rank 2 would be "effective" and
"illusion" in all 8 variants.

**Filters.** No filter disagreement on any E17 rule outcome. f_A_correct (n = 203) and f_none (n = 209) give the same flags. For
example, das4 under f_A_correct: full 0.749, feat 0.542, diff 0.207 [0.144, 0.273].

**Seed stability** (`E17_das_stability.csv`; the rule's alignment threshold, top cosine ≥ 0.5, applied post hoc to DAS–DAS
pairs):

| rank | pair kind | n pairs | top cos min / median / max | mean cos (median) | share "aligned" (top ≥ .5) | learned-part top cos (median) |
|---|---|---|---|---|---|---|
| 1 | different seed | 20 | 0.072 / 0.398 / 0.510 | 0.398 | 0.10 | 0.740 |
| 1 | same seed, different epochs or lr | 8 | 0.659 / 0.903 / 0.970 | 0.903 | 1.00 | 0.847 |
| 2 | different seed | 20 | 0.433 / 0.541 / 0.686 | 0.308 | 0.75 | 0.795 |
| 2 | same seed | 8 | 0.892 / 0.940 / 0.998 | 0.850 | 1.00 | 0.914 |
| 4 | different seed | 2 | 0.645 / 0.658 / 0.671 | 0.284 | 1.00 | 0.869 |
| 8 | different seed | 2 | 0.726 / 0.727 / 0.729 | 0.301 | 1.00 | 0.919 |

* FACT: by the top-cosine rule, different-seed rank-4/8 rotations count as "aligned". But their mean principal cosine is
  ≈ 0.3, so most of each subspace is not shared.
* Same-seed pairs have top cosine ≈ 0.99. This is the shared-initialisation inflation that the deep analysis already flagged.
* The learned component (with the initial span projected out) agrees more across seeds (0.855–0.919 at rank 4/8).
* INTERPRETATION: the DAS solution shares one or two dominant directions across seeds and fills the rest of the rank
  arbitrarily. The effect sizes (0.59–0.90) are seed-dependent, but the qualitative rule outcomes are not.

**Error-U vs DAS alignment** (top principal cosine of pca_pooled rank k with every DAS variant of rank k; `E17_alignment.csv`):
* All rank 2/4/8 variants are "partial": 0.20–0.40.
* Rank 1 ranges from "unrelated" (5 variants, 0.003–0.043) to "partial" (0.063–0.154).
* No variant is aligned.
* Methods note: the shipped `cos_top_das` column keeps one DAS variant per (estimator, rank) (`drop_duplicates` in
  make_final.py). In E17 it equals the seed-0 3-epoch rotation's value: 0.154, 0.263, 0.293, 0.395.

**Reproducibility vs E9 (`E17_vs_E9.csv`).** Bitwise identical; see finding 1. FACT: the E9 RS L18 DAS results are
deterministic given (data, seed, code), so cross-machine variation is zero. It also means E9's numbers carry no
machine-to-machine uncertainty, but this is *one* seed. The seed variation above (das4 full 0.593–0.760) is the relevant
uncertainty, and the bootstrap CIs in E9 do not include it.

## 3. E18 — rank curves to 64 and DAS 1–16 (run 69af834a1a59)

**Question.** Does error-U become effective at higher rank? Does PC DAS fit?

**Cells (f_pilot n eligible).** RS L18 150, OT L12 89, PC L12 150, PC L16 150. Baseline flips (f_pilot):

| cell | full_A | feat_A_errB | feat_A_meanPos |
|---|---|---|---|
| RS | 1.000 | 0.920 | 0.913 |
| OT | 1.000 | 1.000 | 1.000 |
| PC L12 | 1.000 | 0.927 | 0.873 |
| PC L16 | 1.000 | 0.813 | 0.940 |

**Error-U rank curves, f_pilot.** Each cell is flip [lo] / random p95; * = effective.

| cell / estimator / condition | r=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| RS pca full_along_U | .007/0 | .007/0 | .007/0 | .060[.027]/0 | .080[.040]/0 | **.153[.100]/0*** | **.247[.180]/0*** |
| RS pca feat_A_along_U | 0 | 0 | 0 | 0 | 0 | .040/0 | .053/.007 |
| RS answer_sep full_along_U | .033/0 | .027/0 | .067/0 | .060/0 | .080/0 | **.173[.113]/0*** | **.267[.200]/0*** |
| RS answer_sep feat_A_along_U | 0 | 0 | .007 | .007 | .007 | .047/0 | .067/.007 |
| RS pca repair | .913/.927 | .913/.920 | .927/.927 | .993/.927 | .987/.933 | .987/.927 | .993/.940 |
| OT pca full_along_U | **.910/.045*** | **.944/.034*** | **.944/.034*** | **.944/.068*** | **.944/.102*** | **.966/.170*** | **.989/.228*** |
| OT pca feat_A_along_U | .112/.023 | .090/.045 | **.157[.087]/.035*** | **.191/.047*** | **.303/.069*** | **.427/.135*** | **.742/.180*** |
| OT answer_sep full_along_U | **.865*** | **.865*** | **.865*** | **.955*** | **.966*** | **.966*** | **.989*** |
| OT answer_sep feat_A_along_U | .101 | .112 | .112 | **.202*** | **.292*** | **.427*** | **.708*** |
| OT repair (both) | 1.000/1.000 at every rank | | | | | | |
| PC L12, PC L16: full, feat (both estimators) | 0.000/0.000 at every rank | | | | | | |
| PC L16 pca repair | .807/.820 | .820/.814 | .827/.820 | .847/.820 | .887/.827 | .880/.827 | .880/.847 |

**Where does error-U become effective?** FACT, f_pilot:
* **RS L18:** full_along_U at r ≥ 32 for both estimators. By rule it is an *illusion* at r = 32 and 64:
  * pca diff 0.113 [0.060, 0.173] at r32 and 0.193 [0.127, 0.260] at r64;
  * answer_sep diff 0.127 at r32 and 0.200 at r64.

  It is never feature-mediated: max feat flip 0.067. After BH, the margin test is significant for pca r64 and answer_sep r32/64
  but **not for pca r32** (q = .094 under f_pilot; .025 under f_A_correct; .007 under f_none).
* **OT L12:**
  * full_along_U is effective from r = 1 and an illusion at every rank. The diff runs 0.798 [0.711, 0.878] at pca r1 down to
    0.247 at r64.
  * feat_A_along_U becomes effective at r ≥ 4 (pca) or r ≥ 8 (answer_sep), so OT error-U is "illusion and feature-mediated"
    from those ranks on.
  * The r = 4/8 feature-mediated calls rest on point estimates barely past the bar (pca r4 0.157 vs bar 0.135). Their margin
    tests are not significant (q = 1.0 at r4, 0.52 at r8). They become significant at r ≥ 16 (q = .005).
* **PC L12 and L16:** never. Flip is exactly 0.000 on full and feat at every rank up to 64, for both estimators and all
  filters.
* **repair:**
  * Under f_pilot, repair can never be "effective" for RS or OT. The random-U p95 is 0.92–1.00, so the bar p95 + 0.10 exceeds
    1.0.
  * Under f_A_correct/f_none, RS pca repair is rule-effective at r = 8–32 and answer_sep at r = 16–32 (e.g. 0.975 vs p95
    0.867). None of these survive the margin test (q ≥ 0.97).
  * PC L16 repair is reachable but never effective: max flip − p95 = 0.08.
  * FACT: repair is uninformative under these rules because feat_A_errB already flips 81–100% of pairs.

**DAS rank 1–16 per cell (f_pilot).** Random p95 is 0.000 everywhere except OT (0.023–0.102). q-values are BH within E18.

| cell | k | train acc | valid | full [lo] | feat [lo] | full − feat [CI] | eff / feat-med / illusion | q(ill) |
|---|---|---|---|---|---|---|---|---|
| RS L18 | 1 | 0.250 | F | 0.273 [0.200] | 0.120 | 0.153 [0.093, 0.213] | NA | 0.000 |
| RS L18 | 2 | 0.333 | F | 0.347 [0.273] | 0.200 | 0.147 [0.080, 0.220] | NA | 0.000 |
| RS L18 | 4 | 0.733 | T | 0.760 [0.687] | 0.547 | 0.213 [0.147, 0.287] | T T T | 0.000 |
| RS L18 | 8 | 0.890 | T | 0.867 [0.807] | 0.680 | 0.187 [0.113, 0.260] | T T T | 0.000 |
| RS L18 | 16 | 0.830 | T | 0.780 [0.713] | 0.707 | 0.073 [−0.007, 0.153] | T T **F** | 0.194 |
| OT L12 | 1 | 0.970 | T | 1.000 | 0.989 | 0.011 [0.000, 0.037] | T T F | 1.0 |
| OT L12 | 2 | 1.000 | T | 1.000 | 0.955 | 0.045 [0.010, 0.093] | T T F | 0.222 |
| OT L12 | 4 | 1.000 | T | 1.000 | 0.989 | 0.011 | T T F | 1.0 |
| OT L12 | 8 | 1.000 | T | 0.989 | 1.000 | −0.011 | T T F | 1.0 |
| OT L12 | 16 | 0.997 | T | 1.000 | 0.989 | 0.011 | T T F | 1.0 |
| PC L12 | 1 / 2 / 4 / 8 | 0.007 / 0.017 / 0.143 / 0.437 | F | 0.007 / 0.007 / 0.100 / 0.313 | 0.007 / 0.007 / 0.040 / 0.127 | up to 0.187 | NA | – |
| PC L12 | 16 | 0.810 | T | 0.793 [0.727] | 0.420 [0.340] | 0.373 [0.300, 0.453] | T T T | 0.000 |
| PC L16 | 1 / 2 / 4 / 8 | 0.007 / 0.037 / 0.047 / 0.327 | F | 0.000 / 0.020 / 0.040 / 0.253 | ≤ 0.047 | up to 0.207 | NA | – |
| PC L16 | 16 | 0.780 | T | 0.827 [0.767] | 0.273 [0.207] | 0.553 [0.473, 0.633] | T T T | 0.000 |

* **Does PC DAS fit? Only at rank 16** (acc 0.81 at L12, 0.78 at L16). At rank ≤ 8 it fails (≤ 0.437).
* The valid rank-16 fits are effective, feature-mediated and an illusion. The full − feat gap at PC is the largest of any cell:
  0.553 at L16, with 83 vs 0 discordant pairs.
* **Filter disagreement (the only ones in E18):**
  * RS das16 illusion: F under f_pilot (diff 0.073 [−0.007, 0.153]); T under f_A_correct (0.123 [0.049, 0.198]) and f_none
    (0.115 [0.043, 0.190]).
  * RS pca r16 full_along_U effective and illusion: T only under f_none (0.105 [0.066] vs bar 0.10).
* Error-U vs DAS_k alignment (f_pilot, shipped column): never aligned (max 0.443, OT r16).
  * RS and OT: partial at every rank (0.15–0.44).
  * PC: unrelated at low rank: pca r ≤ 4 at both layers (≤ 0.088), answer_sep r ≤ 2 at L12 and r1 at L16. Partial
    otherwise (≤ 0.252).
* Context (E16, RS layer sweep L12–L20): error-U at r ≤ 4 is never effective at any layer, and das4 is effective at every
  layer. This is consistent with E18's RS rank curve.

## 4. E19 — steering dose s = 0.25…16 (run deb20460858c)

**Question.** Is there any scale at which steering along U meets Q2, or beats matched-norm random directions by ≥ 10 pts?

**FACT: Q2 is met in 0 of 567 rows** (3 filters × 3 cells × 9 estimator/rank settings × 7 scales).
* **RS L18:**
  * The max steer flip is 0.220 [0.153, 0.287] (das8, s = 16, f_pilot), where the random p95 is 0.103, so Q2's random
    condition also fails.
  * Error-U never exceeds 0.133 (answer_sep r1, s = 16, p95 0.074).
  * 54 of 63 f_pilot RS rows have random p95 ≤ 5%. The 9 that fail are exactly the s = 16 rows (p95 0.074–0.103).
* **OT L12:**
  * Flip ≥ 0.5 in 41 of 63 f_pilot rows, but random p95 ≤ 5% in 0 of 63. Random directions themselves flip a lot: p95 by scale
    is 0.112–0.146 at s = .25, 0.26–0.37 at .5, 0.46–0.79 at 1, 0.71–0.96 at 2, 0.73–0.95 at 4.
  * Q2 is unreachable on OT because of the null, not because of U.
* **PC L12:** max flip 0.027 (das8, s ≥ 8). The random p95 is 0 at every scale. The das8_e10 fit is valid here (acc 0.543) and
  still reaches only ≤ 0.020.

**Rows that beat random by ≥ 10 pts** (flip ≥ p95 + .10 and lo > p95):

| cell | estimators | scales | count by filter | margin test after BH |
|---|---|---|---|---|
| OT | every estimator at s ≤ 0.5; all except das8 at s = 1; pca r1 and answer_sep r1 at s = 2 | 28 of 63 (f_pilot) | all 28 significant |
| RS | DAS only | 4–16 | 6 (f_pilot), 5 (f_A_correct), 5 (f_none) | f_pilot: das4 s = 4 (0.180 vs p95 0.007, q = .034) is the only survivor; f_A_correct das4 s = 4: q = .058 |
| PC | none | – | 0 | – |

* Under f_pilot the RS rows are das4 s4; das8 s4/8/16; das8_e10 s4/8.
* The only filter disagreement is RS das8_e10 at s = 8: it beats random by ≥ 10 under f_pilot only.
* INTERPRETATION: on RS, the dose-response for DAS rises then falls (das4: 0.093 at s = 2 → 0.180 at s = 4 → 0.027 at
  s = 16). Steering never reproduces what patching along the same subspace does (full_along_U 0.760 for das4). Error-U steering
  is a null result on RS and PC.
* Replication: E19 matches E10 exactly on flips at the shared settings (135 rows). The OT random p95 differs by up to 0.19
  because the random directions were drawn again.

## 5. E20 — wider SAEs at matched L0 (run ece15bdeaed7)

**Question.** Do E9's rule outcomes change with 65k (RS L18 L0 64; OT L12 L0 72) or 262k (OT L12 L0 67) SAEs vs 16k?

**Rule outcomes vs 16k (E9)**, over 10 estimator/rank rows × 3 filters = 30 per SAE:

| SAE | effective changes | illusion changes | feature-mediated changes |
|---|---|---|---|
| RS 65k | 0 | 0 | 0 |
| OT 262k | 0 | 0 | 2: answer_sep r4 F→T (f_pilot, f_A_correct); feat 0.112→0.146 |
| OT 65k | 0 | 0 | 4: answer_sep r8 T→F (3 filters), pca r4 T→F (f_none) |

All six feature-mediated changes are near the 10-pt bar. The wider-SAE feat flip is within 0.034 of p95 + 0.10 in every case
(16k: within 0.057).

**RS L18 65k (f_pilot):**
* The features-only gap is slightly larger. full_A − feat_A_errB: 0.113 [0.067, 0.167] vs 0.080 [0.040, 0.127] at 16k.
  Under f_A_correct: 0.158 vs 0.138.
* Error-U is still never effective at r ≤ 8.
* DAS full flips are identical to 16k (0.760 / 0.867). The rotation is fit on the residual stream, and the train accuracy is the
  same.
* DAS feat flips rise: das4 0.547 → 0.620, das8 0.680 → 0.727. The illusion diff shrinks (das4 0.213 → 0.140 [0.073, 0.207];
  das8 0.187 → 0.140 [0.080, 0.200]) but still meets the rule. So the illusion flag survives at 65k.
* INTERPRETATION: a wider SAE moves part of the DAS-subspace difference into the reconstruction, but not all of it.

**OT L12 65k and 262k:**
* The features-only gap is 0.000 (f_pilot) at both widths, as at 16k.
* Error-U full_along_U is effective and an illusion at every rank (e.g. pca r1 0.921 at 65k, 0.876 at 262k).
* DAS is effective and feature-mediated, never an illusion (diff ≈ ±0.01).

## 6. E21 — multilingual fidelity with a 4× wider SAE (rows in `r5/tables/E12_fidelity_lang.csv`)

**Question.** Does M1 (CE-recovered clearly below English, same layer, same SAE) still hold at 65k? How do recovered and the
splice cost change vs 16k?

**FACT. M1 (mean ablation, primary) is supported for all 11 non-English language sets** (bn ×3, fr, hi ×3, hinglish, ta ×3) at
L12 and L18, at both 16k and 65k: 44 of 44. No set is mixed across layers.
* Zero ablation: 16k 9/11 (L12) and 10/11 (L18); 65k 10/11 and 9/11. The sets that are not supported under zero ablation
  are:
  * fr_web: 16k at L12 and L18, and 65k at L18 (65k supports it at L12);
  * ta_latn_dakshina: L12 at both widths, and L18 at 65k.
* The en_pilot100 parity row is not below English at either width: 16k 0.9855 / 0.9792; 65k 0.9897 / 0.9842.

**16k → 65k, the 11 language sets** (`E21_16k_vs_65k.csv`):

| quantity | L12 (min / median / max) | L18 (min / median / max) |
|---|---|---|
| recovered_mean change | +0.0003 / +0.0137 / +0.0258 (11 of 11 higher) | +0.0063 / +0.0193 / +0.0743 (11 of 11 higher) |
| ce_delta change (nats/token) | −0.0743 / −0.0529 / −0.0011 (11 of 11 lower) | −0.1765 / −0.0679 / −0.0434 (11 of 11 lower) |
| gap to English (lang − en) | narrows for 9 of 11; widens slightly for hi_latn_dakshina (−0.0201 → −0.0237) and fr_web (−0.0047 → −0.0048) | narrows for 11 of 11 |

English itself also improves: en_pile recovered goes 0.9857 → 0.9896 (L12) and 0.9789 → 0.9835 (L18), and ce_delta goes
0.114 → 0.083 and 0.170 → 0.133.

**Largest single change:** ta_latn_dakshina at L18, recovered 0.8857 → 0.9600. Its measured L0 is 43.4 → 38.7, the lowest of all
sets.

**Caveats:**
* The 16k and 65k SAEs differ in L0. Measured on en_pile: L12 84.8 vs 75.7, L18 75.3 vs 65.6. "Matched" means matched only
  approximately.
* Genre differs across sets (stated in the addendum).
* INTERPRETATION: the wider SAE reduces the multilingual fidelity deficit somewhat but does not remove it. Every language stays
  clearly below English by the frozen CI rule.

## 7. E22 — multilingual DAS/PCA, native vs English fit (run 1c85291e335f)

**Question.** In each language, does a U fitted on that language flip more or fewer pairs than the English-fitted U? What do
the E9 rules and the E13 gates say?

**E13 gates (test split; `tables/final/E13_behaviour_lang.csv`):**

| family | clean acc | Gate A | Gate B |
|---|---|---|---|
| rs_fr | 0.802 | ✓ | ✗ (−11.3) |
| rs_hinglish | 0.678 | ✓ | ✗ |
| rs_hi | 0.536 | ✗ | ✗ |
| pc_fr | 0.996 | ✓ | ✓ |
| pc_hi | 0.780 | ✓ | ✗ |
| pc_hinglish | 0.713 | ✓ | ✗ |

**Only pc_fr is readable under the addendum's gating.** By the frozen logic, no E22 comparison is both readable and evidential:
in pc_fr every DAS fit is invalid and PCA flips are 0.000.

**f_pilot (n = 150 eligible per row)**:
* "rules" = eff_full / feat-med / illusion; NA = invalid fit.
* Train acc for the English fit is measured on the English pack.
* Random p95 ≤ 0.007 everywhere.
* q is the BH-adjusted two-sided McNemar for English vs native on the same pairs.

| lang | U | train acc native / English | full flip native / English (Eng − nat, q) | feat flip native / English (Eng − nat, q) | rules native | rules English |
|---|---|---|---|---|---|---|
| rs_fr | das4 | 0.247 / 0.633 | 0.213 / 0.607 (+0.393, q < .001) | 0.147 / 0.387 (+0.240, q < .001) | NA (failed fit) | T T T |
| rs_fr | das8 | 0.593 / 0.687 | 0.500 / 0.580 (+0.080, q = .33) | 0.313 / 0.387 (+0.073, q = .22) | T T T | T T T |
| rs_hi | das4 | 0.160 / 0.633 | 0.260 / 0.520 (+0.260, q < .001) | 0.073 / 0.307 (+0.233, q < .001) | NA | T T T |
| rs_hi | das8 | 0.880 / 0.687 | **0.840 / 0.620 (−0.220, q < .001)** | 0.687 / 0.347 (−0.340, q < .001) | T T T | T T T |
| rs_hinglish | das4 | 0.453 / 0.633 | 0.387 / 0.527 (+0.140, q = .034) | 0.200 / 0.253 (+0.053, q = .61) | NA | T T T |
| rs_hinglish | das8 | 0.580 / 0.687 | 0.587 / 0.613 (+0.027, q = 1) | 0.327 / 0.373 (+0.047, q = .77) | T T T | T T T |
| rs_* | pca r4/r8 | – | 0.020–0.060 / 0.013–0.053 (≤ ±0.007) | ≤ 0.007 | F F F | F F F |
| pc_fr | das8 | 0.453 / 0.373 | 0.320 / 0.047 (−0.273, q < .001) | 0.180 / 0.007 | NA | NA |
| pc_hi | das8 | 0.187 / 0.373 | 0.107 / 0.007 (−0.100, q = .001) | 0.027 / 0.000 | NA | NA |
| pc_hinglish | das8 | 0.293 / 0.373 | 0.160 / 0.020 (−0.140, q < .001) | 0.040 / 0.000 | NA | NA |
| pc_* | das4 | ≤ 0.107 / 0.093 | ≤ 0.040 / ≤ 0.007 | ≤ 0.007 | NA | NA |
| pc_* | pca r4/r8 | – | 0.000 / 0.000 | 0.000 / 0.000 | F F F | F F F |

**English-pack references (f_pilot):**
* rs_en: das4 0.573 (acc 0.633), das8 0.633 (0.687), both T T T; pca r4/r8 0.047 / 0.067, F F F.
* pc_en: das4 0.053 (acc 0.093), das8 0.280 (0.373), both NA; pca 0.000.

**FACT:**
* On RS, the English-fit DAS rotations meet the effective, feature-mediated and illusion rules in all three languages. Full
  flips are 0.52–0.61, close to the English reference.
* Where the native fit is valid (das8 in fr, hi and hinglish), it is also T T T. Native das8 is no worse than English in fr and
  hinglish, and better in hi.
* Native das4 fails to fit in every language.
* PCA error-U flips ≤ 0.07 in every language and fit, and is never effective. Native and English PCA are indistinguishable
  (|Δ| ≤ 0.007).
* On PC, every fit fails. Native das8 still flips more than English das8 in every language (q ≤ .001).
* No filter disagreements in E22. f_A_correct and f_none give the same rule strings and the same signs; for example, rs_hi das8
  Eng − nat is −0.211 under f_A_correct and −0.190 under f_none.

**Caveats:**
* E22's DAS fits used **n_train = 150**, half of E18's 300. The pc_en das8 train accuracy is 0.373 here vs 0.437 in E18, and the
  rotations are not equal to E18's. RS English das4 is 0.633 here vs 0.733 in E9/E17, where it was fit on the pilot families
  instead of the rs_en pack.
* rs_hi, whose native das8 beats English, fails Gate A (test acc 0.536), and the frozen gates would not let it be read.
* The f_pilot subset is 150 pairs in every family. In rs_hi, f_none has 279 pairs (184 eligible).
* The E14 M3 rule (share closed by repair) was not computed for E22. This readout reports flip rates only, as the brief asks.

## 8. Things a methods section must say

1. **All of E17–E22 is exploratory.** The frozen rules are applied mechanically. No result here changes a preregistered
   answer.
2. **The DAS validity gate decides most DAS outcomes.** Within E17, test flips track train accuracy almost perfectly (ρ = 0.96).
   Rank 1/2 RS fits sit at 0.08–0.47 accuracy while still flipping 9–50% of pairs against p95 = 0. This excludes the
   diverged lr 3e-2 rank-1 fit (acc 0, flips 0).
3. **Repair cannot pass the effective rule under f_pilot on RS or OT.** The random p95 is ≥ 0.92, which is a ceiling. "Not
   effective" for repair there is a property of the rule, not of the data.
4. **The OT random-U and random-steer nulls are high and vary between runs** (p95 differs by up to 0.19 between E10 and E19).
   OT Q2 is unreachable by construction of the null.
5. **E23 (norm-matched noise) was not run.** The "illusion" calls for error-U at RS r ≥ 32 compare against Haar-random subspaces
   of the same rank only. They are not compared against a norm-matched perturbation of feat_A_along_U.
6. **DAS seed variance is not in any CI.** All E9/E18/E20/E22 DAS rows are single-seed (seed 0). E17 shows that seed changes full
   flips by up to 0.17 at rank 4 (0.760 vs 0.593), which is larger than the bootstrap CI half-width (≈ 0.07).
7. **The `cos_top_das` alignment column holds one DAS variant per rank.** I recomputed it for all E17 variants in
   `E17_alignment.csv`.

## Files

* Scripts: `$SP/team/scripts/standard_common.py`, `standard_e17.py` … `standard_e22.py`. Run them from `$SP/team/scripts` with
  `OMP_NUM_THREADS=2`.
* Outputs: `$SP/team/standard_out/`:
  * E17: `E17_readout.csv`, `E17_tests.csv`, `E17_mcnemar.csv`, `E17_vs_E9.csv`, `E17_stability_annot.csv`,
    `E17_alignment.csv`
  * E18: `E18_rows.csv`, `E18_das.csv`, `E18_error_U_summary.csv`, `E18_tests.csv`
  * E19: `E19_rows.csv`
  * E20: `E20_vs_E9.csv`, `E20_tests.csv`
  * E21: `E21_m1.csv`, `E21_16k_vs_65k.csv`
  * E22: `E22_readout.csv`, `E22_tests.csv`
* Logs: `$SP/team/standard_out_e17.log` … `standard_out_e22.log`.

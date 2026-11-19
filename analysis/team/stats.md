# Statistician report (role `stats`), 2026-11-19

Everything here is **exploratory and post hoc**: it was designed after all results were seen. Nothing revisits a preregistered answer.
Data: E12 (16k, gnode084) and E21 (65k, gnode058) per-window npz (`$SP/ana_runs/E12/*/per_window_*.npz`; width and L0 from each
manifest), the r5 tables, E17–E22 per_pair.parquet. E23 (norm-matched noise) was **not run**, so "fragility to any perturbation of
size |ε|" cannot be separated from "fragility to SAE error" anywhere below.
Scripts are in `$SP/team/scripts/stats_*.py` (run from `$SP/hw`, `OMP_NUM_THREADS=2`) and outputs in `$SP/team/stats_out/`.

| script | what | output |
|---|---|---|
| `stats_ml_matched.py` | P1a: CEM + nearest-neighbour matching, all 12 sets as targets (Latin targets are placebos) | `mm_matched.csv`, `mm_matched.log` |
| `stats_ml_overlap.py` | P1a: within-language comparison on the only overlapping predictability range | `mo_overlap.csv` |
| `stats_ml_script_mix.py` | P1 diagnostic: script composition of windows (Gemma tokenizer, offline) | `ms_script_mix.csv/.log` |
| `stats_ml_regress.py` | P1b: window regressions, 100-spec grid, per-cell fits, PPML, wild/cluster/window bootstraps, LOSO | `mr_specs.csv`, `mr_cells.csv`, `mr_ppml.csv`, `mr.log`, `mr2.log` |
| `stats_power.py` (earlier attempt, reused after checking) + `stats_power2.py` | P2: MDEs, null classification | `power_*.csv`, `power2_classified.csv`, `power2_e22_vs_english.csv`, `power2.log` |
| `stats_ci_audit2.py` (+ earlier `stats_ci_audit.py`) | P3: CI audit of E18_illusion / E19_steering | `ci_audit2.csv/.log` (`ci_audit.csv`) |

Definitions. Splice cost per window = (NLL_splice − NLL_clean)/pred, in nats/token (123 scored predictions per window).
"Predictability" (nllc) = clean NLL per token. Three error-size measures: **FVU** = |ε|²/|x − set mean|² (the denominator uses each
set's own mean); **|ε|/|x|** = mean relative error per position; **raw |ε|²** per position (mean-free). Native = hi_deva_dakshina,
hi_deva_web, ta_taml_dakshina, ta_web, bn_beng_dakshina, bn_web. Latin = en_pile, fr_web, hinglish_phinc, and the {hi,ta,bn}_latn_dakshina sets.
Main cells: layers 12 and 18 × widths 16k and 65k (these 4 cells have all 12 sets). Layers 6 and 24 exist only at 16k and only for the Hindi sets.

Multiplicity. 1365 matched ratios, 54 overlap ratios, 203 regression fits and 12 script-mix slopes were estimated, all
**uncorrected**. They are read as patterns across many cells and specifications, never as single p-values. P2 applies BH within each
experiment (stated where used). P3 is an exactness check, not a test.

---------------------------------------------------------------------------------------------------------------------------------

## Findings, ranked by strength

### F1 (strong, FACT). The bootstrap CIs in E18_illusion and E19_steering are reproduced exactly

- **What I checked.** I sampled 40 random rows (20 from each table) and recomputed them from per_pair.parquet in fresh code
  (`stats_ci_audit2.py`, no harness imports).
- **Counts.** Neither n_pairs nor flip_n_eligible differs on any row (0 / 40 mismatches).
- **Same generator and seed (B = 10 000, seed 0).** Every point estimate and every 95 % CI bound of flip rate, NLDR and
  prob_recovery, and of their gaps, agrees to ≤ 4.4e-16. This also confirms that make_final used B = 10 000, its default.
- **Independent seed (B = 2000).** The largest difference in any CI bound is 0.0065 for flip rate, 0.0112 for NLDR and
  0.0108 for prob_recovery. That is Monte Carlo noise only.
- **Other columns.** random_mean, random_p95, random_max and random_n match exactly on all 40 rows. illusion_diff and its
  paired CI also match exactly on the 13 E18 rows where it is defined.
- **Earlier audit.** The previous attempt audited 56 rows of the `*_conditions` tables (E9, E2r, E17–E20, E22;
  `ci_audit.csv`), and those were exact as well.
- **Limits.** This verifies the arithmetic and resampling only. Pair-level bootstraps ignore template/family clustering, and
  they do not include fit variability: the U matrices are held fixed.

### F2 (strong, FACT). Predictability alone does not explain the native-script excess over English or French. Predictability plus relative error size explains roughly half of it, in log terms

Matched comparisons (`mm_matched.csv`) at L12/L18, 16k/65k, 6 native sets: 24 cells. "Supported" means ≥ 50 % of target
windows matched and ≥ 100 windows. CIs come from 1000 (CEM) or 300 (NN) bootstraps of windows within each set, with the matching redone in every draw.

| matched on | native / English, median [range] | CI > 1 | native / pooled other Latin | Latin placebo / pooled other Latin |
|---|---|---|---|---|
| nothing (raw, regression geometric mean) | 2.74 overall | — | — | — |
| clean NLL (CEM, 0.25-nat bins) | **2.95 [2.32, 4.08]** | 24/24 | 2.64 [2.05, 3.39], 24/24 > 1 | 1.00 [0.65, 2.25]: 12 above 1, 12 below |
| clean NLL (NN, caliper 0.125 nat) | 2.98 [2.39, 4.09] | 24/24 | 2.63 [2.05, 3.32] | 1.01 [0.65, 2.54] |
| clean NLL + log \|ε\|/\|x\| (CEM) | **1.47 [1.23, 1.75]** (21 supported) | 19/21 | 1.56 [1.31, 2.39], 23/23 > 1 | 0.97 [0.64, 2.28]: 9 above, 8 below |
| clean NLL + log raw \|ε\|² (CEM) | 3.97 [2.77, 4.83] (18 supported) | 18/18 | 2.99 [1.89, 3.48] | 0.99 [0.58, 2.45] |
| clean NLL + log FVU (CEM / NN) | **not identified**: only 1–6 of 24 native cells reach 50 % support | — | — | — |

- **Balance is good.** After nllc + |ε|/|x| matching, the mean nllc and |ε|/|x| of matched English windows are within
  0.04 nat and 0.007 of the native windows. FVU still differs, though (native 0.25–0.36 vs English 0.19–0.27): the two
  error-size measures disagree.
- **Per-set ratios vs English** at nllc + |ε|/|x|:
  - L12: 1.23–1.86
  - L18: 1.35–1.75
  - The ratio rises with depth in the Hindi sets: L6 0.70–0.74 (native *cheaper*), L24 1.94–1.95.
  - Without the error control (nllc only), the same cells give 2.3–4.1.
- **French vs English** (a high-resource Latin placebo): 1.31–1.47 matched on nllc, 1.00–1.11 on nllc + |ε|/|x|.
- **Interpretation.** Predictability does not explain the gap to English or French. Relative error size explains a good part of
  what remains: |ε|/|x| is about 1.2x larger in native windows (L12 16k medians 0.46–0.50 vs English 0.39). Which "error
  size" you condition on changes the answer by a factor of 2–3. Raw |ε|² is *smaller* in native windows (|x| is smaller), so conditioning on it raises the ratio.
- **Alternative explanations.** The native–English contrast confounds script with language, corpus, tokenizer fertility and
  training-data share. Between Latin sets, ratios at matched predictability already range from 0.64 to 2.28, mostly in the
  romanized sets. So a 1.5x native/Latin difference is within ordinary between-corpus variation.

### F3 (moderate-strong, FACT). Within a language, at genuinely matched predictability, the script effect is small and mostly disappears; it exists only at L12 16k

This is the only within-language test that does not extrapolate (`mo_overlap.csv`, `stats_ml_overlap.py`). Native and
romanized Dakshina windows overlap in clean NLL only for Hindi: [3.58, 4.45] nats/token, which holds 79 native windows (4 %)
and 188 romanized windows (9.4 %). Tamil and Bengali have zero overlapping windows, so their within-language contrast is not
identified without extrapolation. The overlapping native windows are genuine native text: 99.2 % of their letters are
Devanagari, against 99.7 % in the rest of the set (`ms_script_mix.log`). They are the least predictable native windows,
though, so they are a selected tail. CIs: 2000 window bootstraps.

| hi_deva_dakshina / hi_latn_dakshina, overlap windows | raw | adjusted for nllc | adjusted for nllc + log \|ε\|/\|x\| |
|---|---|---|---|
| L6 16k | 0.92 [0.80, 1.05] | 0.59 [0.38, 0.89] | 0.53 [0.32, 0.81] |
| L12 16k | 1.67 [1.52, 1.84] | **1.34 [1.09, 1.71]** | 1.21 [0.95, 1.60] |
| L12 65k | 1.39 [1.25, 1.54] | 1.00 [0.75, 1.31] | 0.90 [0.64, 1.29] |
| L18 16k | 0.94 [0.86, 1.03] | 0.71 [0.57, 0.89] | 0.73 [0.59, 0.90] |
| L18 65k | 0.97 [0.88, 1.07] | 0.64 [0.54, 0.78] | 0.63 [0.52, 0.77] |
| L24 16k | 0.99 [0.95, 1.04] | 0.81 [0.73, 0.91] | 0.96 [0.74, 1.16] |

- **Against Hinglish** (romanized Hindi–English), hi_deva_dakshina, overlap [2.46, 4.45], 1897 vs 494 windows:
  - adjusted for nllc: L12 1.27 [1.09, 1.56] (16k) and 1.27 [1.04, 1.67] (65k); L18 1.08 [0.93, 1.37] and 0.98 [0.77, 1.39]
  - with ε added: 1.11, 1.11, 1.05 and 0.89, all CIs including 1
  - hi_deva_web vs Hinglish: 0.68–1.01
- **The CEM version** of the same contrast (bins, 114 vs 233 windows) agrees: L12 1.95 (16k) and 1.54 (65k); L18 1.00 and 1.10.
  Its bootstrap distributions are strongly skewed (the point estimate sits near a CI edge), so I give it less weight.
- **Script composition within native web sets** (`ms_script_mix.log`). At fixed nllc and |ε|/|x|, windows with more Latin
  letters cost less:
  - hi_deva_web: −0.44 [−0.59, −0.31] log-cost per unit Latin share at L12, −0.30 at L18
  - bn_web: −0.46 and −0.37
  - ta_web: −0.12 (n.s.) and −0.21
  - Taken literally, a fully native vs fully Latin window would differ by a factor of 1.13–1.58. This is an extrapolation,
    and the Latin stretches in Indic web text are often English, so language is again confounded.
- **Interpretation.** At matched predictability, inside one language, native script costs at most about 1.3x, and only at
  L12 with the 16k SAE. At L6/L18/L24, and at L12 with the 65k SAE, it costs the same or less. The deep-analysis claim that
  "script, not error size" sets the cost (1.5–4.1x) does not survive here.
- **Caveat.** One language, a tail of 4 % of windows, and a linear adjustment within a 0.9-nat interval. This is the best
  identified contrast available, but it is narrow.

### F4 (moderate, FACT). Regression with set clusters: the script coefficient survives only without language fixed effects, and it is sensitive to the error measure and the predictability specification

Model and inference (`mr_specs.csv`, pooled over the 4 cells, 94 652 windows with cost > 0; the log model drops 0.3–2.4 % of
windows, almost all Latin, which biases *against* a script effect):
- **Model.** log cost = cell FE + native + cell × [error terms] + cell × f(nllc) [+ language × cell FE].
- **CR1 CIs** use t with 11 degrees of freedom.
- **Wild-cluster p.** Restricted Rademacher bootstrap with all 2¹² = 4096 sign vectors enumerated, so the smallest attainable
  p is 1/4096.
- **Cluster bootstrap.** 6 native and 6 Latin sets are drawn with replacement, B = 999.
- **Window bootstrap.** B = 200, main specs only.

| spec (f(nllc), error) | no language FE: ratio [CR1 CI], wild p, cluster-boot CI | with language FE (hi/ta/bn/en/fr; Hinglish = hi) |
|---|---|---|
| none, none | 2.74 [2.27, 3.31], p 0.0002, [2.32, 3.21] | 2.59 [2.08, 3.23], p 0.001 |
| rcs4, log FVU | **1.99 [1.36, 2.91], p 0.008**, [1.17, 2.62] (window-boot [1.95, 2.02]) | **1.60 [1.15, 2.25], p 0.061**, [1.11, 2.96] |
| rcs4, log \|ε\|/\|x\| | **1.53 [1.36, 1.71], p 0.001**, [1.03, 1.76] | **1.41 [1.10, 1.82], p 0.116**, [0.97, 2.09] |
| rcs4, FVU + \|ε\|/\|x\| | 1.56 [1.33, 1.82], p 0.014 | 1.40 [1.14, 1.71], p 0.113 |
| rcs6, \|ε\|/\|x\| | 1.46 [1.32, 1.62], p 0.0002 | 1.14 [0.83, 1.57], p 0.636 |
| bins (0.25 nat), log FVU | 2.03 [1.40, 2.95], p 0.002 | 1.81 [1.28, 2.57], p 0.135 |

- **Spread over the 16 specs that control both predictability (lin, rcs4, rcs6, bins) and error (FVU, |ε|/|x|, raw |ε|²,
  FVU + |ε|/|x|):**
  - without language FE: ratio 1.46–3.03, wild p < 0.05 in 16/16 specs
  - with language FE: ratio 1.14–1.92, wild p < 0.05 in 5/16
  - common-support windows only (a cell × nllc-bin × FVU-bin stratum must contain both scripts): 1.29–2.46 without
    language FE (16/16 p < 0.05; 1.29–1.44 when the error term is FVU and/or |ε|/|x|), and 1.08–1.67 with language FE (4/16)
- **Share of the raw log-gap ln 2.74 explained.** 32 % (rcs4 + FVU), 58 % (rcs4 + |ε|/|x|), 66 % (that spec + language FE),
  87 % (rcs6 + |ε|/|x| + language FE).
- **Leave-one-set-out.** In the rcs4 + FVU spec the ratio stays between 1.53 and 2.22. Dropping English and French together
  gives 1.25 on all windows and **1.05** on common support. The script coefficient at matched predictability therefore rests
  on two Latin clusters, en and fr, the only Latin sets that span the native nllc range.
- **PPML robustness** (arithmetic-mean ratio, keeps cost ≤ 0; `mr_ppml.csv`): rcs4 + FVU 1.61 [1.27, 2.05]; rcs4 + |ε|/|x|
  1.41 [1.28, 1.56]; language FE + rcs4 + FVU 1.42 [1.01, 1.99].
- **Per cell** (`mr_cells.csv`; separate fits, 12 clusters each), rcs4:
  - FVU control: L12 2.59 (16k, p 0.023) and 2.29 (65k, p 0.008); L18 1.54 (p 0.002, but the cluster bootstrap CI is
    [0.88, 1.71]) and 1.78 (p 0.049)
  - |ε|/|x| control: 1.77, 1.51, 1.46, 1.43
  - with language FE + |ε|/|x|: 1.84, 1.48, 1.20, 1.23, all p ≥ 0.094
- **Limits of the inference.** There are 12 clusters, 6 per arm, and only 2–3 Latin sets have support where native windows
  lie. Wild p-values are discrete. The cluster bootstrap with 6 draws per arm is coarse. Set confounds language, corpus,
  domain and tokenizer. Window-level CIs (e.g. [1.95, 2.02]) are about 20x too narrow for a claim about scripts or languages:
  only between-set inference is relevant.

### F5 (verdict on the multilingual crux; INTERPRETATION, quantified)

- **Survives.** Native Indic text loses more next-token loss than English or French text at matched clean loss and matched
  relative error |ε|/|x|.
  - Matched ratio about 1.5x (CEM median 1.47 vs English, 1.56 vs pooled Latin; regression 1.41–1.53).
  - The raw ratio is about 2.7–3.0x, so predictability plus relative error explains roughly half the log-gap.
  - It holds at 16k and 65k. It is larger at L12 than at L18, and it reverses at L6.
- **Not identified.** With FVU as the error measure, matching has no common support. Regression puts it at about 2.0x
  without language FE and 1.6x with them.
- **Does not survive.** "Script, not error size." The only contrast that holds language fixed and matches predictability
  without extrapolation (Hindi Dakshina, 4 % tail) gives 1.34 [1.09, 1.71] at L12 16k and 0.6–1.0 elsewhere. Native vs
  Hinglish gives 0.9–1.3, with CIs that include 1 once ε is controlled. With language FE the regression's script ratio falls
  to 1.1–1.9 and is mostly not significant (wild p 0.06–0.64 in the main specs). Dropping English and French removes it
  (1.05 on common support).
- **Wording I would defend.** "At matched predictability and relative reconstruction error, Gemma Scope splicing costs about
  1.5x more next-token loss on Hindi, Tamil and Bengali text than on English or French. Whether this is due to script,
  language or corpus is not identified. Within Hindi, at matched predictability, romanized and native script cost about the
  same except at L12 (16k), where native costs about 1.3x more."
- **What would settle it.** Token-level NLLs (to match per token, not per window), E23 norm-matched noise per set, and a
  predictable Latin-script low-resource language. The red team proposed the same.

### F6 (moderate, FACT). Power: most E17–E22 nulls are informative at the 10-point bar. The E22 claim that transfer is "as strong as in English" is underpowered

MDE80 = 2.8 × the bootstrap SE (α 0.05 two-sided, power 0.8). A null is "informative" when its 90 % CI lies inside ±10 points
(TOST at the project's 10-point bar). It is "degenerate" when both arms sit at floor or ceiling. Inputs: `power_contrasts.csv`
(per-pair, B = 2000, same pairs), `power2_classified.csv`, `power2_e22_vs_english.csv`.

| family | n eligible | MDE80 (pts), median [range] | significant | informative null | underpowered null | degenerate |
|---|---|---|---|---|---|---|
| E17 seed 1 or 2 vs seed 0 (RS L18 DAS) | 203 | 9.6–9.8 [5.8, 12.4] | 15 / 30 | 8 | 7 | 0 |
| E17 epochs 10/30 vs 3 | 203 | 8.6–9.6 [5.1, 11.3] | 3 / 18 | 14 | 1 | 0 |
| E17 lr 3e-2 vs 1e-2 | 203 | 7.9 [5.8, 9.5] | 5 / 6 | 1 | 0 | 0 |
| E20 RS L18 65k vs 16k (same pairs) | 203 | 3.6 [1.9, 8.2] | 5 / 14 | 9 | 0 | 0 |
| E20 OT L12 65k/262k vs 16k | 98 | 8.3 [1.7, 14.8] | 10 / 28 | 11 | 2 | 5 |
| E22 English-fit − native-fit (paired) | 150–205 | 8.2 [1.2, 13.2] | 16 / 48 | 12 | 2 | 18 |
| E22 language − English home (unpaired) | 150–180 per arm | RS DAS rows 11.4–15.0 | 22 / 96 (95 % CI; BH q in csv) | 36 | 14 | 24 |

- **E17 seeds.** Seed instability is real, not a power artefact. Seed 1 vs 0 lowers full_along_U by 16–19 points for das1,
  das4 and das8 (BH q < 0.001) and lowers feat_A_along_U by 21–23 points for das4 and das8. The illusion contrast
  (full − feat) across seeds is the weakest part: 4 of its seed nulls are underpowered (90 % CI up to +0.135, MDE 10–12 pts).
  So "the illusion flag is seed-stable" cannot be claimed at the 10-point bar.
- **E20 width, RS.** The width null for the main RQ1 quantity is informative: gap(feat_A_errB) flip 65k − 16k = +0.020
  [−0.035, 0.074], 90 % CI inside ±10, n = 203. Width does change the DAS illusion contrast by −6 to −8 points (das4
  −0.079 [−0.138, −0.025], BH q 0.04). The DAS full_along_U arm does not depend on the SAE, so this is a feature-arm change.
- **E20 width, OT.** Five contrasts are degenerate: the flip gap is 0, and the DAS illusion_diff is 0 [0, 0] at 65k and 262k
  because the two arms agree on every pair (no discordant pairs). They carry no information about width. The answer_sep r1/r2 illusion
  contrasts change by −10 to −16 points (262k r1 −0.163 [−0.262, −0.068], q 0.012). At n = 98, effects below ~14 points are
  not detectable.
- **E22, English home vs language.**
  - RS: none of the 12 English-fit DAS contrasts against the English home (das4/das8 × full/feat × fr/hi/hinglish) is
    significant. Ten are *underpowered* and two are informative: MDE80 13–15 points, with CIs such as fr das4 full +0.054 [−0.051, 0.159].
    "Transfers fully, as strong as in English" (multilingual.md F4) is therefore supported only to within about ±15 points.
    The data exclude a large drop, not a 10-point one.
  - PC: the English fit is significantly *below* the English home for das8 full (−0.23 to −0.27; for example hi −0.271
    [−0.344, −0.199]). PC transfer fails.
  - The 18 PC paired contrasts and 24 unpaired ones with flips at 0 vs 0 are degenerate. They show only that both fits fail
    in PC.

---------------------------------------------------------------------------------------------------------------------------------

## Caveats that apply throughout
- All window-level CIs treat windows as independent. Consecutive packed windows share sentences, and an earlier analysis
  (`stats_ml_width.py`, block bootstrap) found little dependence inflation. The relevant uncertainty for claims about scripts
  or languages is between sets (F4), not between windows.
- Only one model and one SAE family (Gemma Scope) per width, and no SAE-training replicate, so the 16k → 65k differences
  carry no training-variance term.
- Splice cost is per token and window-aggregated, and KL was not saved. Token-level matching was not possible: per-token NLLs
  are not stored in these npz files.
- Each script or language is represented by a single corpus. A "language" or "script" effect is a corpus-level effect with
  n = 1 per cell.

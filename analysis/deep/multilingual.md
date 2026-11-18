# Multilingual deep-dive: SAE fidelity and task-state loss across languages and scripts

EXPLORATORY, post hoc (2026-11-18). Gemma-2-2b, Gemma Scope 16k residual SAEs (pilot L0 variants at L12/L18, canonical at
L6/L24). No new compute; no multiplicity correction; nothing here revisits a preregistered answer (M1 / M2 / M3 stay as recorded
in analysis/final_pass_results.md). Every number is reproducible with deep/scripts/ml_*.py (run ml_load.py first).

Data: E12 per-window arrays (13 text sets x up to 4 layers, 2000 windows of 124 scored tokens; hinglish 1663; en_pilot100
excluded), E14 per-pair logits (test split, native fit), E13 gates. CIs are 95% bootstrap over windows (E12) or pairs (E14).
Terms: splice cost `ce_delta` = CE(x -> x_hat) - CE(clean), nats/token; FVU = |eps|^2/|x - mean|^2; `sens` = ce_delta / FVU;
"native" = Devanagari / Tamil / Bengali script; "romanized" = Dakshina Latin transliterations of the same sentence pool.

## F1 (strongest): script, not reconstruction size, sets the cost of SAE error

Natural experiment: Dakshina gives the SAME sentence pool in native script and romanized (same language, genre, source).
Native / romanized ratios (ml_script_pairs.py; 2000 bootstraps):

| lang | layer | ce_delta ratio [CI] | FVU ratio | eps/x ratio | sens ratio [CI] |
|---|---|---|---|---|---|
| hi | 6 | 1.22 [1.18, 1.26] | 0.98 | 1.01 | 1.25 [1.21, 1.29] |
| hi | 12 | 2.44 [2.37, 2.51] | 0.97 | 1.02 | 2.52 [2.46, 2.60] |
| hi | 18 | 1.51 [1.48, 1.55] | 0.99 | 1.01 | 1.53 [1.50, 1.57] |
| hi | 24 | 1.42 [1.40, 1.44] | 0.91 | 0.97 | 1.56 [1.54, 1.58] |
| ta | 12 | 4.13 [3.99, 4.28] | 0.85 | 1.06 | 4.84 [4.67, 5.01] |
| ta | 18 | 1.73 [1.70, 1.76] | 0.91 | 1.02 | 1.89 [1.86, 1.93] |
| bn | 12 | 1.93 [1.88, 1.99] | 0.94 | 1.07 | 2.07 [2.01, 2.12] |
| bn | 18 | 2.29 [2.24, 2.34] | 1.07 | 1.10 | 2.15 [2.11, 2.20] |

Same-size reconstruction error (FVU and |eps|/|x| within ~0.85-1.10x) costs the model 1.2-4.1x more next-token loss when the
text is in native script; 8/8 (language, layer) cells, every CI excludes 1. Against English (ml_ratios.py): every non-English set
has FVU 1.1-1.8x English's (romanized included: ta_latn 1.75x at L12, the largest), but sensitivity is 1.4-2.3x English for
native scripts (e.g. hi_deva_web L12 1.89 [1.81, 1.96]) and <= 1.0x for romanized (ta_latn L12 0.45, hi_latn 0.72) and Hinglish
(1.00). Pooled window-level OLS, ce_delta ~ |eps|/|x| + ce_clean + group (English baseline; ml_script_pairs.py, 200 bootstraps
resampling windows within sets): native-script excess +0.066 nats [0.054, 0.075] at L12 (English mean 0.114), +0.115 [0.099,
0.129] at L18, +0.455 [0.445, 0.467] at L24, and none at L6 (-0.017 [-0.030, -0.003]); romanized -0.076 [-0.086, -0.065] at
L12, +0.029 [0.011, 0.048] at L18. Same pattern with FVU as the covariate (ml_matched.py).

Predictability is not the explanation vs English: native-script windows have the same clean CE as English (median 2.18 vs 2.33
nats at L12) yet ~2x sensitivity. Caveat: romanized text is far less predictable (median ce_clean 5.6 nats; no overlap with
native 1.45-3.19 at the 5-95%), so the native-vs-romanized contrast cannot be separated from predictability by matching; the
linear control says ce_clean's within-set slope is small (-0.005 to -0.05 nats per nat), far too small to explain 0.17 nats.

Paper claim: "Holding the text fixed and changing only the script, Gemma Scope SAEs reconstruct equally well (FVU ratio
0.85-1.07) but the model pays 1.2-4.1x more loss for the same error in native Indic scripts: SAE quality metrics based on
reconstruction size miss where the error matters."

## F2: task-state loss is language-invariant (a constant fraction), despite 2-3x text-level fidelity deficits

E14, native fit, f_A_correct (ml_e14.py, 4000 bootstraps). Gap = 1 - NLDR(features of A + B's error):

| task (layer) | en | fr | hi (Deva) | hi_latn | hinglish |
|---|---|---|---|---|---|
| people-cities (L12) | 0.237 [0.226, 0.249] | 0.241 (+0.004 [-0.011, 0.019]) | 0.244 (+0.007 [-0.007, 0.021]) | 0.244 (+0.007) | 0.221 (-0.016 [-0.032, -0.001]) |
| running sum (L18) | 0.148 [0.125, 0.174] | 0.125 (-0.023 [-0.056, 0.009]) | 0.123 (-0.025 [-0.056, 0.006]) | skipped | 0.116 (-0.032 [-0.067, 0.001]) |
| object tracking (L12) | 0.221 [0.204, 0.238] | skipped | skipped | 0.238 (+0.017 [-0.026, 0.061]) | skipped |

Across 8 non-English comparisons the largest |difference| is 3.2 points; 1/8 CIs excludes 0 (hinglish PC, -1.6, i.e. smaller).
The same holds with f_none (differences -4.1 to +2.3). Absolute logit loss scales with each language's clean margin (PC: hi 4.23
logits on a 17.3 margin vs en 2.83 on 11.9), i.e. the SAE removes a fixed share of the binding signal. Meanwhile, at the same
layer and SAE (L12), ordinary Devanagari Hindi text has 2.5x English's splice cost (F1). The all-candidate version (A vs the
best other answer candidate; ml_candidate_margin.py) agrees for people-cities: 0.118 (en) vs 0.125-0.136 (others).

Caveat: every non-English E14 row fails preregistered Gate B (only pc_fr passes both gates), so this is descriptive and cannot
be an M2 statement; skipped = CellSkip (< 20 train pairs pass f_pilot).

Paper claim: "The share of task state lost to SAE error is the same in English, French, Hindi (both scripts) and Hinglish
(people-cities 22-24%, running sum 12-15%), even where the SAE is 2.5x worse on ordinary text in that language: text-level
fidelity does not predict task-state loss."

## F3: cross-language flip-rate differences are a headroom artefact (methodological)

Flip-rate gaps differ hugely (people-cities feat_A_errB: en 7%, hi 47%, hi_latn 58%, hinglish 42%), while F2's normalized loss
does not. Mechanism (ml_flip_competition.py, ml_headroom.py): in the non-flipped Hindi/Hinglish pairs, A still beats B in
98-100% of pairs, but a third city takes top-1 (pc_hi 47%, pc_hi_latn 58%, pc_hinglish 42% vs en 5%; e.g. Oslo, London, Tokyo).
The clean A-run's lead over the best other candidate is small relative to the A-vs-B swing in these languages (median
headroom/swing 0.09-0.15 vs en 0.29, fr 0.37). One language-invariant fractional loss (median 0.11-0.13) plus each language's
own headroom predicts the observed flip rates: hi 0.59 predicted / 0.56 observed, hi_latn 0.44 / 0.44, hinglish 0.64 / 0.60,
en 0.94 / 0.93, fr 1.00 / 0.98.

Paper claim: "Top-1 flip rates confound SAE loss with the model's baseline confidence over distractors; across languages,
normalized logit-difference metrics are required (a constant 12% loss yields flip gaps from 2% to 56%)."

## F4: CE-loss-recovered (mean ablation) is not comparable across languages (methodological)

The mean-ablation denominator (CE_mean-ablated - CE_clean) ranges 2.4-9.7 nats across sets (ml_setlevel.py): English
8.0-9.7, French 6.8-8.5, Hindi Devanagari 4.1-7.3, romanized Tamil/Bengali 2.4-2.7. Because the mean is the set's own mean,
homogeneous or poorly-predicted text is cheap to mean-ablate and every splice cost looks larger. Two of the frozen M1
"supported" rows are entirely denominator-driven: hi_latn_dakshina L12 recovered 0.966 vs en 0.986 (M1 supported), yet its splice
cost equals English's (ratio 1.01 [0.96, 1.06]); ta_latn_dakshina L12 recovered 0.965, yet its splice cost is lower than English's
(0.79 [0.74, 0.83]). Zero-ablation recovered for ta_latn L12 is 0.991 (en 0.990).

Paper claim: "Recovered-loss with mean ablation depends on the dataset's own ablation baseline (2.4-9.7 nats here); report the
raw splice cost (nats/token) for cross-language comparisons."

## F5: the fertility correlation is a script effect

Across 12 sets, tokens-per-word correlates with recovered (Spearman -0.93 at L12, -0.97 at L18) and with ce_delta (0.74 / 0.84),
but within the 6 Latin-script sets fertility does not predict ce_delta at L12 (rho -0.03; ta_latn has 3.58 tokens/word and the
lowest splice cost, 0.090) and hi_deva_web vs hi_latn_dakshina have equal fertility (1.96 vs 1.95) and 2.5x different splice cost
(ml_fertility.py). Within the 6 native-script sets, more fragmented text costs more (rho 0.83 at both layers; Hindi < Tamil,
Bengali), so fragmentation matters in addition to script. n = 6 per group: weak evidence.

## F6: depth profile (H2) and L0

- The non-English / English splice-cost ratio is roughly depth-stable (hi_deva_web 2.12, 2.50, 2.29, 2.65 at L6/12/18/24;
  hi_deva_dakshina 2.08, 2.46, 2.13, 2.27), so the absolute deficit grows with depth as English's own cost grows 5.5x
  (0.064 -> 0.352 nats). French is the exception: 1.12-1.18 up to L18, 1.57 at L24. Romanized Hindi is non-monotone (1.71,
  1.01, 1.41, 1.60).
- The native-script excess beyond reconstruction size appears after L6 (F1 OLS: none at L6, +0.07 / +0.12 / +0.46 at
  L12 / 18 / 24).
- L0 flips with depth: at L6 every Indic set (both scripts) and Hinglish uses 1.5-1.6x English's active latents; at L12 ~1x;
  at L18 0.58-0.87x (ml_ratios.csv). Deeper SAEs explain non-English text with fewer active latents and not less error.

## F7: shift, not tail (H4)

80-98% of native-script windows exceed English's 90th-percentile splice cost (L12/L18), and the worst 10% of windows carry only
16-19% of the excess. Quantile ratios are largest at the low end (q10 ratio 5.6-8.6 at L12) and ~1 at q99: native text has a
cost floor that English's easy windows do not (ml_tails.py).

## Nulls and limits

- Within a set, window covariates explain little (|Spearman| <= 0.47 with |eps|/|x|, <= 0.37 with ce_clean; fragmentation ~0):
  windows hardly vary in tokenization within a set (ml_window_reg.py).
- Coarsened matching on (ce_clean, FVU) is infeasible: English and non-English FVU distributions barely overlap, so the matched
  estimates in ml_matched.csv rest on tiny strata (CIs often undefined); the regressions extrapolate.
- Confounds stated in the addendum remain: genres differ (Pile vs FineWeb-2 vs Wikipedia vs tweets); the Dakshina native and
  romanized windows come from overlapping but not identical sentences; one model and one SAE family. Per-token data were not
  saved, so token-level mechanisms (e.g. word-continuation pieces) cannot be tested on DEV.
- E14 cross-language rows are not readable under the preregistered gates; F2/F3 are descriptive.

## Suggested next compute (if any)
Per-token splice cost on the Dakshina pairs (word-initial vs continuation pieces), and the same native/romanized contrast at 65k
(E21 runs it at the set level).

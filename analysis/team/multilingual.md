# Multilingual & width (E21, E22): role report

EXPLORATORY, post hoc (2026-11-19). Nothing here revisits a preregistered answer. Gemma-2-2b, Gemma Scope residual SAEs.
Data are only what is on disk: E12/E21 per-window npz (`$SP/ana_runs/E12/*`, width/L0 from each `manifest.json` config),
E22 run `$SP/ana_runs/E22/1c85291e335f` (per_pair.parquet, directions/*.npz|json), `$SP/r5/tables/E22_*.csv`,
`$SP/hw/tables/final/E13_behaviour_lang.csv`, `E14_lang_gap.csv`. E23 (norm-matched noise controls) was NOT run.

Scripts (run from `$SP/hw` with `OMP_NUM_THREADS=2`; outputs in `$SP/team/multilingual_out/`):
- `team/scripts/multilingual_load.py` -> `windows.parquet` (one row per set x layer x width x window; checks pairing)
- `team/scripts/multilingual_width.py` -> `E21_width_by_set.csv`, `E21_contrasts.csv`, `E21_shift.csv`, `E21_ols.csv`, `E21_window_paired.csv`
- `team/scripts/multilingual_width_elasticity.py` -> `E21_elasticity.csv`
- `team/scripts/multilingual_fragility.py` -> `fragility_ablation_ratios.csv`, `fragility_window_rho.csv`
- `team/scripts/multilingual_e22.py` -> `E22_transfer_paired.csv`, `E22_geometry.csv`, `E22_e13_gates.csv`

Definitions. Splice cost `ce_delta` = sum(NLL_splice - NLL_clean)/sum(pred) (nats/token, pooled over windows); FVU = sum|eps|^2 /
sum|x - mean|^2; `sens` = ce_delta / FVU (cost per unit FVU); `nz` = ce_delta / zero-ablation range = 1 - recovered_zero; `nm` =
1 - recovered_mean. "Native" = Devanagari / Tamil / Bengali script; "romanized" = Dakshina Latin transliterations of the same
sentence pool. Windows: 2000 per set (Hinglish 1663), 124 positions each.
Statistics: 2000 bootstrap resamples of windows within set. The 16k and 65k runs score the **identical windows** (nll_clean,
nll_mean, nll_zero, pred, fvu_den equal for all 22 set x layer cells; `multilingual_load.py`), so width contrasts are paired
per window. Two-sided percentile bootstrap p-values (floor 1/2000), BH within each named family (stated where used).

SAEs: 16k at L12 L0 82 / L18 L0 74 (pilot variants, same as E12); 65k at L12 L0 72 / L18 L0 64 (`l0_rule: match16k`, the
nearest available). "Matched L0" is nominal only: observed L0 per token is 6-15 % LOWER at 65k in every set (ratio 0.85-0.94,
`E21_width_by_set.csv`). Any width gain is therefore obtained with fewer active latents.

---------------------------------------------------------------------------------------------------------------------------

## Findings ranked by strength

### 1 (strong). The native-script cost excess replicates at 65k on the same sentences (FACT)

Dakshina natural experiment at 65k (`E21_contrasts.csv`, rows `* nat/rom`); native / romanized ratios, 95 % CI:

| lang | layer | splice cost 16k | splice cost 65k [CI] | FVU ratio 65k | sens ratio 65k [CI] | 1-rec_zero ratio 65k [CI] | 1-rec_mean ratio 65k [CI] | L0 ratio 65k |
|---|---|---|---|---|---|---|---|---|
| hi | 12 | 2.44 | 1.94 [1.89, 2.00] | 0.967 | 2.01 [1.96, 2.07] | 1.54 [1.50, 1.59] | 1.57 [1.52, 1.61] | 0.99 |
| hi | 18 | 1.51 | 1.63 [1.59, 1.67] | 1.007 | 1.62 [1.58, 1.66] | 1.00 [0.97, 1.03] | 1.30 [1.27, 1.34] | 1.10 |
| ta | 12 | 4.13 | 4.25 [4.10, 4.42] | 0.896 | 4.74 [4.57, 4.93] | 3.19 [3.08, 3.32] | 3.06 [2.95, 3.18] | 1.27 |
| ta | 18 | 1.73 | 3.97 [3.84, 4.13] | 1.051 | 3.78 [3.65, 3.93] | 2.47 [2.38, 2.56] | 2.64 [2.55, 2.73] | 1.31 |
| bn | 12 | 1.93 | 2.69 [2.61, 2.77] | 0.981 | 2.74 [2.66, 2.83] | 1.82 [1.77, 1.88] | 2.16 [2.09, 2.23] | 1.15 |
| bn | 18 | 2.29 | 2.68 [2.60, 2.76] | 1.103 | 2.43 [2.36, 2.50] | 1.31 [1.28, 1.35] | 1.91 [1.86, 1.96] | 1.18 |

- With a 4x wider SAE, same-size error (FVU ratio 0.90-1.10) still costs the model 1.6-4.2x more next-token loss in native
  script; 6/6 cells, every CI excludes 1 (BH over 6 Dakshina tests: all q <= 0.002). The 16k values reproduce the prior
  analysis exactly (hw/analysis/deep/multilingual.md F1: 2.44, 1.51, 4.13, 1.73, 1.93, 2.29).
- Alternative metrics: cost per unit FVU (sens) 6/6 > 1; mean-ablation-normalised 6/6 > 1; zero-ablation-normalised 5/6 > 1 — the
  Hindi L18 cell disappears under zero-ablation normalisation (1.00 [0.97, 1.03] at 65k, 0.93 at 16k) because native text has a
  1.63x larger zero-ablation range (`fragility_ablation_ratios.csv`). L0 is not the explanation (native/romanized active latents
  0.99-1.31). KL was not saved, so it could not be checked.
- Shift, not tail, holds at 65k (`E21_shift.csv`): 74-98 % of native-script windows exceed English's 90th-percentile splice cost
  (16k: 80-98 %).
- Pooled window OLS (`E21_ols.csv`; ce_delta ~ |eps|/|x| + ce_clean + group, English baseline, 200 bootstraps within sets):
  native-script excess at 65k +0.042 [0.030, 0.051] nats (L12) and +0.062 [0.048, 0.077] (L18), vs 16k +0.066 [0.054, 0.079] and
  +0.115 [0.100, 0.127] (the 16k values reproduce F1). With FVU as covariate: 65k +0.130 / +0.105, 16k +0.163 / +0.137.
- Caveats: same as F1. Romanized text is far less predictable (ce_clean ratio native/romanized 0.35-0.60), so predictability
  cannot be matched away. Dakshina native and romanized windows come from overlapping but not identical sentences. One model, one SAE family.

### 2 (strong). Width lowers cost everywhere but helps native script LESS than English, because it removes less native error (FACT + INTERPRETATION)

Per set, 65k / 16k paired ratios (`E21_width_by_set.csv`, `E21_elasticity.csv`):
- Splice cost falls in 23/24 set x layer cells (BH q < 0.05 over 24). The exception is romanized Hindi L12: 0.990 [0.971, 1.011].
  FVU falls in 24/24 (ratios 0.76-0.92) and FVU falls in 100 % of individual windows in every cell. 65k is cheaper in 51-99 % of
  windows (`E21_window_paired.csv`).
- English: -27.3 % cost at L12 (ratio 0.727 [0.711, 0.741]) and -21.7 % at L18 (0.783 [0.771, 0.796]). Native-script sets:
  -15 % to -22 % (ratios 0.790-0.850 at L12, 0.779-0.850 at L18). Relative gain minus English's (BH over 22, 19 q < 0.05):
  all 6 native sets are below English at L12 (-0.063 to -0.123, all CIs exclude 0); at L18 4/6 are below English (-0.021 to
  -0.067), while hi_deva_dakshina (+0.004 [-0.009, 0.018]) and bn_beng_dakshina (-0.013 [-0.027, 0.001]) are not different.
- Decomposition log(cost ratio) = elasticity x log(FVU ratio): the wider dictionary removes less of the native-script error. Native
  log FVU change is -0.085 to -0.107 at L12 vs English -0.160, and -0.119 to -0.135 at L18 vs -0.185. The native-minus-English
  difference has a CI excluding 0 in 12/12 cells. Elasticity, i.e. how much cost falls per unit of relative FVU removed, is not
  lower for native script: Dakshina native 2.31-2.59 vs English 2.00 at L12 and 1.64-1.93 vs 1.32 at L18 (CIs exclude 0). Web
  native sets are about equal to English (1.79-1.89 vs 2.00 at L12; 1.31-1.42 vs 1.32 at L18).
- Marginal cost of the error that width removed (Δcost/ΔFVU, nats per unit FVU): native 1.5-2.4 vs English 0.87 (L12) / 1.06 (L18).
  This matches the average exchange rate (sens): native error removed by width was about 2x as costly per unit as English error.
- Which languages gain most: romanized Indic, but erratically: ta_latn L18 -65 % (0.350 [0.340, 0.359]), bn_latn L12 -42 %,
  bn_latn L18 -32 %, hi_latn L18 -28 %, and hi_latn L12 0 %. Then English and Hinglish L18 (-26 %), then French (-20/-22 %). The
  native scripts gain least; native web text gains the least of all (-15 to -18 %).
- INTERPRETATION: a 4x wider dictionary at fewer active latents reconstructs English (and, erratically, romanized text)
  proportionally better than native-script text. Native-script error is no less costly per unit removed. SPECULATION: the SAE
  training distribution is English/Latin-heavy, so extra latents go to Latin-script structure. This was not testable here.
- Alternatives: lower L0 at 65k (it should hurt all sets, and could hurt native text more if native windows need more latents);
  bootstrap CIs cover window sampling only, not SAE-training variability (one SAE per width).

### 3 (moderate-strong). Relative to English, the script effect is width-invariant per unit FVU; relative to romanized text it is not (FACT)

- Native / English (`E21_contrasts.csv`, `* / en` rows): splice-cost ratio at 65k L12 2.67-3.88 (16k 2.46-3.39), L18 2.12-3.21
  (16k 2.13-2.99). The ratio-of-ratios (65k/16k) for cost: L12 1.09-1.17 for all 6 native sets (CIs exclude 1); L18 0.995-1.085
  (hi_deva_dakshina 0.995 and bn_beng 1.017 include 1). Cost per unit FVU (sens) native/English at 65k: L12 1.86-2.50, L18
  1.43-1.86; ratio-of-ratios 0.94-1.09 (within ±9 % of width-invariant). The absolute excess over English shrinks by 0.018-0.055
  nats/token but stays 0.139-0.239 nats at L12, against English's own 65k cost of 0.083.
- Native / romanized: ratio-of-ratios hi L12 0.80 [0.78, 0.82], hi L18 1.08, ta L12 1.03 [1.00, 1.06] (n.s., q = 0.08),
  ta L18 2.30 [2.24, 2.37], bn L12 1.39, bn L18 1.17. The changes come from the romanized denominators, whose costs move a lot
  across SAEs (elasticities 0.09-4.19 vs 1.3-2.6 for all other sets). The romanized-Tamil L18 16k cost (0.271, three times its
  L12 cost of 0.090) looks like a 16k-specific failure that 65k removes (0.095).
- INTERPRETATION: "native script costs about 2x English per unit of reconstruction error" survives a 4x change of dictionary size.
  The exact native/romanized multiplier does not; report it as a range (1.5-4.2x), not as a constant.

### 4 (strong for RS, uninformative for PC). E22: the English-fitted running-sum DAS subspace transfers fully to French, Hindi and Hinglish (FACT)

Test split, f_A_correct, flip rate (eligible n 162-180 per family), random-U p95 <= 0.006 (20 Haar draws per rank);
`E22_transfer_paired.csv`; paired bootstrap on the same pairs; BH over 48 transfer-minus-native tests:

| RS L18 | English home | fr: Eng-fit / native-fit | hi: Eng-fit / native-fit | hinglish: Eng-fit / native-fit |
|---|---|---|---|---|
| das4 full_along_U | 0.539 | 0.593 / 0.204 | 0.503 / 0.246 | 0.525 / 0.384 |
| das8 full_along_U | 0.611 | 0.580 / 0.481 | 0.632 / 0.842 | 0.605 / 0.588 |
| das4 feat_A_along_U | 0.306 | 0.364 / 0.142 | 0.298 / 0.088 | 0.243 / 0.186 |
| das8 feat_A_along_U | 0.333 | 0.383 / 0.290 | 0.351 / 0.673 | 0.367 / 0.311 |
| DAS train acc (das4 / das8) | 0.633 / 0.687 | native 0.247 / 0.593 | native 0.160 / 0.880 | native 0.453 / 0.580 |

- All 12 English-fit transfer cells are "effective" under the E9 rule (flip >= p95 + 10 pts, lower CI > p95). They are as strong
  as in English itself. All 6 transferred full_along_U rows meet the frozen illusion and feature-mediated rules
  (`r5/tables/E22_illusion.csv`). The full-minus-feature difference under transfer is 0.20-0.28 (English home 0.23 / 0.28;
  native das8 fits 0.17-0.28).
- English fit vs native fit: the English fit wins wherever the native fit failed to train (das4: +0.39, +0.26, +0.14 full; all
  q <= 0.005; native das4 accuracies are below the 0.5 validity bar). For das8, where the native fits are valid, the English fit is
  +0.099 [0.012, 0.191] in French (q = 0.11), +0.017 (n.s.) in Hinglish, and -0.211 [-0.295, -0.128] in Hindi (q = 0.002). The
  Hindi das8 fit has the best training accuracy of any fit in E22 (0.880). Same signs with f_none and with NLDR.
- PCA "error directions" (pca_pooled r = 4/8) are not effective in any language under either fit, including English (max flip
  0.067 at English r8 full_along_U, against a bar of p95 + 0.10 = 0.106). Their transfer is moot behaviourally. (Some NLDR
  differences reach BH significance at magnitudes <= 0.03; they are negligible.)
- People-cities L12: no transfer (English-fit flip <= 0.047 in fr / hi / hinglish vs native das8 0.320 / 0.107 / 0.160; paired
  differences -0.273, -0.100, -0.140, q = 0.002). Uninterpretable under the frozen rules, because every PC DAS fit, including
  English (0.093 / 0.373), is invalid (< 0.5). The English fit does not even meet validity at home.
- Relation to E13 gates: transfer does not track behavioural accuracy. rs_hi fails Gate A (clean acc 0.536 [0.489, 0.581]), yet
  English-fit flips there (0.50 / 0.63) are as high as in rs_fr (acc 0.802) and rs_hinglish (0.678). Only 3 languages, so no
  correlation is computed. Flip is evaluated on f_A_correct pairs, i.e. on the subset where the model does the task.

### 5 (moderate). Geometry: the running-sum error directions and the DAS subspaces agree across languages, but agreement does not predict transfer (FACT)

`E22_geometry.csv`; Haar p99 for the top principal cosine: k=4 0.097, k=8 0.131.
- Error directions (pca_pooled): RS is language-stable. Top principal cosine English vs fr / hi / hinglish is 0.956 / 0.926 / 0.916
  at r = 4; the rank-1 vectors give |cos| 0.86 / 0.92 / 0.89. PC is language-specific: en-fr 0.65, en-hi 0.22, en-hinglish 0.25
  (rank-1 |cos| 0.57 / 0.02 / 0.11), and hi-hinglish 0.67. This replicates deep analysis E (0.86-0.92; PC fr 0.57, hi 0.02).
- DAS: raw cross-language cosines (0.71-0.99) are dominated by the shared initial rotation. A cross-task control (RS L18 vs PC L12
  fit) gives raw 0.71-0.96 but learned-component cosines of only 0.05-0.10. Learned components (init projected out,
  make_final.das_learned) across languages: RS das8 top 0.81-0.86 (mean 0.47-0.52); PC das8 top 0.76-0.86 (mean 0.47-0.53).
  The cross-language agreement is real, and it is as high for PC (no behavioural transfer, invalid fits) as for RS (full transfer).
- The RS DAS learned subspace puts 8-16 % of its energy in its own language's top-8 error PCs. PC's puts 0.4-1.6 % (Haar
  expectation 0.35 %). The causal RS variable partly overlaps the error's principal directions; PC's does not.
- INTERPRETATION: subspace similarity is necessary but not sufficient for causal transfer. Behaviour must be tested directly.

### 6 (moderate, replication). Task-state loss stays language-invariant while text fidelity does not (FACT; prior F2)

- E22's feat_A_errB rows reproduce E14 bit-for-bit (2945 pair-condition rows, max |Δlogit| = 0 between E14 on gnode084 and E22 on
  gnode058). This is a reproduction, possibly through cached activations, and not an independent sample. Gap = 1 - NLDR
  (f_A_correct): RS en 0.148 [0.124, 0.174], fr 0.125, hi 0.123, hinglish 0.116; PC en 0.237 [0.225, 0.248], fr 0.241, hi 0.244,
  hinglish 0.221 (`r5/tables/E22_conditions.csv`).
- At the same SAE and layer (L18 16k), ordinary Devanagari Hindi text costs 2.1-2.3x English per token (E12). On the RS task, Hindi
  loses a fraction of task state equal to or smaller than English's.
- Important alternative explanation (new, from reading the prompts): the task-critical tokens are script-neutral. rs_hi uses Arabic
  digits ("पहले 1 लें। 2 जोड़ें।"), and pc_hi uses Latin city names (E13: "Deva (city answers Latin)"). The language-invariance of
  task-state loss and of the RS subspace may reflect identical answer and operand tokens rather than a language-general mechanism.
  The data cannot separate the two. A test would need Devanagari numerals or Devanagari city names.

### 7 (weak). Direction vs size without E23: what can and cannot be concluded

E23 (norm-matched noise, x_hat + eta / x + eta) was not run. Without it **no claim that the SAE error's direction (rather than any
same-size perturbation) causes the native-script excess is warranted**. Native-script contexts may simply be more fragile.
Existing data give only partial, indirect substitutes:
- (a) Whole-residual ablations as a "generic perturbation" (`fragility_ablation_ratios.csv`). Native/romanized ratio of
  mean-ablation cost is 1.24-1.51 and of zero-ablation cost 1.26-2.04. The splice-cost ratio is 1.51-4.25. Splice excess over
  mean-ablation excess (ratio of ratios) is > 1 in 12/12 cells (1.15-3.06). Over zero-ablation it is > 1 in 10/12 (Hindi L18
  0.93 / 1.00). So native text is somewhat more fragile to wholesale destruction, but much less so than to SAE splicing. This
  weakly favours "something specific to the SAE error". Caveats: ablations are 10-100x larger than the error, not
  norm-matched, saturate, and are bounded by clean predictability (native ce_clean is 0.35-0.60x romanized).
- (b) Width as a size manipulation on identical windows. Width shrinks the error in every window, but per window the cost
  reduction barely tracks the FVU reduction (Spearman -0.02 to 0.24 within set; `E21_window_paired.csv`). Within a set, splice cost
  correlates only weakly with FVU (|rho| <= 0.44, English highest; native <= 0.32) and with window fragility (mean-ablation cost,
  rho -0.23 to 0.35; `fragility_window_rho.csv`). At set level the native/English cost-per-FVU ratio is width-invariant (finding 3).
  Size changes within a script move cost at a script-specific exchange rate. Width changes the error's size and its direction
  together, so this is not a noise control.
- Conclusion: the robust statement is "error SIZE (FVU, |eps|/|x|, L0) does not account for the native-script cost". Whether the
  cause is the error's DIRECTION or the context's general sensitivity remains open until E23 or per-token NLLs exist.

---------------------------------------------------------------------------------------------------------------------------

## Synthesis: one multilingual story (INTERPRETATION, labelled)

1. Text fidelity: the cost of SAE error on ordinary text is set by script, not by reconstruction size. The effect holds on
   identical sentences, at two widths, under four metrics (splice cost, cost per FVU, mean- and zero-ablation-normalised: 23/24
   cells), and it is a whole-distribution shift. A 4x wider SAE lowers every cost but keeps the native/English exchange rate
   (cost per unit FVU about 1.9-2.5x at L12). It improves native text proportionally less (FVU -8 to -10 % vs -15 % for English
   at L12), so scaling width does not close the multilingual fidelity gap. It slightly widens the gap in relative terms at L12.
2. Task state: on structured tasks the SAE removes a fixed fraction of the decision signal in every language (RS 12-15 %, PC
   22-24 %), independent of the 2-3x text-level deficit.
3. Causal subspace: the running-sum variable lives in a subspace that is shared across English, French, Hindi and Hinglish. An
   English-fit DAS rotation works as well abroad as at home, and both its feature-carried part and its error-carried part
   (full - feature 0.20-0.28) transfer. For people-cities, DAS never trained validly, so nothing can be said.
4. Reconciliation (SPECULATION): the native-script deficit concerns the script's own token-level structure, such as word pieces of
   Devanagari / Tamil / Bengali. The task variables (digits, Latin city names) are script-neutral and represented
   language-generally, so both the SAE's loss on them and their causal subspace carry over unchanged. Prediction (untested):
   tasks whose critical tokens are themselves in native script would show the text-level penalty in task-state loss.
5. Methodological, for any multilingual SAE audit: (i) mean-ablation recovered is not comparable across languages (F4), and
   zero-ablation normalisation can also erase a real effect (Hindi L18); use raw splice cost plus cost per FVU. (ii) Flip-rate
   differences are headroom artefacts (F3); use NLDR. (iii) Report the native/romanized multiplier as SAE-dependent. (iv) Check
   DAS validity per language before reading transfer, and project out the shared DAS initialisation before reading geometry.

## Multiplicity
E21: BH within families: width ratios (24 each for cost / FVU / sens), gain-minus-English (22), contrasts at 65k and
ratio-of-ratios (6 Dakshina + 22 vs-English per metric), excess change (28). E22: 48 paired transfer tests per metric (flip,
NLDR), BH. The OLS, elasticity, fragility and geometry numbers are descriptive and uncorrected. About 400 bootstrap intervals in
total; most headline effects have q <= 0.002.

## What failed / nulls
- Romanized Hindi L12 gains nothing from width (0.990 [0.971, 1.011]).
- No PCA error-direction subspace is causally effective in any language, so error-direction "transfer" has no behavioural content.
- PC DAS: all 8 fits are invalid, so the PC transfer contrast is unreadable.
- Zero-ablation normalisation removes the Hindi L18 script effect.
- The E22 / E14 agreement is bit-identical, so it is not independent evidence.
- KL, per-token NLLs and norm-matched noise do not exist on disk.

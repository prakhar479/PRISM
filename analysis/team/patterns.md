# PATTERN HUNTER: unexpected regularities in the PRISM results (2026-11-19)

**Status:** everything here is exploratory and post hoc. I found these patterns bottom-up, by mining the data, after seeing it.
- **Scripts:** `$SP/team/scripts/patterns_*.py`. The shared loader is `patterns_common.py`.
- **Outputs:** CSVs in `$SP/team/patterns_out/`.
- **Data used:** per-pair parquet files in `$SP/ana_runs/E*/` and prompt files in `$SP/hw/data/prompt_sets/`. Token strings were decoded with the local Gemma-2 tokenizer (`~/.cache/huggingface/hub/models--google--gemma-2-2b`).
- **Pairs:** unless stated otherwise, analyses use eligible test pairs, meaning clean A predicts A and clean B predicts B. Counts: RS L18 n = 203, OT L12 n = 98, PC L12/L16 n = 151.
- **Per-pair NLDR** = (D − D_B)/(D_A − D_B), with D_A and D_B taken from clean_A and clean_B.
- **Split-half checks:** pairs are split at random into two halves. Each finding states its multiplicity handling (BH or uncorrected).
- **Not run:** E23 (norm-matched noise) was not run, so nothing below uses it.

---

## 1. RS steering is a pair-independent push along the number line, and works only on pairs where A > B (strong, replicated)

**FACT (E19, RS L18, n = 203 eligible; `patterns_steering.py` → `steer_rs_numberline.csv`, `steer_rs_sign.csv`)**

Steering adds the same vector to every pair: s·|m|·P_U m/|P_U m|, with m = the train mean of x_A − x_B. In the train set A − B is balanced: mean −0.04, A > B in 48.8 % of 900 pairs.

**Flips depend entirely on the sign of A − B, and the sign depends on the estimator:**

| U, scale (where flips peak) | flip if A > B (n = 106) | flip if A < B (n = 97) | Fisher p | split-half asymmetry |
|---|---|---|---|---|
| das4, s = 4 | 0.320 | 0.000 | 1.3e-11 | +0.42 / +0.23 |
| das8_e10, s = 8 | 0.254 | 0.010 | 6.6e-8 | +0.25 / +0.24 |
| das8, s = 16 | 0.339 | 0.103 | 5.1e-5 | +0.18 / +0.29 |
| answer_sep r1, s = 16 | 0.273 | 0.000 | 8.2e-10 | +0.30 / +0.25 |
| pca_pooled r1, s = 16 | 0.018 | 0.288 | 3.0e-8 | −0.24 / −0.28 |
| pca_pooled r4, s = 16 | 0.009 | 0.237 | 1.7e-7 | −0.23 / −0.22 |

All six survive BH over the 9 estimator rows. answer_sep r4/r8 and pca r8 show no asymmetry (q ≥ 0.06).

**The top-1 digit moves monotonically with dose.** Digits were tracked across s = 0.25…16; the output "1" is coded as 10 (see the interpretation).
- das4: every pair's answer moved, and 100 % of them moved monotonically upward. Mean digit change from s = 0.25 to s = 16 is +2.37.
- pca r1 and r4: 86 % and 96 % of moved pairs moved monotonically downward.
- Pooled over pairs, das4 shifts the top-1 distribution upward: "9" goes from 0.24 to 0.40 at s = 4. Then "1" takes over: 0.21 at s = 4, 0.47 at s = 8, 0.83 at s = 16.
- Random-U steering at s = 16 does not produce "1" and does not collapse onto a single digit. Each random direction's modal digit is 8, 7 or 9 with share ≤ 0.41 (`steer_random_attractors.csv`).

**Replication:**
- E10 (gnode084, older run): das1/2/4 flip 0.27–0.37 for A > B and 0.000 for A < B at every scale.
- E10b (fresh templates rs_plus/story/symbolic, 534 eligible pairs): das1/2/4 at s = 4 flip 0.27–0.36 for A > B and 0.000 for A < B. pca r1 flips 0.024 for A < B and 0 for A > B.

**INTERPRETATION**
- The steering vector projected onto these subspaces acts as an ordinal "increase / decrease the running total" push. It does not carry a pair-specific value.
- With balanced A − B signs, a fixed push can flip at most the pairs on one side. RS therefore cannot meet the frozen Q2 criterion (steer flip ≥ 50 %) by construction. This is a design property, not evidence about the error.
- The overshoot to "1" at high dose is plausibly the first token of a two-digit number ("10"…): the model "counts past 9". This is SPECULATION; there is no logit lens on "10" to check it.
- Why P_U m points "up" for DAS and "down" for PCA is unexplained. The design is sign-balanced, so the push must come from a small systematic component of m.

**Caveat:** same pairs and same DAS settings across E10 and E19. The fresh-template E10b and the split halves are the independent checks.

---

## 2. People-cities DAS is city-selective: which city A is explains 57–71 % of the per-pair DAS effect (strong within English; weak across languages)

**FACT (`patterns_answer_selectivity.py` → `answer_selectivity.csv`; `patterns_pc_cities.py` → `pc_city_means.csv`, `pc_city_rank_corr.csv`; 1164 tests, BH; permutation p from 500 shuffles)**

**Variance explained by A's city.** Measured as eta² of per-pair full/feat_along_U NLDR on A's city, across 16 levels, median over DAS keys:

| cell | eta² | null p95 | DAS keys with q < .05 | split-half ρ of per-city means |
|---|---|---|---|---|
| E18 PC L12 | 0.571 | 0.161 | 10/10 | 0.67 |
| E18 PC L16 | 0.705 | 0.161 | 10/10 | 0.75 |
| E22 pc_en | 0.53 | – | – | – |
| E22 fr | 0.45 | – | – | – |
| E22 hi | 0.34 | – | – | – |
| E22 hinglish | 0.32 | – | – | – |

- All four E22 cells have every DAS key at q < .05. Their null p95 is 0.08–0.17.
- B's city explains much less: eta² 0.19–0.22.
- **RS and OT show no comparable answer selectivity.** DAS eta² sits at its null in every RS/OT cell: for example OT L12 0.09 vs null 0.125, and RS L18 0.08 vs 0.08–0.22 (story/calc, null depends on the cell's n). This is a null finding.

**Examples (full_along_U das4 NLDR by A's city, E18 L12 / L16):**
- Rome 0.59 / 0.60, Athens 0.57 / 0.38, Dubai 0.42 / 0.40.
- Vienna 0.06 / 0.07, Berlin 0.11 / 0.11, Prague 0.14 / 0.02.
- **das1 is effectively an "Athens" direction:** Athens 0.43 at both layers, every other city ≤ 0.14.

**City rankings across cells (Spearman over 16 cities, 153 correlations, BH):**
- Same DAS rank, L12 vs L16: ρ = 0.78 (das1), 0.77 (das2), 0.82 (das4), 0.86 (das8), all q ≤ 0.011. das16 gives ρ = 0.14.
- Across languages (E22 fr/hi/hinglish vs English), same rank: median ρ 0.42, range −0.23 to 0.74. Only 6/33 cross-cell same-key pairs reach q < .05.

**INTERPRETATION**
- A rank-k subspace cannot hold a 16-way categorical variable. DAS seems to learn a few city-identity contrasts, and patching restores only pairs whose A city is among them.
- The aggregate PC DAS NLDR (≈ 0.2–0.3) is a mixture: some cities are restored at about 0.6 and others at about 0.

**ALTERNATIVE:** the L12/L16 agreement may come from the shared DAS initialisation (same (k, seed) → same starting rotation; see deep_analysis). There are no PC seed replicates to separate the two.

**Caveat:** E22 non-English rows pool the native-fit and English-fit directions (300 rows per key = 2 fits × ~150 pairs).

---

## 3. Per-pair response matrices are low-rank; the main axis is "how patchable by a subspace", which is unrelated to "how much the error costs" (moderate)

**FACT (E18; `patterns_profiles.py` → `profiles_svd.csv`, `profiles_loadings_*.csv`, `profiles_assoc.csv`)**

Profile = per-pair NLDR over 86 conditions: feat_A_errB, feat_A_meanPos, 3 conditions × 3 estimators × ranks, plus random-U means. NLDR is clipped to [−1, 2] and columns are centred.

| cell | variance explained (PC1 / PC2 / PC3) | column-permutation null p95 for PC1 | split-half \|cos\| of V1 / V2 (null) |
|---|---|---|---|
| RS L18 | 0.385 / 0.291 / 0.067 | 0.081 | 0.79 / 0.78 (0.71) |
| OT L12 | 0.715 / 0.113 / 0.061 | 0.120 | 0.98 / 0.84 (0.26) |
| PC L12 | 0.456 / 0.274 / 0.102 | 0.135 | 0.97 / 0.96 (0.23) |
| PC L16 | 0.548 / 0.223 / 0.065 | 0.130 | 0.97 / 0.95 (0.45) |

- **Mode 1 (all cells)** loads on the DAS full_along_U and feat_A_along_U columns. In RS it also loads on error-PCA/answer_sep at r = 32–64.
- **Mode 2 in RS** loads equally on feat_A_errB and on every repair column, including random-U repair. Repair is anchored on x̂_A + ε_B, so this mode is the feature-only cost.
- In RS the two modes are nearly degenerate. That explains the weak split-half stability of the individual vectors.
- **Cost vs patchability per pair** (Spearman of feat_A_errB NLDR vs das4 full_along_U NLDR):
  - RS −0.15 (halves −0.31 / +0.03)
  - OT −0.05 (halves −0.07 / −0.05)
  - PC L12 +0.07 (halves −0.06 / +0.17)
  - PC L16 +0.21 (halves −0.00 / +0.38)

  None is consistent across halves.

**Mode 1 is explained by prompt features** (138 association tests, BH; top hits replicate in both halves):
- **PC:** A's city (Kruskal p < 1e-4), as in §2.
- **OT:** the corrupted step (ρ = −0.63, halves −0.59 / −0.63), see §5.
- **RS:** the start number (ρ = 0.45, halves 0.37 / 0.53), see §4.

**k-means:** k = 2 or 3 splits along mode 1 (OT: along corrupt_step). I found no discrete pair types beyond that. This is a null for "clusters".

**INTERPRETATION:** the two pair properties the project measures (error cost and subspace patchability) are close to independent at the pair level. One should not expect pairs that "need the error" to be the ones a low-rank direction repairs.

---

## 4. RS: pairs that start at 0 are the least patchable by any low-rank subspace (moderate; replicated on independent pairs)

**FACT (`patterns_rs_start.py` → `rs_start_regression.csv`; `patterns_rs_start_e22.py` → `rs_start_e22.csv`)**

**E18 RS L18 (n = 203), full_along_U NLDR by the chain's start number:**

| start | n | das4 | pca r8 |
|---|---|---|---|
| 0 | 78 | 0.65 | 0.05 |
| 1 | 45 | 0.77 | 0.14 |
| 2 | 46 | 0.85 | 0.15 |
| 3 | 22 | 1.00 | 0.32 |
| 4 | 12 | 0.89 | 0.17 |

feat_A_errB moves the opposite way: 0.89 at start 0, 0.69 at start 3.

**OLS coefficient on start, controlling for log margin, template and corrupt_step:**
- das1 +0.094 (SE 0.020)
- das4 +0.079 (0.018)
- pca r8 +0.053 (0.011)
- answer_sep r8 +0.053 (0.011)
- All p ≤ 0.005, with the same sign in both halves.

Adding A, B, |A − B| and both increments as covariates attenuates the start effect, but it stays: das4 +0.050 (p = 0.019), pca r8 +0.034 (p = 0.008). For feat_A_errB it becomes non-significant (−0.014, p = 0.32).

**Independent pairs (E22, e13 prompt set; 16 tests, BH):** ρ(full_along_U NLDR, start) is positive in 15/16 family × estimator cells. It reaches q < .05 in 12/16, with ρ 0.16–0.31:
- **DAS:** all 8 cells positive across en/fr/hi/hinglish.
- **PCA:** null in Hindi (ρ 0.03 / −0.02).

**INTERPRETATION:** when the chain starts at 0, the state may be encoded differently (the first increment is itself the running total). Low-rank directions fitted on all pairs then capture it less well. SPECULATION.

**Caveats:** start is correlated with the answer and the margin. The covariate model is linear. The effect is modest (about 0.05–0.10 NLDR per unit of start).

---

## 5. OT: pairs whose second move is corrupted respond 1.4–2× more to along-U patches and lose more under the feature-only patch (strong within the pilot template)

**FACT (`patterns_ot_step.py` → `ot_step.csv`; 110 Mann-Whitney tests, BH)**

Pairs: corrupt_step 0 (n = 36, median 18 patched positions) vs corrupt_step 1 (n = 62, 11 positions).

| median NLDR | step 0 | step 1 | estimators with q < .05 |
|---|---|---|---|
| DAS full_along_U, E18 16k | 1.50 | 2.14 | 5/5 |
| pca full_along_U, E18 16k | 0.35 | 0.58 | 7/7 |
| feat_A_errB, E18 16k | 0.81 | 0.73 | – |
| feat_A_errB, E20 65k | 0.82 | 0.73 | – |

The feat_A_errB step difference is q < .05 at 16k (E9, E18) and 65k, but not at 262k (0.90 vs 0.84).

- The same pattern holds in E9 (16k) and E20 at 65k and 262k. Every DAS and every full_along_U PCA/answer_sep comparison is at q < .05.
- **feat_A_meanPos shows no step effect** (q ≥ .05 in all 4 SAEs).
- NLDR > 1: along DAS directions the patched logit difference overshoots clean A's, by a median factor of about 1.5–2.2.

**INTERPRETATION:** with only one move left to update, B's "moved" information is concentrated in fewer, later positions. That makes it both more error-borne and easier to remove along one direction. The overshoot fits the prefix-fallback reading (deep_analysis §C): erasing B's update leaves A's answer with less competition than clean A has.

**Caveats:**
- corrupt_step and the number of patched positions are perfectly confounded in this template.
- E9, E18 and E20 reuse the same 98 pairs, so the replication is across SAEs and fits, not across pairs. Split halves of the E18 mode-1 association: ρ −0.59 / −0.63.

---

## 6. Which pairs are "hard" for the SAE depends on (pair, layer), not on the dictionary (moderate)

**FACT (`patterns_consistency.py` → `consistency_hardness.csv`; 267 Spearman correlations of per-pair feat_A_errB NLDR between cells, BH)**

**OT at L12, across 11 SAEs (16k to 524k, two L0s each):** median ρ = 0.55 (range 0.31–0.77, all q < .01). Regressing out each cell's log margin leaves this unchanged (median 0.54).

**OT across layers, by layer distance:**

| layer distance | median ρ | share q < .05 |
|---|---|---|
| 1 | 0.32 | 71 % |
| 2 | 0.22 | 48 % |
| 4 | 0.13 | 11 % |
| ≥ 6 | ≈ 0 | 0 % |

**RS:** weaker and flat with depth. L18 16k vs 65k ρ = 0.32. Any two layers L4–L20: 0.08–0.44, median about 0.22.

**INTERPRETATION**
- Hard pairs stay hard when the dictionary is 32× wider. This fits the deep_analysis finding that task-relevant error directions persist with width.
- The hard set reshuffles within about 2 layers, which matches the error-direction rotation half-life (about 1.5–2 layers) reported there.

---

## 7. Layer shape: B's own error pulls toward B in middle layers; A-specific error is needed in late layers (moderate)

**FACT (E8 atlas, 16k; `patterns_layers.py` → `layer_shape_E8.csv`)**

Definitions (ratio-of-means NLDR gaps, 1000-sample bootstrap):
- **B-pull** = gap(feat_A_errB) − gap(feat_A_meanPos). It is the part of the feature-only loss caused by inserting B's error rather than a mean error.
- **A-need** = gap(feat_A_meanPos). It is the loss when A's error is replaced by the mean error.

**B-pull peaks in mid layers, at the same layer in both halves:**
- OT: L12 0.193 (halves 0.18 / 0.20), against 0.06–0.08 at L11/L13.
- RS: L10 0.062 and L12 0.053 (halves 0.06 / 0.06), against ≤ 0.04 elsewhere.
- IOI: rises monotonically to L20 (0.187).
- VB: about 0.02 throughout.

**A-need rises late:**
- RS: about 0 at L4–12, then 0.06 (L14–16) and 0.12 (L18).
- IOI: 0.17 at L20.
- OT: about 0.05 at L18–20.

Split-half correlation of the gap curves: 0.97–0.999 for all four tasks.

**Anomaly:** OT L15 has a negative feature-only gap of −0.050 [−0.074, −0.025]. The feature-only patch restores A more than the clean A run does, so B's error at L15 pushes toward A. It is a single cell (n = 98) and worth a look.

**INTERPRETATION:** the RS L18 confirmatory gap (0.142) is mostly A-need (0.122): losing A's specific error rather than B's error interfering. The OT L12 gap (0.237) is mostly B-pull (0.193). These are two different ways the error can matter, and they peak at different depths.

---

## 8. Large steering collapses OT outputs onto one token; which token depends on the estimator; dose response is non-monotone (moderate, partly generic)

**FACT (E19 OT L12, n = 98; `steer_attractors.csv`, `steer_dose_shape.csv`, `steer_random_attractors.csv`)**

**Peak and collapse:**
- Flips to A peak at s = 1 for PCA and answer_sep (0.97–0.98) and at s = 0.25–0.5 for DAS (0.96–0.99). They fall to 0–0.18 by s = 16.
- 62–99 % of pairs have D peaking at an interior scale.

**Where the outputs collapse:**
- At s = 16, all 6 PCA/answer_sep U output " G" in 82–100 % of pairs (halves 0.82–1.00). " G" is the top-1 for only 6 % of pairs on clean B.
- Intermediate scales pass through " C" (s = 4) and " E" (s = 8).
- DAS directions collapse instead to non-candidate tokens: " in" (das4/das8 at s = 8, 74–78 %), " and", " //" (das8_e10 at s = 16, 99 %). This means the model leaves the answer format altogether.

**Random-U controls:**
- Random directions also collapse at s = 16: per direction, the modal-token share has median 0.85–0.97.
- Their modal token varies by direction (letters, "." or blank).
- **So the collapse itself is generic at this norm (16|m|); its identity is estimator-specific.**
- The attractor is not predicted by the train answer imbalance: " G" ranks 6th of 7 in n(A = c) − n(B = c).

**PC:** DAS steering at s = 16 drives 26–52 % of pairs to a single city: Paris (das4), Tokyo (das8), Rome (das8_e10). PCA/answer_sep never flip PC.

**RS:** das4 collapses to "1" (see §1).

**Caveat:** the s ≥ 8 results are far outside the training-norm regime. Treat them as model-behaviour curiosities, not as evidence about the error.

---

## 9. Anomaly / provenance: patched outputs are bit-identical across three machines (needs a check before claiming cross-machine replication)

**FACT (`patterns_consistency.py` → `consistency_clean_range.csv`, `consistency_featAerrB_runs.csv`)**
- clean_A, clean_B and full_A logits are exactly equal (max abs diff 0.0) for every pair across all runs.
- feat_A_errB logits for RS L18 are exactly equal between E4 (Poseidon), E9 (gnode084) and E18 / E17 / E8 / E2r (gnode058).
- The only differences come from:
  - E3 (max 0.07–0.09 logits, median 0; different prompt-set batch);
  - E22 PC (max 0.12, median 0).

**INTERPRETATION:** either the GPU forward passes are bit-deterministic across nodes, or activations and logits come from content-addressed caches shared between machines (harness/cache.py keys by model revision, prompt-set hash, layer and dtype). In the second case, the deep_analysis "gaps reproduce within ~1 point across 10+ runs on three machines" is mostly re-reading the same cached tensors, not independent recomputation.

**Suggested check:** compare cache roots / `key.json` per node. Only DAS refits, as in E17, are certainly recomputed.

---

## 10. Smaller findings and nulls

**DAS fits agree on which pairs they restore** (E17 RS, `patterns_das_seeds.py` → `das_seed_pair_corr.csv`):
- Same-rank fits (seed / epochs / lr): per-pair NLDR median ρ 0.62 (0.11–0.96, 62 pairs of fits). Different ranks or estimators: 0.45.
- Even the failed das1_e10_lr3e-2 fit (mean NLDR 0.019) ranks pairs like the others (ρ 0.55–0.76 with other das1 fits).
- This is a shared "pair patchability" factor; it is the RS mode-1 axis of §3.

**Nulls (each was tested):**
- Answer identity does not explain along-U effects in RS or OT (§2).
- Feature-only cost and patchability are unrelated per pair (§3).
- No discrete pair clusters exist (§3).
- In E18/E19 RS, the apparent "low sums fail under DAS" (ans_A 3–4: NLDR 0.16–0.44, n = 7) is not significant (eta² at null).

**Bug checks that passed:**
- Every clean_A / clean_B D is consistent across runs and cells.
- No pair has a clean-D range > 0.5.
- Pair counts per cell match the manifests: RS 215 → 203 eligible, OT 249 → 98, PC 159 → 151.

## Multiplicity summary
BH was applied within each script's test family:
- §1: 9 Fisher tests
- §2: 1164 permutation tests and 153 ranking correlations
- §3: 138 association tests
- §4: 16 replication tests
- §5: 110 tests
- §6: 267 correlations

§7 and §8 are descriptive with bootstrap CIs and split halves; no formal tests. Nothing here was preregistered.

# What does the SAE error carry for task state? Per-pair mechanism analysis

**Status:** exploratory and post hoc, run on DEV only, from existing per-pair outputs; no multiplicity correction.

**Model and SAEs:** Gemma-2-2b with Gemma Scope 16k residual SAEs. The input is every complete run in `ana_runs/` (E1, E2, E2r, E3, E8, E9, E11, E14).

**Notation:**
- `x̂` is the SAE reconstruction (the features) and `ε` is the SAE error.
- A is the clean prompt and B the corrupt prompt.
- D is the logit difference, logit(A's answer) − logit(B's answer).
- A pair is **eligible** when clean A's top-1 is A's answer and clean B's top-1 is B's answer.

**Scripts:** `deep/scripts/mech_*.py`; shared loaders are in `mech_common.py`. CSV outputs are in `deep/`. Bootstrap CIs use 2000 resamples over pairs.

---

## 1. Object tracking: B's "the object moved" information is held jointly with the error in layers 8–16, and A's answer is a prefix fallback

### Design confound (affects every OT feature-only result, including release 2's "features restore 74–100%")

The corruption changes which object moves at one step, so B always has one more move of the queried object than A (`mech_ot_parse.py`). Two consequences:
- Of the 98 eligible pilot test pairs at L12, **96 are "stay" pairs**: in A the queried object never moves, so A's answer is its initial box.
- Gemma-2-2b fails the "moved" pairs, where A has 1 move and B has 2: clean-B accuracy is 18% (`ot_roles.csv`).

The OT confirmatory contrast is therefore effectively "did the queried object move, and to where". A's answer is already implied by the unpatched prefix, before the first changed token. Destroying B-specific information at the patched positions therefore falls back to A's answer. A feature-only patch that "restores A" says little about whether the features encode A.

### The diagnostic is whether B survives without its own error

| measure | L4 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|
| P(B \| x̂_B + ε_A), pilot OT, n = 102 | 0.91 [0.84, 0.95] | 0.20 [0.13, 0.28] | 0.28 [0.20, 0.37] | **0.02 [0.005, 0.07]** | 0.08 [0.04, 0.15] | 0.19 [0.12, 0.27] | 0.57 [0.47, 0.66] | 0.71 [0.61, 0.79] |
| P(A \| x̂_B + noise) | 0.40–0.41 | 0.95–0.97 | 0.87–0.91 | 0.89–0.92 | 0.72–0.76 | 0.75–0.78 | 0.35–0.39 | 0.22 |

Sources: E1 (first row) and E8/E11 (second row); `h6_layers.csv`, `symmetry.csv`.

`x̂_B + ε_A` differs from the clean B run only by ε_B − ε_A, an in-distribution error from a prompt that differs only in the moving object's name. At L12 that swap alone turns B's answer into A's in 98% of pairs. With the error swapped the other way, `x̂_A + ε_B`, the answer stays A (100%).

The **2×2 factorial** (`mech_factorial.py`, `factorial.csv`) crosses features from A or B with error from A or B. Its interaction term isolates the joint (non-additive) contribution. At L8–16 the interaction is −0.25 to −0.38 of the total A–B logit difference (L12: −0.38 [−0.42, −0.35]). At L4 it is about 0, and at L18–20 about 0 as well.

Reading: B's answer needs **both** B's features and B's error. Removing either collapses to the prefix answer.

### A single direction found from the error alone carries it

E9, L12, eligible stay pairs (`ot_alongU_fallback.csv`):
- **Full residual along error-PCA rank 1:** patching A's component into B along this one direction, `x_B + P_U(x_A − x_B)`, flips 0.89. Random rank-1 directions flip at most 0.02 (p95), so B's information along this direction carries the update.
- **Feature difference along the same direction:** `x̂_B + P_U(x̂_A − x̂_B) + ε_B` flips only 0.11. Along this direction, the decisive difference is in the **error**, not the features.
- **DAS rank 1, contrast:** the same feature-only patch flips 0.99. B's answer is fragile, so ablating its feature component along the DAS direction also erases it. Both components are needed (see the conjunction above).

### Generality caveats
- On fresh templates the effect is weaker. E3 at L12, P(B | x̂_B + ε_A): ot_bin 0.36 (n = 117), ot_drawer 0.53 (n = 128), pilot 0.02.
- The factorial interaction for those families is −0.04 and +0.07.
- The extreme conjunction is partly a property of the pilot template.

**Possible paper claim:** in mid layers (L8–16) of Gemma-2-2b, the binding of a move event to the queried object is not recoverable from Gemma Scope features alone. Swapping only the SAE error erases it in 72–98% of pairs, while the same swap at L4 erases it in 9%. A one-dimensional direction estimated purely from SAE errors carries it. A second, methodological claim: feature-only patching on this task is confounded by a prefix fallback answer, which inflates apparent feature sufficiency.

---

## 2. Who carries the A-vs-B contrast: error share by task and depth (2×2 factorial, ratio of means with 95% bootstrap CIs)

`factorial.csv`. Error share = ½[(D_AA − D_AB) + (D_BA − D_BB)] / (D_AA − D_BB).

| task | L4 | L8 | L12 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|
| running sum, story, n = 155 | 0.02 | 0.04 | 0.08 | 0.08 | **0.18 [0.15, 0.20]** | 0.13 |
| IOI, n = 150 | 0.00 | 0.04 | 0.08 | 0.14 | 0.25 | **0.35 [0.34, 0.36]** |
| variable binding, n = 173 | 0.00 | 0.01 | 0.03 | 0.04 | 0.05 | 0.04 |
| object tracking, pilot | 0.02 | 0.18 | 0.44* | 0.15* | 0.12 | 0.09 |

\* Large negative interaction (section 1).

People-cities at L12, from E14, for five languages:

| language | error share |
|---|---|
| English | 0.25 |
| French | 0.24 |
| Hindi (Devanagari) | 0.24 |
| Hindi (romanized) | 0.24 |
| Hinglish | 0.22 |

The CIs are about ±0.01. Running sum at L18 is 0.13–0.15 in all four languages tested. Interaction is about 0 for every task except OT.

- The error's share of task-state contrast grows with depth for running sum and IOI, and stays near zero for variable binding.
- For people-cities, the share is the same in every language even though SAE text fidelity is lower for Hindi (E12). Treat this as descriptive: the Hindi rows fail Gate B.

---

## 3. The feature-only loss is a near-uniform shrink of each pair's logit difference, not a failing subset

`h1_margin.csv`. Per pair: r_i = (D(x̂_A + ε_B) − D_B) / (D_A − D_B). Values are r_i quantiles (q10 / median / q90).

| cell | quantiles | pairs with r_i < 0.5 |
|---|---|---|
| IOI L20 | 0.56 / 0.65 / 0.70 | 2% |
| people-cities L16 | 0.59 / 0.67 / 0.76 | 0.7% |
| OT L12 | 0.64 / 0.76 / 0.88 | 1% |
| running sum L18 | 0.60 / 0.86 / 1.01 | 5% |

Correlation with the clean margin is weak, Spearman −0.2 to 0.4.

- Decisions change only where margins are small. IOI at L20 loses 36% of D, yet only 1.3% of decisions flip, because its clean margin is about 13 logits.
- A one-scalar interpolation model, l' = l_B + a(l_A − l_B) with a cross-fitted, predicts the observed flip rates for OT, IOI and variable binding almost exactly (`interp_model.csv`).

**Possible paper claim:** the SAE error acts like a gain on the task's logit difference, uniform across pairs. Decision-level metrics hide large confidence losses, and NLDR hides nothing about which pairs fail.

---

## 4. Running sum: decision failures are value errors (near-misses), not partial reversion to B

For eligible running-sum pairs whose feature-only patch gives neither A nor B, the wrong answer is a near-miss (A ± 1, excluding B) in 58–79% of cases. A uniform choice among the 8 other digits would give 11–17% (`h5_ladder_*.csv`). E3 fresh, n = 737, feat_A_errB: 43 of 59 non-B wrong answers are near-misses (0.73; null 0.14).

The interpolation model explains the logit difference but not the decisions:

| cell | observed flip | interpolation prediction |
|---|---|---|
| E3 fresh, x̂_A + ε_B | 0.85 | 0.98 |
| E3 fresh, mean-error | 0.82 | 0.98 |
| E8 L18 | 0.85 | 0.985 |

The missing flips are "other" answers (8–13% observed vs 1% predicted). The running-sum shortfall behind RQ1 is mostly the model computing a **neighbouring number**, not drifting toward B.

**Not specific to B's error.** Near-misses dominate under every feature-only or noise patch: zero 0.64, mean 0.66, noise 0.58, covariance-noise 0.61. Features plus any error keep the approximate magnitude; the exact value needs the context-matched error. Under B's error, wrong digits lean toward B's side in 0.63 of cases (37 of 59), against 0.48–0.55 under the other patches. That lean is weak, around p ≈ 0.07.

---

## 5. Running sum: the error carries answer-specific information

`resample_same_vs_diff.csv`. The harness resamples errors from another test context matched on family, length and first-changed position. The donor's answer either equals A's (`sameAns`) or is neither A's nor B's (`diffAns`). The table gives sameAns − diffAns as paired differences.

| run | n | Δ NLDR (points) | Δ flip (points) |
|---|---|---|---|
| E3 fresh, 5 families | 737 | **+7.5 [5.7, 9.3]** | **+11.4 [7.1, 15.7]** |
| E2 pilot | 204 | +8.3 [5.1, 11.5] | +15.2 [7.4, 22.6] |
| E2r pilot (seed 1, new donors) | 203 | +5.6 [2.4, 8.9] | +5.4 [−1.5, 12.8] |
| E3 OT (contrast) | 369 | +2.3 [0.3, 4.4] | +2.4 [0.0, 4.9] |

Every one of the 5 running-sum families is positive (NLDR +6.1 to +9.1).

**Donor pull** (`mech_donor_pull2.py`, `donor_pull_controlled.csv`). I reconstructed the donor choices with the harness's own `select_donors`; the stored donor status matches in 100% of pairs.

| E3, P(top-1 = donor's answer) | value | difference |
|---|---|---|
| under diffAns | 0.088 | |
| under sameAns, same pairs and digit | 0.048 | +4.1 points [1.9, 6.1]; softmax mass +3.8 [2.7, 4.8] |
| under mean-error | 0.038 | |
| under B's error | 0.023 | |
| under Gaussian noise | 0.106 | no excess over noise |

The pull is real against the matched resample control but small, and does not exceed unstructured noise. Running-sum error mixes generic context and position structure (mean-error alone reaches 0.83 flip) with a small, replicated answer-value component.

---

## 6. Which running-sum pairs fail without their own error?

`h2_rs_pairs.csv`. Logistic regression, E3 fresh, n = 737, failure = top-1 ≠ A.

| predictor | under x̂_A + ε_B (OR, 95% CI) | under mean-error (OR) |
|---|---|---|
| log clean margin | **0.06 [0.03, 0.13]** | 0.13 |
| A > B (B's total smaller) | **3.3 [1.8, 5.8]** | 1.3, ns |
| \|A − B\| | 3.2 [1.9, 5.1] | 2.4 |
| corrupted step = last (vs first) | **2.2 [1.3, 3.8]** | 0.9, ns |
| A value, template family | ns or small | ns or small |

Two effects appear only when B's own error is present: "A > B" and "late corruption". This fits B's error carrying B's (smaller, recently changed) value. Clean confidence is the dominant predictor overall.

---

## 7. Depth: where the error starts to matter

`h6_layers.csv`.
- **Running sum:** the decision gap under x̂_A + ε_B is 0.02–0.04 at L4–16, then **0.14 [0.10, 0.20]** at L18. The paired L16 → L18 increase is +12.3 points [7.4, 17.2], n = 204. The error share in section 2 jumps at the same depth.
- **Object tracking:** B's move information needs the error in the L8–16 window (section 1).
- **People-cities:** the logit-difference loss grows steadily, NLDR 0.87 → 0.68 from L8 to L16. Per-pair shrink is uniform (r_i IQR about 0.1), with no fallback (P(A | x̂_B + noise) = 0).

---

## Nulls and limits
- Near-miss structure (section 4) is generic to degraded numeric patches, so it is not, by itself, evidence that the error carries the value.
- Donor pull does not exceed the noise conditions.
- OT same-answer vs different-answer is small.
- OT conclusions rest on 96 eligible stay pairs from one template. Fresh templates show the same direction, but weaker.
- One model, one SAE family, synthetic tasks.
- All p-values are uncorrected and the hypotheses were formed after seeing the data.
- I did not test whether the OT error-PCA direction aligns with dead or subthreshold SAE latents. That is a geometry question outside this analysis.

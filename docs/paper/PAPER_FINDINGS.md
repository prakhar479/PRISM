# PRISM — findings for the paper (frozen 2026-11-19)

Every number here comes from `analysis/team_analysis_2026-11-19.md` after the independent claims audit
(`analysis/team/audit.md`): 78 of 88 claim parts reproduced with new code, and the minor mismatches are corrected here.
Figures are in `docs/paper/figures/` (PDF, PNG and the exact plotted data as CSV). Generating scripts are in
`analysis/team/scripts/fig_*.py`.

Scope, to state in the paper: one model (gemma-2-2b), one SAE family (Gemma Scope JumpReLU, residual stream), and synthetic
tasks. **Preregistered** results are labelled as such; everything else is exploratory and post hoc.

## Working title and one-paragraph abstract
*What Sparse Autoencoders Lose When a Model Tracks State: A Causal Audit of Gemma Scope, and Five Pitfalls in Doing One.*

Sparse autoencoders (SAEs) are used to read model internals as features, and what they cannot reconstruct is left as error.
We audit, causally, how much task state Gemma Scope SAEs lose in gemma-2-2b, on running sums, object tracking and
person–city binding.

- **Small but real loss (preregistered).** For running sums at layer 18, patching only the clean run's SAE features (with a
  mean error) leaves a 14.7-point shortfall in answer flips (95% CI 11.3–18.0; n = 450).
- **The causal variable is low-dimensional and mostly made of features.** Distributed alignment search (DAS) finds it with
  at least 4 dimensions, and 7 decoder features recur across fits. SAE features carry 65–79% of its effect (median 75%).
  Directions estimated from the SAE error do not find it below rank 32.
- **Binding is distributed and high-rank.** Person–city state needs rank 16 for DAS, and error directions never capture it
  (0 flips up to rank 64).
- **Wider SAEs only partly help.** A 4× wider SAE moves part of the causal subspace into the features (75% → 83%) but does
  not close the task gap.
- **Five pitfalls that change conclusions:**
  - decision metrics hide logit-level loss;
  - default answers in tracking tasks inflate apparent feature sufficiency;
  - a single steering vector cannot meet common success criteria on ordinal variables;
  - DAS conclusions shift with seed by more than their bootstrap error;
  - cross-lingual fidelity metrics are confounded by predictability. At matched predictability and error size, Indic
    native-script text still costs about 1.5× more than English or French, but script, language and corpus cannot be
    separated.

## Contributions (what to present, in order)

### Headline results

**R1. Preregistered existence result (RQ1 GO, running sum L18).**
- **The gate.** The flip shortfall full_A − feat_A_meanPos on fresh template families is 0.147 [0.113, 0.180], n = 450.
  This is above the preregistered 10-point bar.
- **Qualifiers to report.** With B's actual error the shortfall is 0.098. It is 0.204 for low-margin pairs and 0.089 for
  high-margin pairs. The family-cluster CI rests on 3 clusters.
- **Replication.**
  - Bit-identical across three RTX 2080 Ti nodes (deterministic recomputation).
  - Within 0.7 points on an RTX 5060 laptop.
  - Seed-1 rerun (E2r).
- Figure F1 places this cell in the task × layer atlas.

**R2. The running-sum causal variable: low-dimensional, mostly features, invisible to error PCA (F2, F7).**
- **DAS validity.** Rank 1–2 is never a valid fit (train accuracy ≥ 0.5) across 16 variants (3 seeds; 3, 10 and 30 epochs;
  2 learning rates); the best is 0.473. All rank-4/8 fits are valid, up to 0.98.
- **A seed-stable rank-1 direction exists.** With the shared initial rotation removed, the median top cosine is 0.74 over
  16 pairs, against a random 99th percentile of 0.05. It flips 9–35% of pairs against a random p95 of 0. So the variable is
  there, but not in 1–2 dimensions.
- **Feature-made.**
  - Seven 16k decoder atoms (15376, 7686, 8352, 11219, 13950, 411, 14691) are in the top 10 of every rank-8 fit,
    including seed 1. They make good Neuronpedia lookups for the paper.
  - Feature capture of the causal subspace, NLDR(feat_A_along_U) / NLDR(full_along_U), is median 0.754 (0.650–0.794,
    7 fits).
- **Missed by error PCA.** Error-PCA directions become effective only at r ≥ 32 (flip 0.15–0.27) and are never
  feature-mediated up to r = 64. Repair along DAS-4/8 closes 0.90–1.06 of the residual logit gap, against 0.28 for error
  PCA at r = 4.

**R3. Cross-task picture: how much of the causal subspace SAE features carry (F7, F5).**
- **Capture by task (16k).** Running sum 0.75; object tracking 0.57 (0.52–0.66, 5 fits); people-cities 0.69 (L12) and
  0.66 (L16), one fit each.
- **People-cities is high-rank and answer-selective.**
  - Error-derived U gives 0 flips up to r = 64.
  - DAS is valid only at rank 16 (train accuracy 0.81 / 0.78; full 0.79 / 0.83, features 0.42 / 0.27).
  - A's city explains 57% (L12) and 71% (L16) of per-pair DAS variance.

**R4. Width (F4).** Read this from the paired same-rotation numbers, not from the capture medians: those rest on 2 fits at
65k/262k and 1 for people-cities.
- **Capture rises.** With the identical DAS rotation, feature capture rises with SAE width at matched L0:
  - running sum 0.754 → 0.827 at 65k (paired Δ +0.073 [0.047, 0.098]);
  - object tracking 0.532 → 0.574 at 65k and 0.591 at 262k.
- **The gap does not close.**
  - Running sum: the flip gap is unchanged (+0.020 [−0.035, 0.074], an informative null); the NLDR gap shrinks by only
    0.024.
  - Object tracking: the gap falls only at 262k (−0.105).

### Methodological contributions (the pitfalls)

**P1. Decision metrics hide logit-level loss (F1).**
- IOI L20 loses 36% of the logit difference but flips 1.3% of decisions; object tracking L12 loses 26% and flips 0%.
- Audits should report both.

**P2. Default answers inflate feature sufficiency (object tracking; F3d).**
- **The design problem.** In the pilot design A's answer is already fixed by the unpatched prefix, so anything that erases
  B's update "restores" A. The evidence:
  - at steering scale s ≤ 1, 99–100% of random-direction outcomes that aren't B land on A's box;
  - noise in place of B's error gives A in 86–93% of pairs.
- **What steering in object tracking shows.** "The object moved", not where it went: DAS flips 0.98–1.00 at s = 0.25
  against a random p95 of 0.11–0.15.

**P3. Single-vector steering on an ordinal variable (F3a–c).**
- **One-sided flips.** Running-sum steering moves every answer the same way: DAS-4 always up, PCA directions down. DAS-4
  at s = 4 flips 0.32 of A > B pairs and 0.00 of A < B pairs (n = 106/97, Fisher p = 1e-11). Fresh templates (E10b)
  replicate this.
- **A ceiling by construction.** One vector can flip at most max_k P(A − B = k) = 0.353, so the preregistered 50% steering
  criterion (Q2) cannot be met. Within that ceiling steering is specific: 0.18 vs random p95 0.007.

**P4. DAS seed dependence and shared initialisation.**
- **Seed effect.** Seed 1 vs seed 0 moves DAS-4 full_along_U from 0.760 to 0.593 and feat_A_along_U from 0.547 to 0.307
  (q < 0.001). That is more than twice the bootstrap CI half-width of about 0.075.
- **Shared start.** Fits with the same (k, seed) share their initial rotation, and 3-epoch fits keep 41–76% of their
  energy in it. Raw DAS-to-DAS cosines (e.g. 0.85 across tasks) drop to 0.08 once the start is projected out.
- **The validity gate.** It uses in-sample accuracy and is nearly a threshold on the outcome itself (Spearman 0.96).

**P5. Cross-lingual SAE fidelity (F6).**
- **The metric.** CE-loss-recovered with mean ablation is not comparable across languages: the ablation baseline spans
  2.4–9.7 nats.
- **The ratio, step by step.** Native-script Indic text costs 2.8× English per token raw (median of cells), 2.95× matched on
  predictability, and 1.47× matched on predictability and relative error. All 21 supported cells are above 1; the CI
  excludes 1 in 19.
- **Not identified: script vs language vs corpus.**
  - Placebos between Latin-script sets range from 0.64 to 2.28.
  - With language fixed effects the ratio is 1.14–1.92 (significant in 5 of 16 specifications).
  - Within Hindi at matched predictability, native costs 1.34× [1.09, 1.71] only at L12 16k, about the same at L12 65k,
    and *less* at L6/L18/L24.
- **Wider SAEs.** At L12, 65k removes less native error than English error (cost −15 to −21% vs −27%).
- **Task-state loss is language-invariant.** Running sum 0.12–0.15, people-cities 0.22–0.24. Caveat: the task-critical
  tokens are script-neutral.

### Supporting or appendix material
- **Preregistered answers of record.**
  - E9: error-U not effective; DAS-4/8 "illusion" and "feature-mediated", flags that depend on the seed (P4).
  - E10: Q2 not met (P3 explains why).
  - M1: supported at 16k and 65k (44/44).
  - M2: testable only for French people-cities, not supported.
  - M3: not evaluable.
- **Multilingual DAS transfer (E22).** The English-fit running-sum DAS works in French, Hindi and Hinglish (0.50–0.63 vs
  0.54–0.61 at home), equivalent only within about ±15 points. People-cities transfer fails.
- **Running-sum error carries graded magnitude.** Same-answer resampled errors beat different-answer ones by +0.16 to +0.21
  at donor distance ±3/±4, and by about 0 at ±1. It replicates in 3 runs.
- **Per-pair response matrices are low-rank** (first mode 39–72%). Running-sum pairs starting at 0 are hardest to patch
  (replicates on E22 pairs).
- **Error directions** are task-specific, stable across templates and languages for running sum (cosine 0.86–0.92), off the
  16k dictionary, and persist into wider SAEs.

## Claims NOT to make
- "Script, not error size, sets the cost of SAE error": retracted (predictability confound).
- "Fertility predicts fidelity."
- "The SAE error is a uniform gain on logit differences": holds only for object tracking and IOI; for running sum the loss
  is additive.
- "A single error direction carries the binding update" in object tracking: confounded by the fallback.
- "The running-sum error stores the exact answer."
- "Steering fails because the subspace is wrong" (see P3).
- Any effect computed from `ans_A_id − ans_B_id`: running-sum digit token ids are not contiguous.

## Figures
| fig | content | supports |
|---|---|---|
| F1 | task × layer atlas: NLDR gap vs flip gap of feat_A_errB (E8) | R1, P1 |
| F2 | running sum: flip vs rank for DAS (all E17 variants; invalid fits hollow) and error-U to r = 64 | R2, P4 |
| F3 | steering dose-response, running sum (all / A > B / A < B, with one-vector ceilings) and object tracking, with random envelopes | P2, P3 |
| F4 | width: capture of the fixed DAS subspace and the feat_A_errB gap, 16k / 65k / 262k | R4 |
| F5 | people-cities: DAS validity and flips vs rank; per-city DAS selectivity | R3 |
| F6 | multilingual: native/English cost ratio raw / matched / matched + error; width change per set | P5 |
| F7 | feature capture of the causal subspace per valid DAS fit, by task and width | R2, R3 |

## Limitations to state
- One model and one SAE family.
- Synthetic tasks.
- The object-tracking template has a default-answer design.
- 3 template families for the RQ1 cluster CI.
- DAS with 3 seeds, validity judged in-sample; people-cities DAS from one seed.
- One corpus per language.
- E23 (norm-matched noise controls) was not run.
- No per-token losses were saved.
- All post hoc analyses are uncorrected unless BH is stated.

## Experiments that would most strengthen the paper (all need a GPU)
1. **E23 noise controls with per-token NLLs.** Already implemented in tag run-20261118-2. They separate error direction from
   context fragility.
2. **Object tracking without a default answer,** plus the reverse sufficiency patch x_A + P_U(x_B − x_A).
3. **Multi-seed DAS with held-out validity,** including people-cities.
4. **A predictable Latin-script low-resource language,** to separate script from language.

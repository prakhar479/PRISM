# Team analysis of all results, including the final exploratory run (2026-11-19)

> **Audit (2026-11-19, `analysis/team/audit.md`):** an independent recomputation with new code checked 88 claim parts.
> - **Results:** 78 match, 5 are minor mismatches, 2 cannot be rebuilt, and 3 were not audited. The text below carries the
>   corrections.
> - **Mismatches and not-rebuildable:**
>   - The laptop agrees within 0.7 points, not 0.5.
>   - The donor-order mismatch is 1.9–2.3%, not 1.9–2.8%.
>   - "39,560 shared values" depends on which runs are joined (the identity itself holds).
>   - node-B is recorded in DECISIONS, but the E3/E4/E5/E8 manifests carry the DEV host that finalized them.
>   - The matched multilingual ratio is not larger at L12.

Eight analysts worked independently from the same brief (`analysis/team/BRIEF.md`). Their roles: standard methods,
statistics, pattern mining, DAS mechanism, multilingual and width, steering, geometry, and red team. Their reports, scripts
and small outputs are in `analysis/team/`.

This document is the lead synthesis. Everything in it is **exploratory and post hoc** except where marked *preregistered*.
It supersedes the claims in `analysis/deep_analysis_2026-11-18.md` wherever the two differ; the "Corrections" section
lists those differences.

## Provenance of the final run
- **What ran.** The final exploratory queue ran on node node-D, at commit 240e5f6. That commit is tag run-20261118-1 plus a
  server-side `#SBATCH` node edit, so the runs are untagged and were ingested with `--allow-untagged`.
- **Not run: E23**, the norm-matched noise controls. It was added in run-20261118-2, but the server ran the older tag.
- **What came back.** 5 experiment runs (E17, E18, E19, E20, E22) and 9 E21 fidelity runs, all complete. Hook checks are 0.
- **Reproducibility.**
  - **Same GPU model:** results are bit-identical across three RTX 2080 Ti nodes (node-B, node-C, node-D). This
    covers DAS rotations, all shared per-pair logits (max |Δ| = 0) and all shared table rows. node-B is the release-2 node per
    DECISIONS; the E3/E4/E5/E8 manifests record the DEV host that finalized them. The node-D logs show stage 1
    recomputed and DAS retrained, so this is deterministic recomputation, not cache reuse.
  - **Different GPU architecture:** runs on the RTX 5060 laptop agree within about 0.7 points.
  - **Statistics:** an independent recomputation of 96 table rows reproduced every point estimate and CI exactly.

## Preregistered answers (unchanged)
- **RQ1:** GO for running sum (RS) L18; the gate reproduced exactly (0.147 [0.113, 0.180], n = 450).
- **E9:** error-derived U is not effective for RS L18. DAS-4 and DAS-8 are effective, with both the "illusion" and
  "feature-mediated" flags set.
- **E10 Q2:** not met.
- **M1:** supported for every non-English set (now at 16k and 65k: 44/44).
- **M2:** testable only for French people-cities, and not supported there.
- **M3:** not evaluable.

The red team's qualifications, which are post hoc:
- With B's actual error the RS shortfall is 0.098, just under the 10-point bar.
- The shortfall is 0.204 for low-margin pairs and 0.089 for high-margin pairs.
- The family-cluster CI rests on 3 clusters.
- The E9 labels depend on the DAS seed (see C2).

## What survives (ranked by strength)

### C1. Running sum: the causal variable needs at least 4 dimensions for DAS, is mostly built from SAE features, and is missed by error directions (strong)
- **DAS rank 1–2 never reaches a valid fit (E17).** Across 16 variants (3 seeds; 3, 10 and 30 epochs; two learning rates),
  the best train accuracy is 0.473. Ranks 4–8 are valid in all 6 fits, reaching 0.98 for das8_e10.
- **A rank-1 direction still exists.** After removing the shared initial rotation, the rank-1 learned direction is
  seed-stable: median top cosine 0.74 (range 0.60–0.86 over 16 non-independent pairs from 7 fits), against a random-subspace
  99th percentile of 0.05. The 7 rank-1 fits flip 9–35% of pairs (27–35% for seed 0) against a random p95 of 0. So the variable is not absent; rank 1 simply does not capture enough of it to
  pass the 0.5 bar. The gate is effectively a threshold on the outcome: train accuracy vs test flip, Spearman 0.96.
- **Feature-made, with a small error slice.**
  - Seven decoder atoms (15376, 7686, 8352, 11219, 13950, 411, 14691) are among the top 10 of the 16k dictionary in every
    rank-8 fit, including seed 1, and in the top 11 of the rank-16 fit.
  - 20 atoms explain 0.29–0.49 of each rank-4/8 learned subspace (0.19 at rank 16), against a random 99th percentile of 0.03–0.12.
  - A separate slice of 0.15–0.31 lies in the 32-dimensional error subspace (random 0.014).
- **Error-derived U becomes effective only at r ≥ 32 (E18):** flips 0.15–0.27 under f_pilot (0.28 under f_none), never feature-mediated even at r = 64.
- **Repair along DAS-4/DAS-8 closes 0.90–1.06 of the residual NLDR gap**, against 0.28 for the error PCA at r = 4.

### C2. DAS effect sizes carry seed variance larger than their bootstrap CIs (strong; methodological)
- Seed 1 vs seed 0: das4 full_along_U 0.760 → 0.593 and feat_A_along_U 0.547 → 0.307 (BH q < 0.001).
- Fits with the same (k, seed) share their initial rotation, and 3-epoch fits keep 41–76% of their energy in it.
  Validity is judged on in-sample training accuracy.
- So DAS conclusions need several seeds. Fixed-seed bootstrap CIs (half-width about 0.07) understate the uncertainty.
- The E9 "illusion" flag for RS turns false at DAS-16, and its seed-robustness cannot be shown at the 10-point bar.

### C3. A single fixed steering vector cannot meet Q2 on an ordinal variable (strong; methodological)
- Running-sum DAS steering moves every pair's answer the same way along the number line:
  - das4 moves answers up in 100% of trajectories and never below B;
  - pca r1/r4 move answers down in 86–96%.
- It therefore flips only pairs on one side: das4 at s = 4 flips 0.32 of A > B pairs and 0.00 of A < B pairs
  (both-correct pairs, n = 106/97, Fisher p = 1e-11; under f_pilot 0.31 vs 0, p = 4e-8). E10b replicates this on fresh templates.
- One added vector can flip at most max_k P(A − B = k) = 0.353 of pairs, so the 50% Q2 bar is unreachable by construction.
- Within that ceiling, steering is specific but weak: 0.18 flips against a random p95 of 0.007 (q = 0.005).

### C4. Object tracking: the prefix-fallback confound and its consequences (strong; methodological)
- **The confound.** A's answer is determined by the unpatched prefix, so any manipulation that erases B's update produces A.
  The evidence:
  - at s ≤ 1, 99–100% of random-steered non-B outcomes land on A's box (chance would be 1/6);
  - how often a pair flips under random steering tracks its clean-B margin (Spearman −0.85);
  - noise in place of ε_B already gives A in 86–93% of pairs on the pilot templates (about 50% on fresh templates).
- **Steering in OT shows "the object moved", not where it went.** DAS steering at s = 0.25 flips 0.98–1.00 against a random
  p95 of 0.11–0.15, and error-U flips 0.58–0.67.
- **"One error direction carries the binding update" is weakened.** The features alone also erase B along DAS-1
  (0.989), and the negative 2×2 interaction is exactly what a fallback predicts. What would settle it is the reverse
  sufficiency patch x_A + P_U(x_B − x_A), on a task design without a fallback.

### C5. People-cities: the task state is high-rank and distributed, and DAS is answer-selective (moderate-strong)
- Error-derived U flips 0/150 at every rank up to 64, at both L12 and L16.
- DAS is valid only at rank 16 (train accuracy 0.81 / 0.78), with full flips 0.79 / 0.83 and feature flips 0.42 / 0.27.
  These are the largest full − feature gaps of any DAS fit (0.37 / 0.55; OT error-U r1/r2 gaps are larger, 0.80/0.85).
- Which city A is explains 57% (L12) and 71% (L16) of per-pair DAS variance; das1 acts as an "Athens" direction.
- Steering pushes answers toward a fixed city ("Tokyo" in 93% of landings at das8, s = 16).
- **Caveat:** there are no PC seed replicates, and all PC fits at rank 8 or below are invalid.

### C6. Width moves part of the causal subspace into the features without shrinking the task gap (moderate-strong)
- Feature capture of the same DAS subspace rises with width:
  - RS das4: 0.754 → 0.827 at 65k;
  - OT das4: 0.532 → 0.574 at 65k and 0.591 at 262k (paired CIs exclude 0).
- The total feat_A_errB flip gap does not change at 65k (RS +0.020 [−0.035, 0.074], an informative null); the NLDR gap shrinks slightly (−0.024 [−0.046, −0.002] under f_A_correct). It falls only for OT
  at 262k (−0.105).
- Error-U repair changes in opposite directions for the two tasks: RS pca r1 0.225 → −0.033, OT pca r8 0.547 → 0.734.

### C7. Multilingual text fidelity: Indic native-script text costs about 1.5× more than English/French at matched predictability and error size (moderate)
- **Raw gap.** Native/English splice-cost ratio of about 2.7× (geometric mean).
- **Matched on clean NLL:** 2.95× (24/24 cells > 1).
- **Matched on clean NLL plus relative error |ε|/|x|:** 1.47× (1.23–1.75; all 21 supported cells > 1, CI above 1 in 19), and 1.56× against pooled
  Latin sets. Regression with set clusters gives 1.53 (1.41 with language fixed effects, wild p 0.12, 12 clusters).
- **It holds at 16k and 65k.** It reverses at L6 (only the two Hindi native sets exist at L6/L24).
- **Not identified: script vs language vs corpus.**
  - Placebos between Latin sets already range from 0.64 to 2.28.
  - With language fixed effects the ratio is 1.14–1.92, and significant in only 5 of 16 specifications.
  - Dropping English and French removes it (1.05 on common support).
- **Width helps native text less.** At L12 cost falls 15–21% for native sets vs 27% for English (at L18 English's −22% lies inside
  the native range), because width removes less native FVU (all 12 CIs exclude 0).
- **Task-state loss is language-invariant** (RS 0.116–0.148, PC 0.221–0.244). Caveat: the task-critical tokens (digits,
  Latin city names) are script-neutral.
- **The English-fit RS DAS transfers** to French, Hindi and Hinglish: flips 0.50–0.63 vs 0.54–0.61 at home. Equivalence
  holds only to within about ±15 points (underpowered). PC transfer fails (−0.23 to −0.27).

### C8. Decision metrics hide logit-level loss (moderate)
- The names task (IOI) at L20 loses 36% of the logit difference but flips only 1.3% of decisions; OT at L12 loses 26% and
  flips none.
- The earlier "uniform gain" model is weakened: in RS the per-pair loss is additive (slope −0.005 [−0.067, 0.054]), and a
  gain shape appears in only 7 of 36 E8 cells.

### Other robust patterns (moderate)
- **Hard RS pairs.** RS pairs starting at 0 are the least patchable by any subspace (DAS4 NLDR 0.65 vs 0.77–1.00 for starts 1–4). This
  replicates on independent E22 pairs (15/16 cells positive).
- **Per-pair responses are low-rank.** The first mode explains 39–72% of variance and is stable across halves; it is
  unrelated to per-pair error cost.
- **RS error carries graded magnitude.** Same-answer resampled errors beat different-answer ones by +0.16 to +0.21 when the
  donor's answer is ±3/±4 from A's, but by about 0 at ±1. So the error carries graded magnitude, not the exact answer.

## Corrections to earlier readings
| earlier claim | status |
|---|---|
| "Script, not error size, sets the cost of SAE error" (deep analysis A; 1.5–4.1× on identical sentences) | **Retracted as stated.** Native and romanized windows do not overlap in predictability. The only identified within-language contrast (Hindi, 4% tail) gives 1.34 [1.09, 1.71] at L12 16k (1.21 [0.97, 1.55] after also adjusting for |ε|/|x|), about 1.0 at L12 65k, and significantly below 1 at L6, L18 and L24 (native is cheaper there). What survives is C7. |
| "Fertility predicts fidelity" | Retracted; neither fertility nor script is identified separately from language/corpus |
| "The error acts as a uniform gain" | Weakened to OT and IOI; RS loss is additive (C8) |
| "A single error direction carries the OT binding update" | Weakened (C4) |
| "RS error carries the exact answer" | Graded magnitude only |
| Donor-pull analysis (deep/mech_donor_pull*.py) | Bug: donors were rebuilt in the wrong order (1.9–2.3% match). Recomputed, the conclusion is unchanged |
| "Gap replicates across 3 machines" | Bit-exact across 3 nodes of the same GPU model; independent hardware within about 0.7 points |

## What a paper can claim, and what would still be needed
**Strongest framing:** *auditing what sparse autoencoders lose for task state, and the pitfalls of doing so.*

Main claims (one model, one SAE family, synthetic tasks):
1. A small, robust, preregistered task-state shortfall in arithmetic. Its causal variable is low-dimensional (DAS needs
   ≥ 4 dimensions), feature-made with an error slice, and invisible to error-PCA below rank 32 (C1).
2. Binding task state is distributed and high-rank (people-cities, C5), and error directions do not capture it.
3. Wider SAEs absorb part of the causal subspace without closing the gap (C6).
4. Five audit pitfalls, each demonstrated on these data:
   - decision vs logit metrics (C8);
   - prefix-fallback defaults in tracking tasks (C4);
   - single-vector steering criteria on ordinal variables (C3);
   - DAS initialisation, seed dependence and in-sample validity (C2);
   - cross-lingual fidelity metrics confounded by predictability and the ablation baseline (C7, and the mean-ablation note
     in deep analysis A).

**Secondary:** multilingual text fidelity (C7) at the strength stated there.

**Settling experiments, in priority order** (all need a GPU):
1. E23 noise controls (in run-20261118-2) plus per-token NLLs. This tells error direction from fragility, and allows
   token-level matching.
2. An OT design without a fallback, plus the reverse sufficiency patch.
3. DAS with several seeds and held-out validity, and PC seed replicates.
4. A predictable Latin-script low-resource language, to separate script from language.

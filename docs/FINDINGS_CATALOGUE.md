# Findings catalogue (living document; last update 2026-11-26)

One row per finding, with the evidence behind it, how far it can be trusted and what it is good for. **Add to this file after every
run; never delete a row, change its status.** Status: **S** = solid (preregistered, replicated, or large effect with its controls),
**Q** = qualified (real but confounded or single-run), **R** = rejected / closed, **O** = open (not run, or run without a decision).

Sources: PRISM = the frozen first study (docs/paper/PAPER_FINDINGS.md on branch `harness`, gemma-2-2b, Gemma Scope JumpReLU residual
SAEs, 3 synthetic tasks). BP = breadth pilot (P0-P11, results/pilot-run-20261124-pilot-b). G3 = second pilot (P12, P13, results/pilot-g3-run-
20261125-g3e). Per-pilot tables: analysis/pilot/*.csv, verdicts and caveats: analysis/pilot/VERDICTS.md, NOTES.md.
Notation: *capture* = NLDR(features-only write-back along U) / NLDR(full write-back along U), U = DAS subspace (rank 8, kl objective);
*gap* = NLDR(full_A) - NLDR(feat_A_errB), the task state lost when only the dictionary's features are copied from the clean run.

## A. What we know (S / Q)

| id | finding | evidence | status | caveat | use |
|---|---|---|---|---|---|
| S1 | **Task state is mostly carried by SAE features, not by the error.** Capture of the causal subspace is 0.55-0.96 across 3 tasks, 2 models, many layers. | PRISM R2-R3 (RS 0.75, OT 0.57, PC 0.66-0.69); BP P1 (PC, Gemma 2: 0.65-0.90), P11 (OT-v2: 0.74-0.78); BP P8 (Gemma 3: PC 0.64-0.96, OT 0.63-0.91) | S | a quarter to a half of the subspace's effect is *not* in the features: that is the "dark matter" with a task | thesis of any audit paper |
| S2 | **The missing part is small in decisions, large in logits, and task-dependent.** RS L18: flip shortfall 0.147 [0.113, 0.180] (preregistered, n=450). OT-v2: gap 0.013-0.079 (Gemma 2), 0.024-0.117 (Gemma 3). PC: 0.29-0.33 at Gemma 3 L13/L17, ~0 at L22. | PRISM R1; BP P11, P8 | S (RS), Q (others: one SAE each) | decision metrics hide logit loss (PRISM P1): report both | the size of the problem |
| S3 | **Wider dictionaries raise capture but do not close the gap.** 16k -> 65k: +0.06 to +0.10 capture; RS flip gap unchanged (+0.020 [-0.035, 0.074]). | PRISM R4; BP P1 (PC, 3 layers) | S | one SAE family | width is not the fix |
| S4 | **Error directions chosen by variance (PCA) do not find the variable; directions chosen by output sensitivity do.** Gradient-chosen directions repair: PC L12 r=8 closes 0.95 vs 0.01 for error PCA; OT L12 r=2 0.75 vs 0.02; RS L18 r=4 0.59 vs 0.28 (they tie at r=8). | PRISM R2 (error PCA needs r>=32); BP P7 | Q | P7 ran on an untagged commit (c566c86); the share-of-gap on OT is inflated (feature-only already 0.966, repair overshoots to 1.05: steering, not restoring); the gradient is task-specific (an oracle) | **core mechanism candidate** (D) |
| S5 | **SAE error is less damaging than random noise of the same norm.** splice cost / noise cost = 0.56-0.71 for English, Hindi (2 scripts), French at L12, L18. | BP E23 (4 corpora x 2 layers, 246k tokens each, CIs +-0.01) | S | gemma-2-2b 16k only; CE on generic text, not tasks | the error is *anti-aligned* with sensitive directions; a calibrated fidelity metric |
| S6 | **The native-script "excess cost" is generic fragility, not error direction.** Norm-matched noise costs 2.2x as much on Hindi (Devanagari) as English (L12 0.40 vs 0.18 nats); splice/noise is 0.71 vs 0.64. | BP E23 | S | corrects the PRISM P5 retraction chain | closes the multilingual line |
| S7 | **Dictionary family matters at a fixed site; crosscoders are worst; capture tracks reconstruction error.** Gemma 3 1B L13, people-cities, 35 cells: Spearman(capture, relative error) = -0.80; crosscoders 0.19 (below the error trend by 0.26); seed floor 0.024. | G3 P12 (43 PC cells) | Q | one task (RS skipped, OT = P14 pending); composite L0 is summed over sublayers; CE/FVU per composite not measured; tc_single replaces one MLP of 14 | benchmark across families |
| S8 | **Replacement-model error nodes carry task state.** Largest feature-only gap 0.30 (CLT + attention SAEs) with error-only NLDR 0.28 at L13; ~0.04 at L17. | G3 P12 | Q | n=1 per family at that site; one task | circuit-tracing relevance |
| S9 | **Refusal in Gemma 3 1B-IT is mostly in SAE features.** Difference-of-means direction captured 0.82-1.00 by residual SAEs at L7-22 (crosscoder 0.59-0.68); error-only NLDR 0.51 at L7 16k but ~0 at L22. | G3 P13 (11 cells, n=150) | Q | one prompt template family; answer = first token | safety angle |
| S10 | **The DAS-based capture number is only as good as its protocol.** Capture moves 0.06-0.13 across objective x seed; code-level and subspace write-back differ by up to 0.4 (PC up to 1.8); PC rank-4 held-out IIA >= 0.5 in 0% of fits; seed 1 vs 0 moved DAS-4 feature capture 0.55 -> 0.31 (PRISM P4). | BP P0; PRISM P4 | S | | differences below ~0.15 are noise: every table needs kl objective, held-out validity, >=3 seeds |
| S11 | **Running-sum causal variable is low-dimensional (rank 4-8), seed-stable, and made of ~7 recurring decoder atoms.** | PRISM R2 (16 DAS variants) | S | one model | interpretability hook |
| S12 | **Object tracking needs a no-default design.** OT-v2 keeps 74% of pairs; DAS rank-8 full-patch flip 0.23-0.41 vs random p95 ~0; the v1 design's "sufficiency" was inflated by the default answer. | PRISM P2; BP P11 | S | | OT usable in any paper |
| S13 | **KL+MSE fine-tuning of a Gemma Scope SAE cuts held-out CE delta by ~27%** (L18 0.171 -> 0.126; L12 0.116 -> 0.085). | BP PD (trained dictionaries) | Q | task effect (P4) never completed | does fixing logits fix state? |
| S14 | **Gemma 3 1B capability map:** running sum 0/60 pairs pass the filter (answer correct 10-28%, chance 10%); variable binding 2%; people-cities 40%; object tracking 62-70%; IOI 100%. | DEV CPU probe 2026-11-25; BP P8; G3 P12 (43 skipped cells) | S | 60 pairs per family | task choice for Gemma 3 |
| S15 | Smaller PRISM results: running-sum error carries graded magnitude (same-answer resampled errors beat different-answer by 0.16-0.21 at donor distance 3-4); per-pair response matrices are low-rank; error directions are task-specific, stable across templates and languages (cos 0.86-0.92). | PRISM supporting | Q | exploratory | appendix |

## B. Closed directions (R): do not spend compute here

| id | claim we expected | what the data said | evidence |
|---|---|---|---|
| R1 | **Binding is a blind spot** (content kept, relations lost) | no: content, binding-swap and query-swap capture agree within +-0.04 (query is higher) at L8/12/16, 16k and 65k | BP P1 (18 cells) |
| R2 | **Magnitude goes dark** (the dense value axis lives in the error, the periodic digit code in the features) | no: the value line is captured as well as the period-10 plane (0.62-0.85 vs 0.53-0.77) and carries <= 0.13 of the resampled-error effect | BP P5 (7 cells; untagged commit, donor status backfilled) |
| R3 | Multilingual SAE failure is an error-direction effect | no: generic fragility (S6) | BP E23 |
| R4 | Cross-lingual CE-recovered is a comparable fidelity metric | no: the mean-ablation baseline spans 2.4-9.7 nats; script/language/corpus not separable | PRISM P5 |
| R5 | One steering vector can pass a 50% criterion on an ordinal variable | no: the one-vector ceiling is 0.353 by construction | PRISM P3 |
| R6 | Error PCA finds the missing variable | no below rank 32 (S4) | PRISM R2, BP P7 |
| R7 | A running-sum study on Gemma 3 1B | the model is at chance (S14) | G3 P12, BP P8 |
| R8 | "The error stores the exact answer"; "single error direction carries binding in OT-v1"; "script, not error size, sets cost" | retracted in the PRISM audit | PRISM claims-not-to-make |

Consequence for the decision matrix of docs/PILOT_BREADTH_PLAN.md: papers **A** (taxonomy: binding / magnitude lost) and **C**
(magnitude goes dark) lose their evidence; **D** (sensitivity, not variance) is the strongest mechanism; **B** (a causal benchmark of
dictionaries) and **E** (logit fixes do not fix state) wait for P2 / P4.

## C. Open questions (O): what could still change the plan

| id | question | status of the evidence | decided by |
|---|---|---|---|
| O1 | Does a **task-agnostic** sensitivity subspace (from generic text) contain the task-relevant error, so a cheap fix exists without task labels? | S4 used task gradients (an oracle) | batch 2, branch repair |
| O2 | Is the missing part a **decoder problem** (predictable from the codes by a nonlinear map) or an **information problem** (not recoverable from codes)? | nothing: Engels et al. show linear predictability from x, not from codes, and not causally | batch 2, branch repair |
| O3 | Does capture **dip where the variable is computed** (mid-depth) or fall monotonically? Gemma 3 L13 is the minimum for PC and OT; Gemma 2 PC/RS fall with depth | 4 layers per model | batch 2, branch atlas |
| O4 | Do **standard metrics** (CE recovered, FVU, L0, SAEBench) predict task-state capture across ~100 dictionaries? P12 says relative error does within one family | P2 (98 cells) never completed | P2 resubmit; batch 2 branch atlas |
| O5 | Does KL+MSE fine-tuning, a linear error add-on or dense low-rank add-backs restore **state** (P4)? | never completed | P4 resubmit |
| O6 | Do **attribution scores** pick the features DAS finds causal? | not run | batch 2 (stretch) |
| O7 | Do the findings hold on a **second model family** (Llama 3.2 1B) or larger Gemma (3 4B)? | Gemma 2 2B and Gemma 3 1B agree qualitatively | later |
| O8 | P5 / P7 / PD ran on commit c566c86485, which is not in the repository | provenance gap | ask what it changed |

## D. Where the data are

- Published runs: `results_server/pilot-breadth-server` (branch results/pilot-run-20261124-pilot-b: P0, P1, P5, P7, P8, P11, PD, E23 as `E12`),
  `results_server/pilot-g3-server` (branch results/pilot-g3-run-20261125-g3e: P12, P13). Not yet: P2, P4 (breadth), P14 (g3).
- Derived tables: `analysis/pilot/*.csv`; verdicts: `analysis/pilot/VERDICTS.md`; how to read them: `analysis/pilot/NOTES.md`.
- Integrity check: `scripts/verify_pilot_runs.py --queue <queue> --runs <runs>` (provenance, config equality, cell coverage, nulls).
- Protocol rules every new run must follow (from S10): DAS v2 with the kl objective, held-out split, per-cell init seed, >=3 seeds,
  rank- and norm-matched random nulls, a stated write-back operator, and a frozen-random-decoder control where a dictionary is compared.

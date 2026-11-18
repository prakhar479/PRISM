# Deep analysis of all results to date (2026-11-18): what is strong enough to publish

Status: **exploratory, post hoc** (hypotheses formed after seeing the data; no multiplicity correction unless stated). The
answers of record under the frozen preregistration are in `analysis/final_pass_results.md` and are not changed here.
Detailed reports, scripts and intermediate CSVs: `analysis/deep/` (`multilingual.md`, `mechanism.md`, `geometry.md`,
`scripts/*.py`). "Verified" = recomputed independently from raw per-window / per-pair data, not only by the script that found it.

## The thesis that ties the results together

**How large an SAE's reconstruction error is (FVU, |eps|/|x|) says little about what the error costs the model. The cost
depends on where the error points: in which script, at which depth, and along which task-specific, off-dictionary directions.**
Five independent lines of evidence:

| # | evidence | key numbers | strength |
|---|---|---|---|
| 1 | Same sentences, same error size, different cost (Dakshina native vs romanized script, E12) | FVU ratio native/romanized 0.85-1.06; splice cost ratio 2.44 [2.37, 2.51] hi L12, 4.13 [3.98, 4.28] ta L12, 1.93 bn L12; 1.51-2.29 at L18 (all 6 CIs exclude 1) | strong, verified |
| 2 | Task-state loss is not predicted by error size on the task prompts (E8 atlas x pilot exp1) | within-layer rank correlation of relative error with NLDR gap r = -0.06 (n = 16, L8-20); OT gap 26 % at L12 vs 3.5 % at L14 with similar relative error | moderate, verified |
| 3 | Task-state loss is language-invariant although text-level splice cost is 2.5x higher for Devanagari (E14 vs E12) | people-cities loss 22-24 % in all five languages (CI +-1); running sum 12-15 % | moderate (rows fail Gate B: descriptive) |
| 4 | Task-relevant error directions are off-dictionary and persist in 32x wider SAEs (geometry) | nearest 16k atom \|cos\| 0.10-0.25 vs random p99 0.10; 16k OT error direction stays in 32k-524k errors (rank-1 cos 0.57-0.76) | moderate (65k+ decoders not inspected) |
| 5 | How concentrated the error variance is does not predict whether low-rank repair works | rho = 0.15 (n = 36); people-cities error as concentrated as OT's, repair closes ~0 % | moderate |

## Findings ranked for a paper

### A. Script, not error size, sets the cost of SAE error (multilingual; strongest new result)
- Natural experiment: Dakshina gives the same sentence pool in native script and in romanization. Gemma Scope 16k reconstructs
  both equally well (FVU ratio 0.85-1.06, |eps|/|x| ratio 0.97-1.10), yet splicing x_hat costs the model 1.5-4.1x more loss
  per token in native script (all language x layer cells, CIs exclude 1). Verified.
- Against English with matched predictability (median clean loss 2.05-2.36 vs 2.33 nats): Devanagari/Tamil/Bengali text costs
  1.9-2.2x English's loss per unit FVU; Latin-script high-entropy sets (Hinglish, romanized Hindi) about 1x.
- The reported fertility correlation (Spearman -0.93/-0.97, final_pass_results.md) is a script effect: within Latin-script
  sets tokens-per-word does not predict cost (rho -0.03); Devanagari web and romanized Hindi have equal fertility (1.96 / 1.95)
  and 2.5x different cost.
- Methodological: CE-loss-recovered with mean ablation is not comparable across languages (the mean-ablation baseline spans
  2.4-9.7 nats); two frozen M1 "supported" rows (hi_latn, ta_latn at L12) have splice cost equal to or below English's.
- Shift, not tail: 80-98 % of native-script windows exceed English's 90th-percentile cost.
- Open (E23, queued): is it the error's direction, or are native-script contexts fragile to any perturbation? Norm-matched
  noise (x_hat + eta, x + eta) decides it; per-token NLLs localise it (word-initial vs continuation pieces, byte-level tokens).
  E21 asks whether a 4x wider SAE removes it.

### B. The SAE error acts as a gain on task logit differences; decision metrics hide it (mechanism)
- Per pair, the feature-only patch keeps a near-constant fraction of the logit difference (median r_i 0.65-0.86; at most 5 % of
  pairs below 0.5). A one-parameter interpolation model reproduces the observed flip rates for OT, IOI, variable binding.
- Consequence: IOI at L20 loses 36 % of the logit difference but flips 1.3 % of decisions; OT at L12 loses 26 % and flips 0 %.
  Decision-level (flip) and logit-level (NLDR) audits of SAEs can disagree completely; RS L18 is the case where both move
  (flip gap 14 %, NLDR gap 14.5 %), because its clean margins are small.
- Replication: RS L18 and OT L12 gaps reproduce within about 1 point across 10+ runs on three machines and two seeds
  (`deep/replication.csv`).

### C. Patching audits of entity tracking are confounded by a prefix fallback (methodological)
- In the pilot object-tracking design the corruption adds a move of the queried object after the first changed token, so A's
  answer is already determined by the unpatched prefix. Any patch that erases B's update "restores" A: feature-only
  sufficiency is inflated. Verified on the prompt files.
- The informative test is the swap x_hat_B + eps_A: B's answer survives in only 2 % [0.5, 7] of pairs at L12 (9-28 % at
  L8-16; 91 % at L4). B's update needs both its features and its error (2x2 interaction -0.38 of the logit difference at L12).
- A single direction estimated only from SAE errors carries it: patching the full residual along it flips 0.89 (random <= 0.02)
  while the features' component along it flips 0.11.
- Caveat: on fresh templates the swap effect is weaker (B survives 36-53 %); partly a property of the pilot template.

### D. Running-sum error carries a small, replicated answer-specific component
- A resampled error from a context with the same answer beats one with a different answer by +7.5 NLDR points [5.7, 9.3]
  and +11.4 flip points (n = 737, all 5 template families positive), replicated in E2 (+8.3) and E2r (+5.6).
- Feature-only failures are near-misses (A +- 1 in 73 % vs 14 % by chance) but near-misses dominate under every degraded patch,
  so this alone does not show the error stores the exact value.
- The RS gap appears between L16 and L18 (+12.3 flip points [7.4, 17.2], same pairs); the share of DAS's learned subspace
  inside the error subspace rises one layer band earlier (0.02 at L12-14, 0.13-0.18 at L16, 0.18-0.30 at L18-20).

### E. Error directions: task-specific, template- and language-stable, off-dictionary
- Cross-task rank-1 cosine median 0.016 (8 % of 66 pairs above chance); RS pilot vs fresh templates 0.78-0.90; RS English vs
  French / Hindi / Hinglish 0.86-0.92 (as stable across languages as across templates). People-cities: French 0.57, Hindi 0.02.
- Rotate with depth (half-life ~1.5-2 layers).
- RS L18 DAS (learned component) is built from a few recurring SAE atoms (15376, 7686, 8352, 11219, 13950, 411, 14691;
  20 atoms explain 44-59 %): the causal running-sum variable is feature-made, the residual error direction is not.

## Corrections to earlier readings
- DAS geometry: every fit with the same (k, seed) starts from the same rotation and 3-epoch fits keep 41-73 % of their energy in
  it; raw DAS-to-DAS cosines (e.g. 0.85 across tasks) are artefacts (0.08 after projecting the start out). make_final now
  reports `das_init_capture` and learned-component cosines. E9's DAS fits are short; E17 trains longer.
- Object tracking's release-2 "features restore 74-100 %" reading is inflated by the prefix fallback (C).
- Flip-rate differences across languages in E14 are a headroom artefact (one fractional loss plus each language's margin over
  distractors predicts them: hi .59 vs .56 observed, en .94 vs .93).
- "Fertility predicts fidelity" (final_pass_results.md, exploratory note) should read "script predicts fidelity cost".

## Suggested paper (one model, one SAE family; state it)
Working title: *Not How Much, But Where: The Cost of Sparse Autoencoder Error in Gemma-2*.
1. Setup and replication (gap reproduces across 10+ runs, 3 machines).
2. Error size does not predict cost: script natural experiment (A) + task atlas (thesis rows 2, 3, 5).
3. What the error does to task state: gain on logit differences (B), mid-layer binding needs the error (C), answer-specific
   component in arithmetic (D).
4. Where the error lives: off-dictionary, task-specific, persists with width (E); DAS finds feature-made subspaces for RS.
5. Pitfalls for SAE audits: decision vs logit metrics (B), prefix fallback (C), mean-ablation CE-recovered across languages (A),
   shared DAS initialisation.
Preregistered results are reported as such (RQ1 GO for RS L18; E9/E10/M1-M3 answers of record) and kept separate from all of
the above. Pending evidence from the last run: E23 (direction vs size), E21 (width), E17 (DAS training), E18 (rank), E22.

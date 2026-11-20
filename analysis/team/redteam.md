# Red team / skeptic report: attacks on PRISM's headline claims (2026-11-19)

Role: hostile reviewer. Everything here is **exploratory and post hoc**, written after seeing all results. CIs are 95 %
percentile bootstraps over windows (E12) or pairs (task runs), 1000–2000 resamples, **uncorrected**. Roughly 250 intervals
were computed in total (14 ratio cells × 10 normalisations, 46 matched comparisons, 108 gain-model fits, plus about 40
resample and RQ1 intervals), so a single marginal interval means little. Every verdict below rests on large effects or on
whole patterns, not on one p-value.
Scripts are in `team/scripts/redteam_*.py` and outputs in `team/scripts/out/`. E23 (norm-matched noise controls) was **not run**,
and several open questions below hinge on it.

| # | claim | verdict | the decisive point |
|---|---|---|---|
| 1 | Script, not error size, sets the cost of SAE error (Dakshina natural experiment) | **weakened** | The native and romanized sets do not overlap in predictability. Extrapolating the within-set slopes would explain 46–149 % of the native–romanized gap. Native Indic text vs English at matched clean CE survives (2.3–3.7x), but that compares languages, and FVU is 1.3–1.7x English's there. |
| 2 | The error acts as a uniform gain on logit differences; decision metrics hide it | **weakened** (gain part); the decision-metric point survives | In the confirmatory RS L18 cell the per-pair loss does not scale with the A–B swing (slope −0.005 [−0.067, 0.054], intercept 0.84 [0.45, 1.24] logits): the loss is additive, not a gain. A positive slope (CI > 0) appears in 7 of 36 cells. |
| 3a | OT patching is confounded by a prefix fallback | **survives** (it is a design fact) | It cuts against the project's own OT claims; see 3b. |
| 3b | A single error direction carries the OT binding update | **weakened** | Along DAS-1 the *features* alone flip 0.989. B's state is destroyed by noise (P(A \| x̂_B + η) 0.89–0.92). A negative 2×2 interaction is what a fallback predicts. Against a matched-norm random push the direction is specific (0.64 vs random p95 0.11 at ×0.25). |
| 4 | RS error carries an answer-specific component | **survives the lexical attack, but must be reworded as "graded magnitude"** | Same-answer donors beat different-answer donors only when the donor's answer is ≥ 3 away (+0.16 to +0.21 NLDR). At ±1 the difference is +0.02 [−0.006, 0.050] (E3), −0.005 (E2), −0.010 (E2r). |
| 5 | RQ1 GO for RS L18; E9 readings | **GO survives as preregistered** (reproduced exactly); the E9 labels are weakened | With feat_A_errB the gate-cell shortfall is 9.8 pts (< 10). It halves in high-margin pairs (20.4 vs 8.9). One of three families is < 10. The DAS "illusion" flag depends on rank (false at r = 16) and on seed. |
| 6 | E17–E22 readings | mixed (details in section 6) | E20's DAS full_along_U is SAE-independent by construction, so it is not a width replication. DAS validity is fragile (rank-4 fails in fr and hi, rank-8 passes). The random null is one fixed set of 20 subspaces reused everywhere. |
| – | Code audit | no leakage found; one analysis bug | The deep-analysis donor reconstruction (`mech_donor_pull*.py`) used the wrong pair order, so 97–98 % of donors are wrong on pilot pools. Its conclusion still holds after recomputation. |

---

## 1. "Script, not error size, sets the cost of SAE error" — WEAKENED

**Claim (deep_analysis A; multilingual.md F1).** Dakshina native vs romanized: equal FVU, 1.5–4.1x splice cost. Predictability "is
far too small to explain" the gap.

**What survives (fact, `redteam_ml.py`, `out/redteam_ml_ratios.csv`; 2000 windows per set, 2000 bootstraps).**
- Error size is equal, even on a mean-free measure. FVU uses each set's own mean, so I also compared raw mean |ε|² per position.
  The native/romanized ratio is 0.96–1.27 (FVU 0.85–1.10). FVU is therefore not an artefact of the set-specific mean.
- The direction of the effect holds under every normalisation at L12. ce_delta per token, per word and per character, cost
  relative to clean CE, and 1 − recovered (mean or zero ablation) all exceed 1. For example, hi L12: per token 2.44 [2.37, 2.51],
  1−rec_mean 1.96 [1.91, 2.02], 1−rec_zero 1.94 [1.88, 1.99]. ta L12: 4.13 / 2.98 / 3.10.
- Window alignment is not the issue. Both sets are prefixes of one shuffled sentence order (`build_lang_sets.py`, packed windows),
  and the native set covers fewer words (hi 116k vs 127k; bn 64k vs 99k). Truncating the longer set to the same sentences changes
  no ratio by more than 0.05 (rows `restrict_same_sentences=True`).
- BOS and skip handling are consistent. skip = 4 positions are neither spliced nor scored, and every window has 123 scored
  predictions, so per-window and per-token normalisation give identical results.
- At matched predictability, native-script sets cost 2.3–3.7x English (`redteam_ml2.py`, `out/redteam_ml2_matched_vs_en.csv`;
  0.25-nat clean-CE bins with ≥ 20 windows each, covering 98–99.5 % of target windows). Examples: hi_deva_web L12 2.58,
  ta_taml L12 3.71, bn_web L18 2.50; at 65k, 2.5–4.1. Over the same bins, fr_web is 1.29–1.66.

**Attacks that land.**
1. *Predictability confound: not identified (fact).* Romanized text is far less predictable: median clean CE 5.0 / 5.5 / 6.3 vs
   3.0 / 2.4 / 2.2 nats (hi / ta / bn). The overlap is zero: 0.0–0.1 % of native windows fall inside the romanized 5–95 % range.
   Within every set, splice cost *falls* with clean CE. The mean within-set slope at L12/L18 is −0.035 to −0.092 nats per nat
   (16k). Extrapolated over the 2.0–4.1-nat gap, it explains **46 % [38, 53] (hi L12) to 149 % [135, 162] (ta L18)** of the
   native–romanized cost difference. At L18 the share is ≥ 100 % for all three languages at 16k: hi 1.30 [1.15, 1.46],
   ta 1.49, bn 1.29 (`out/redteam_ml2_slope_share.csv`). The deep report's statement that the slope is "far too small" does not
   hold at L18. Neither conclusion is identified, because the comparison rests entirely on extrapolation.
2. *Normalisation sensitivity at depth (fact).* The 1.5x at hi L18 becomes 1.21 [1.18, 1.24] relative to the mean-ablation cost
   and 0.93 [0.91, 0.95] relative to zero ablation. At hi L24 it is 1.03 [1.01, 1.04] (mean ablation). At ta L18 (16k) it is
   1.15 and 1.07. So the L18/L24 "script effect" largely disappears once cost is scaled by how much the layer's information
   matters for that text. L12 remains robust (≥ 1.3x under every normalisation).
3. *E21 (65k): the romanized baseline is unstable (fact).* At 65k the native/romanized ratio persists (hi 1.94 / 1.63, ta 4.25 /
   3.97, bn 2.69 / 2.68 at L12 / L18). But the ta L18 jump from 1.73 to 3.97 happens because romanized Tamil's cost falls 65 %
   (0.271 → 0.095 nats) while native Tamil's falls 20 %. A ratio whose denominator moves this much between two SAEs of the same
   family is fragile.
4. *Against English, error size is not equal.* Native-script FVU is 1.32–1.77x English's (matched table), so "not error size"
   rests only on the Dakshina contrast, which point 1 undermines. Language and resource level are confounded with script: French
   (high-resource, Latin) costs ~1.2–1.3x English; Hindi, Tamil and Bengali in native script cost 2.3–3.7x.

**Interpretation.** Defensible claim: "for Gemma Scope SAEs, Indic native-script text loses 2–4x more next-token loss than
English at matched clean loss, more than its 1.3–1.8x larger FVU predicts, and a 4x wider SAE does not close it." Not
defensible yet: "script, not error size". **What would settle it:** (i) E23-style norm-matched noise (x̂ + η, x + η) per set, to
test whether native contexts are fragile to *any* perturbation of size |ε|; (ii) per-token NLLs restricted to tokens with
matched clean NLL in both scripts (token-level matching avoids the window-level non-overlap); (iii) a predictable Latin-script
low-resource language (e.g. Indonesian, Swahili web) as the language-vs-script control.

## 2. "The error acts as a uniform gain on task logit differences" — WEAKENED (gain); "decision metrics hide it" survives

**Test (`redteam_gain.py`, `out/redteam_gain.csv`; E8 atlas, f_pilot, pairs with clean A and B correct, n = 89–150 per cell, 36
cells).** Per pair, loss_i = D_A − D_c and swing_i = D_A − D_B. A gain model predicts loss ∝ swing with zero intercept; an
additive model predicts a constant loss.
- **RS L18 (the confirmatory cell): loss is additive.** Affine slope −0.005 [−0.067, 0.054], intercept 0.84 [0.45, 1.24] logits,
  corr(loss, swing) −0.015. Additive residual SD 0.753 vs gain residual SD 0.791. RS L4–16 shows the same pattern (slopes ~0,
  intercepts > 0 with CIs excluding 0 at L4–L16).
- The slope CI is above 0 in 7 of 36 cells: OT L12 (0.22 [0.12, 0.34], intercept 0.05 [−0.30, 0.37], consistent with a
  gain), IOI L12, L14, L18 and L20 (0.09–0.28), RS L20 and VB L20. In 12 of 36 cells the intercept CI excludes 0. The gain model fits better than the
  constant model in only 15 of 36 cells.
- Swing CVs are small (0.13–0.32), so a narrow distribution of r_i follows from either model. "Uniform r_i" does not identify a
  gain.
- The same narrow r_i appears for feat_A_zero (median r 0.99) and feat_A_meanPos (0.98). Whatever shape the loss has, it is
  generic to losing ε_A, not specific to B's error.

**Verdict.** "Decision metrics hide logit loss" survives: it follows from margins, and the project's E14 headroom analysis
supports it. "The error is a gain" holds for OT L12 and deep IOI only. **To settle:** fit the affine model with pair-level
covariates (margin, family) on E3 fresh data (n = 450–737), and test the interpolation model's flip predictions out of sample
for RS, which the deep report already says it fails.

## 3. Object tracking

### 3a. "OT patching is confounded by a prefix fallback" — SURVIVES
This is a property of the pilot prompts (96 of 98 eligible L12 pairs are "stay" pairs; mechanism.md). Its consequences for 3b
are spelled out below.

### 3b. "A single error direction carries the binding update" — WEAKENED
Evidence for the claim: at L12, error-PCA rank 1 gives full_along_U 0.910 [0.846, …] and feat_A_along_U 0.112 (E18, n = 150 with
89 eligible), with random p95 0.045. In E19 steering at matched norm, error-PCA r1 flips 0.64 at ×0.25 vs random p95 0.11 (mean
0.04), and 0.96 vs 0.26 at ×0.5. So the direction is more effective than random directions of equal push. That is real
specificity, and the strongest support for the claim (E19 table, n = 150).

Against it:
- **Under a fallback, "flip to A" means "B's update was erased"**, not "A's state was written". Erasure is cheap. Replacing ε_B
  with norm-matched Gaussian noise already gives A in 89–92 % of pairs at L12 (mechanism.md table, E8/E11). The ε_A swap gives
  98 %, only 6–9 points more. Random steering at ×1 flips 46–79 % (p95, E19).
- **Features do the same along another direction.** Along DAS-1 (L12), x̂_B + P(x̂_A − x̂_B) + ε_B flips **0.989**, and 0.955–1.0
  at ranks 2–16 (E18). The deep analysis calls this "fragility" for DAS but "carrying" for the error direction. One standard has
  to apply to both.
- **The 2×2 interaction (−0.38) is the fallback's signature.** If any mixing of A and B components erases B, then x̂_A + ε_B → A
  and x̂_B + ε_A → A both hold, whatever the information content.
- **Feature-only along an error-fitted U is small by construction.** U is the top variance direction of last-position *errors*,
  so |P_U(x̂_A − x̂_B)| is small relative to |P_U(ε_A − ε_B)|. The random null for full_along_U is not norm-matched either:
  |P_U(x_A − x_B)| ≈ √(r/d)·|x_A − x_B|.
- On fresh templates the swap effect is weaker (P(B | x̂_B + ε_A) 0.36–0.53; mechanism.md), and the OT RQ1 shortfall there is
  1.9 points (RQ1 table).

**To settle (no fallback):** (i) the *reverse* sufficiency patch x_A + P_U(x_B − x_A): does it *write* B's answer into A's run?
Unlike erasure, writing is not explained by fallback. (ii) A design where both A and B contain a move after first_diff.
(iii) full_along_U with random U rescaled to the same |P_U(x_A − x_B)|. (iv) A rank-1 direction from feature-difference PCA as
a structured control.

## 4. "RS error carries an answer-specific component" — SURVIVES the lexical attack; reword as graded magnitude

**Attack 1: lexical overlap (`redteam_resample.py`, `out/redteam_resample.csv`).** Same-sum donors might share more operand
tokens with A. They do not, to any material degree. Token agreement on the patched span is 0.727 vs 0.725 (E3 eval fresh) and
0.762 vs 0.750 (E2); digit-token agreement is 0.264 vs 0.279 (E3) and 0.280 vs 0.201 (E2). Regressing the per-pair same−diff NLDR
effect on the overlap difference leaves the intercept intact: +0.074 [0.053, 0.094] (E3 fresh, n = 532), +0.074 [0.042, 0.104]
(E2, n = 202), +0.056 [0.026, 0.088] (E2r, n = 201). Among pairs with *equal* digit overlap: +0.101 [0.055, 0.150] (E3, n = 140).
**The attack fails.**

**Attack 2: exact answer or magnitude (`redteam_resample2.py`, `out/redteam_resample2_distance.csv`).** Splitting by |value(C) −
value(A)| for the different-answer donor, the paired same−diff NLDR difference is:
- ±1: +0.021 [−0.006, 0.050] (E3 fresh, n = 222); −0.005 [−0.045, 0.032] (E2, 73); −0.010 [−0.060, 0.050] (E2r, 74)
- ±2: +0.028 [−0.007, 0.064]; +0.036; +0.007
- ±3: +0.162 [0.113, 0.216]; +0.200 [0.130, 0.269]; +0.098 [0.036, 0.167]
- ±4: +0.210 [0.135, 0.285] (E3); +0.180 [0.088, 0.277] (E2r)

The diffAns slope on distance is −0.10 [−0.134, −0.066] NLDR per unit (E3).

**Interpretation.** The error carries a graded representation of the running sum's *magnitude*. Near-value donors are as good as
same-value donors. That is consistent with the near-miss pattern (73 % ±1 failures), and calling it "answer-specific" overstates
it. The +7.5 average comes from far donors. Caveat: distance is not randomised (it depends on A's value and the donor pool), but
the comparison is paired within pairs.

**Bug found in the deep analysis (fact).** `deep/scripts/mech_donor_pull.py` and `mech_donor_pull2.py` rebuild donors over pairs
sorted by `pair_id`. The harness selects donors over `te` sorted by `(family, rank)` (`harness/run.py` run_stage3;
`load_spans` keeps that order), and `select_donors` draws sequentially from one RNG. Only **1.9 % (E2), 2.8 % (E2r) and 2.3 %
(E3 eval-pilot)** of donors coincide between the two orders; the E3 eval-fresh pools coincide 100 %
(`out/redteam_resample2_identity.csv`). The stored `donor_status` is almost always "ok", so the "100 % status match" check cannot
detect this. With the correct donors, the conclusion is unchanged: diffAns − noise is −0.022 [−0.052, 0.009] (E3 fresh), so
there is no pull beyond noise, while diffAns − errB is +0.064 [0.039, 0.088] (`out/redteam_resample2_pull.csv`). Also, the E3
fit=pilot and fit=fresh resample rows are the *same patch* (resampling uses no fitted object). They are duplicates, not
replications.

**To settle exact vs graded:** resample donors stratified by |ΔA| ∈ {0, 1, 2, ≥ 3} by design, with fixed donors across pairs.

## 5. RQ1 GO for RS L18 and the E9 readings

**RQ1 — survives as preregistered (fact, `redteam_rq1.py`).** My own code reproduces the gate: fresh families, f_pilot,
n = 450 eligible, full_A − meanPos flip shortfall **0.147 [0.113, 0.180]**. Qualifications:
- With B's actual error (feat_A_errB) the same cell's shortfall is **0.098**, below the 10-point bar. GO depends on the
  preregistered meanPos baseline. That is legitimate, but "~12 pts" should not be quoted for feat_A_errB on fresh templates.
- It depends on margins. Splitting at the median clean lead of A over the best other candidate: low lead (median 1.76 logits)
  0.204, high lead (median 3.23) 0.089. This matches the project's own point B: decisions flip where margins are small.
- It varies by family: rs_plus 0.087, rs_story 0.213, rs_symbolic 0.140 (n = 150 each). The family-cluster CI rests on 3
  clusters, so it is not a meaningful interval.
- No "x_A + norm-matched noise" task control exists. Without it, "the shortfall measures information in ε_A" cannot be separated
  from "small-margin RS decisions are fragile to any perturbation of size |ε|". That is E23's task analogue, and it was not run.

**E9 readings — weakened.**
- The `illusion` and `feature_mediated` flags are both true for DAS-4/8. As labels they contradict each other, and the illusion
  flag is not stable. DAS-16 (E18) is *not* illusion: full 0.780 vs feat 0.707, while it remains feature-mediated.
- Seed sensitivity (E17, n = 150). das4 full 0.760 vs das4_s1 0.593; feat 0.547 vs 0.307. das8 full 0.867 vs 0.733. So "features
  carry 55–68 % of DAS's effect" is seed-dependent (31–55 % at rank 4).
- "Effective" means beating a random p95 of 0.000. Random U is not norm-matched (see 3b), so every DAS fit with train accuracy
  ≥ 0.5 clears this bar automatically.
- E9 compares error-U fitted on the **final-position errors only** (`bundle`: `E_last`) and applied at all span rows against DAS
  trained on **all span rows** with a supervised loss. So "error subspace not effective, DAS effective" confounds the estimator
  with position coverage and supervision. Error-U effectiveness at RS L18 only appears at r ≥ 32 (full 0.153–0.267, E18).

## 6. New experiments E17–E22

- **E17 (DAS robustness).** Rank-1/2 failure is *not* fixed by training longer (10 or 30 epochs) or by new seeds: train accuracy
  ≤ 0.473 in all 16 rank-1/2 fits. With lr 3e-2, rank 1 diverged (train acc 0.000, loss 3.77 above the ~2.1 starting loss), and
  the fit was correctly marked invalid. E9's rotations reproduce exactly on node-D (das4 0.760, acc 0.733; das8 0.867, 0.890).
  DAS keeps 56–67 % of its energy in the random initial subspace even after 10 epochs (das8_e10 0.584), and 76 % at rank 16
  (E18). The learned components agree across seeds (top learned cosine 0.86–0.92 at rank 4/8), but the mean learned cosine is
  only 0.52–0.60. A *direction* is reproducible; a *subspace* is not. The "shared initialisation artefact" correction survives.
  Any geometric statement about DAS must use the learned component.
- **E18 (rank).** PC error-U has zero effect up to r = 64 at L12 and L16 (full_along_U 0.000; NLDR ≤ 0.10), although the PC gap is
  22–24 %. So PC's lost information is not in a low-rank last-position error subspace. DAS for PC is valid only at rank 16
  (acc 0.81 / 0.78). The OT random p95 grows with rank (0.045 at r1 to 0.228 at r64), which again shows OT fragility.
- **E19 (steering dose).** The Q2 null for RS is robust across doses: max flip 0.22 (das8, ×16), below 0.5 at every scale.
  OT dose curves are non-monotone (error-U r1 peaks at 1.00 at ×1 and falls to 0.19–0.30 at ×8–16) because of overshoot.
  Random baselines are nested and shared: one set of 20 subspaces per (cfg.seed, layer) (`Dr.random(d, max_rank, seed+101s+layer)`),
  reused across conditions, scales, tasks at the same layer and experiments with seed 0. The random p95 is close to the
  19th–20th order statistic of 20 draws. That is enough for a 5 % bar, but it is not a 20-fold independent replication across
  tables.
- **E20 (width).** For DAS, `full_along_U` does not depend on the SAE (x_B + P_U(x_A − x_B), rotation cached on the full
  residual). The identical 0.760/0.867 at 16k and 65k is therefore by construction, not a replication. Only feat_A_along_U is
  informative: das4 0.547 (16k) vs 0.620 (65k), and das8 0.680 vs 0.727. Wider-SAE features carry somewhat more of DAS's
  variable. The OT error-U "illusion" persists at 65k and 262k (r1 full 0.876–0.921 vs feat 0.045–0.067).
- **E21.** See section 1.3: the gap persists at 65k, and English benefits more from width than native text. At L12, hi_deva_web
  ce_delta goes 0.285 → 0.236 (−17 %) while en_pile goes 0.114 → 0.083 (−27 %).
- **E22 (multilingual DAS).** Validity is fragile. Native rank-4 fits fail in fr (acc 0.247) and hi (0.160) while rank 8 passes
  (0.593 / 0.880). DAS for PC is invalid at every rank ≤ 8 in every language (acc ≤ 0.453). The DAS effect depends on the prompt
  set: rs_en from the e13 set gives das4 0.573 (acc 0.633), vs 0.760 on the pilot templates. English-fit rotations transfer to
  fr, hi and hinglish at 0.52–0.62 full flip (rank 4–8), but non-English RS rows fail the capability gates (E13), so this is
  descriptive only.

## 7. Code audit (harness/fidelity.py, final.py, e8.py, conditions, das.py, run.py, make_final.py, run_fidelity.py)

No leakage or sign errors found.
- **Fit vs test.** Bundles use `T.select(rows, fl, "train", fit_filter)`. `FitSet.require_train` guards MeanError, CovNoise,
  MeanDiff (steering), pca/answer_sep and greedy sub-threshold, and `bun.fitset.assert_disjoint(test ids)` runs before stage 3.
  DAS is trained on train spans and evaluated on test pairs. **DAS "fit valid" uses train interchange accuracy (in-sample).** That
  is a reasonable fit criterion, but the rotation has 2304·k parameters fitted on ~150 pairs, so held-out accuracy would be the
  better validity gate.
- **Metrics.** NLDR and prob_recovery are ratios of means. Flip eligibility is clean_B top-1 = B (e8.metric_samples). `gap = M(full_A)
  − M(cond)`. full_A − feat_A_along_U = P_U(ε_A − ε_B) as documented. Signs check out.
- **Paired contrasts.** `paired_diffs` intersects pairs and shares resamples. `condition_table` silently skips a condition with
  any missing pair (`top1_id.isna()`), whereas resample rows with no donor are filled with ε_B placeholders and are *not* NaN.
  So resample rows in `*_conditions.csv` include placeholder pairs (2–5 per cell), a conservative bias.
- **Position coverage.** Error-U uses final-position errors only; DAS and steering use all span rows (section 5).
- **Random U.** Shared and nested across conditions, tasks and experiments, and not norm-matched for full_along_U (sections 3b
  and 6).
- **E12.** Splice, scoring, FVU and the mean all use positions ≥ skip (4). The ratio-of-sums summary equals the per-token mean
  because every window has the same number of predictions. The FVU denominator uses each set's own mean, but the mean-free
  |ε|² per position gives the same picture (section 1).
- **Analysis bug (deep analysis, not harness).** Donor reconstruction order in `mech_donor_pull*.py` (section 4).
- The DAS init depends only on (k, seed) and not on task or layer. That is known, and make_final now reports it.

## Experiments that would settle the open points (priority order)
1. E23 as planned (norm-matched η per set and per task: x̂ + η and x + η), plus a task-level x_A + η control for RQ1.
2. OT sufficiency patch x_A + P_U(x_B − x_A), with a norm-matched random U, on a design without the fallback.
3. Token-level matched splice cost (native vs romanized tokens with equal clean NLL), plus one predictable Latin-script
   low-resource language.
4. Resample donors stratified by |ΔA|; DAS validity on held-out pairs; DAS rank 1–16 with ≥ 3 seeds reported as distributions.

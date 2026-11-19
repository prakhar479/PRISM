# Steering / intervention dose-response (E19, E10, E10b): role `steering`

**Status:** everything here is exploratory and post hoc (E19 is exploratory by its config; E10's RS L18 Q2 reading is the only
preregistered steering result and it is unchanged: not met). DEV analysis of stored per-pair outputs only; no model runs.
E23 (norm-matched noise controls) was NOT run; nothing below depends on it.

**Data:** `ana_runs/E19/deb20460858c` (gnode058), `ana_runs/E10/9eb5f4d77046`, `ana_runs/E10b/23529335daaf` (gnode084);
patching comparisons from `r5/tables/E18_conditions.csv` (E18) and `r5/tables/E17_conditions.csv` (das8_e10, RS only).
**Primary filter:** f_pilot (RS n = 150 eligible, OT 89, PC 150); f_none is the robustness check (RS 209, OT 106, PC 152).
Flip rate = share of eligible pairs (clean-B top-1 = B's answer) whose top-1 becomes A's answer; NLDR as harness/metrics.py.
**Steering:** x_B + s·|m|·P_U m/|P_U m| at every patched row, m = train mean of x_A − x_B per row key (no test x_A used).
**Random envelope:** 20 Haar U per rank (r = 1, 4, 8), same construction; mean / p95 (np.nanpercentile) / max over seeds.
**Specificity** = estimator flip − random p95; 95 % CI by pair bootstrap (1000) recomputing the random p95 inside each resample;
one-sided bootstrap p for spec > 0, BH within (experiment, filter): E19 189 tests per filter, E10/E10b 81 each.

**Scripts** (`team/scripts/`, run from that directory with the project venv; outputs in `team/steering_out/`):
`steering_common.py` (loaders), `steering_dirs.py` (U identity across runs, DAS meta), `steering_dose.py` (all dose tables,
envelope, specificity; reproduces `r5/tables/E19_steering.csv`, `tables/final/E10_steering.csv`, `E10b_steering.csv` exactly:
max |Δflip| = 0, |ΔNLDR| < 2e-8, p95 and max identical, 702 rows checked), `steering_outcomes.py` (where the answer goes: OT box
roles, OT fragility, RS shift by Δ = A − B, PC landing city), `steering_fit.py` (logistic fits in log2 s), `steering_link.py`
(steering vs patching with the same U; mean-component bound; RS ceiling; E10b Δ split), `steering_das_epochs.py`
(das8 vs das8_e10), `steering_repro.py` (E10 vs E19).

---

## Headline

1. **No regime meets the Q2-type bar** (flip ≥ 0.50 with random p95 ≤ 0.05) in any measured cell: 0 of 702 (E19 + E10 + E10b,
   f_pilot and f_none). Each task fails for a different reason, and the reasons are informative:
   * **RS L18:** steering is specific but weak (DAS: ≤ 0.22 flips, random p95 ≤ 0.013 up to s = 4). It works as a **constant
     "add k" dial**: every DAS U shifts the predicted sum UP for every pair, so only pairs with A > B can flip
     (flips on A < B pairs: 0 for das4 at every scale and das8 at s ≤ 8, 0.016 for das8_e10 at s = 8; 0 for das1/2/4 at every
     scale in all three E10b fresh families). A single additive
     vector that shifts every answer by the same k can flip at most max_k P(A − B = k) = **0.353** of eligible pairs (k = +1).
     So the 0.50 bar is out of reach for mean-difference steering on this task by construction, not because of a poor U.
   * **OT L12:** steering is effective at every dose ≥ 0.25 for every U (DAS 0.98–1.00 at s = 0.25) and is far above the random
     mean at low dose (DAS spec 0.83–0.86 at s = 0.25), but random directions also flip (p95 0.11–0.15 at s = 0.25, 0.46–0.80 at
     s = 1). The random flips are the **prefix fallback**: at s ≤ 1, 99–100 % of random-steered non-B outcomes are A's box
     (null 1/6). The flip target is a default, so OT steering cannot show that U encodes "which box".
   * **PC L12:** nothing flips (≤ 0.027 at s = 16, random 0). DAS steering pushes all pairs toward a few fixed cities
     (das8 s = 16: 93 % of the non-A/B landings are "Tokyo"); A's city is pair-specific and the train mean cannot encode it.
2. **Patching effect and steering effect of the same U are rank-correlated across cells** (Spearman 0.91 for patch flip vs best
   steer flip, n = 25 cells, p = 2e-10, uncorrected, largely driven by the task difference), but a causally sufficient
   subspace is NOT a good steering direction where the A–B contrast is pair-specific: RS DAS-8 patching flips 0.87 vs best steering
   0.22; PC DAS-8 0.31 vs 0.027. For RS and PC the mean component can account for **at most 1–2 %** of the patching NLDR
   (bound below); the rest needs the pair's own x_A − x_B.
3. **Longer DAS training does not steer better.** das8_e10 vs das8 (identical init and data): RS no difference at s ≤ 8 and worse at
   s = 16 (0.12 vs 0.22, McNemar q = 0.007); OT better at s = 1–2 only because it overshoots less (1.00 vs 0.88, 0.85 vs 0.64,
   q ≤ 0.007); PC no difference. In E10 the failed DAS-1 fit (train acc 0.25) steers RS as well as the valid DAS-4 (0.17 vs 0.18).

---

## Findings, ranked by strength

### F1 (strong). RS L18: DAS steering is a one-signed value dial; flips are capped by the Δ = A − B distribution

FACT (`rs_delta.csv`, `rs_shift.csv`, `rs_ceiling.csv`, `e10b_delta.csv`; E19 f_pilot, 150 eligible pairs; E10b f_none):
* Every DAS U pushes the predicted number up. das4: share of pairs whose top-1 moves below B = 0.000 at every s; share moved up
  0.20 (s = 2), 0.57 (s = 4), 0.76 (s = 8), 1.00 (s = 16); mean shift (top-1 − B) 0.31 / 1.12 / 2.00 / 3.22.
  At s = 4 the shift is bimodal: 0 (43 %) or +2 (42 %), +1 only 8 %, +3 6 %.
* Hence flips only on pairs with A > B. das4 s = 4: flip 0.846 on Δ = +2 pairs, 0.075 on Δ = +1, 0.143 on Δ = +3, **0 on every
  Δ < 0 pair**. A < B pairs flip 0 for das4 (all s) and das8 (s ≤ 8); das8_e10 0.016 at s = 8. pca_pooled pushes DOWN
  (r1 s = 16: mean shift −0.33, flips only on Δ < 0, 0.19); answer_sep pushes up. Random U: no systematic shift
  (s = 16: mean −0.06, 86 % unshifted).
* Replication on fresh templates (E10b, s = 1/2/4, rs_plus 207 / rs_story 185 / rs_symbolic 156 eligible pairs, f_none):
  flips on A < B pairs = 0.000 for das1, das2, das4 at every scale in all three families; das4 s = 4 flips 0.83 / 0.61 / 0.64
  of Δ = +2 pairs.
* Ceiling: eligible f_pilot Δ distribution −3 .053, −2 .207, −1 .167, +1 .353, +2 .173, +3 .047 (f_none: max .359). A vector
  that adds the same integer k to every answer can flip at most 0.353. Observed best: 0.22 (das8, s = 16), 0.18 (das4, s = 4,
  ≈ the +2 share 0.173 × 0.85 + residue).
* Specific but weak: das4 s = 4 flip 0.180 [0.126, 0.247], random mean 0.002, p95 0.007, spec 0.173 [0.113, 0.233], BH q = 0.005;
  NLDR 0.273 vs random mean 0.005. das8 / das8_e10 at s = 4–8: 0.14–0.18, spec 0.13–0.14 (q = 0.005). Same in f_none
  (das4 s = 4: 0.163, spec 0.148 [0.096, 0.196]). Error-derived U (pca_pooled, answer_sep): flips ≤ 0.13 and only at s ≥ 8
  where random p95 is 0.03–0.15; no pca/answer_sep cell survives BH in f_pilot.
* Overshoot: beyond s = 4 (das4) the answer moves past A to larger numbers (out_other_cand 0.73 at s = 8, 0.97 at s = 16),
  not to non-numbers (non-candidate top-1 = 0 throughout RS).

INTERPRETATION: the train mean difference m projected on a DAS subspace is a "bigger number" vector (the first-300 train
candidates have mean Δ = +0.24, mean |Δ| = 1.49; the actual f_pilot fit set's flags are not stored, so this is a proxy).
Steering with it moves the running-sum variable coherently (a dial), which is evidence that the DAS subspace carries the value in
a graded way. But flipping to A needs a pair-specific shift of either sign, which no single added vector can supply; Q2's
"flip ≥ 50 %" bar is unattainable for RS with this design even with a perfect U. Alternative: the +2 preference could reflect
how m mixes slot/offset rows rather than a value code; not testable without m.

### F2 (strong). OT L12: random-direction flips are the prefix fallback; estimator steering is far more potent per unit norm but its target is the same default

FACT (`ot_roles.csv`, `ot_fragility.csv`, `dose_summary.csv`; 89 eligible pairs, 20 seeds → 1780 random rows per (r, s)):
* Random steering at s ≤ 1: among non-B candidate outcomes, A's box is 99.2–100 % (r = 1/4/8; e.g. r = 4, s = 1: 0.991
  [0.976, 1.000] pair-cluster bootstrap; null 1/6 for a uniform pick among 6 non-B boxes; mentioned-box null 0.41). At s = 2:
  0.92 / 0.96 / 0.985. Only from s = 4 do random pushes land on other boxes (A share 0.68–0.80; other mentioned boxes
  0.16–0.22), at s = 8 0.39–0.48, and at s = 16 mostly non-box tokens (0.56–0.66).
* So "random flips" are not generic disruption to any non-B answer: they are reversions to the answer implied by the unpatched
  prefix (mechanism.md §1). Which pairs flip is a pair property: the per-pair fraction of seeds that flip it correlates with the
  clean-B margin (Spearman of fraction vs logit_B − logit_A: −0.85 at r = 4 and r = 8, s = 1; −0.84 at r = 1; n = 89, p < 1e-20).
  No pair is flipped by all 20 seeds at s ≤ 1 (r = 8: 84 % flipped by ≥ 1 seed).
* Estimator steering at s = 0.25: DAS4 0.978 [0.944, 1.0], DAS8 1.000, DAS8_e10 0.978 vs random mean 0.058–0.077, p95
  0.114–0.146, max 0.146; spec 0.83–0.86 (lower CI ≥ 0.75; q = 0.005). pca/answer_sep (r = 1, 4, 8): 0.58–0.67, spec 0.47–0.53.
  f_none: DAS 0.887–0.915, spec 0.745–0.773. NLDR at s = 0.25: DAS 0.78–0.90, pca/answer_sep ≈ 0.19–0.21, random mean ≤ 0.025.
  At s = 1 every U flips 0.88–1.00 but random p95 is 0.46 (r = 1), 0.64 (r = 4), 0.79 (r = 8): specificity falls to 0.09–0.54.
* Overshoot is estimator-specific: DAS goes to non-box tokens at s = 8–16 (das4 0.78 / 1.00; das8_e10 1.00 from s = 8) and to
  other mentioned boxes at s = 2–4 (0.24–0.38); pca / answer_sep go to other boxes, mostly unmentioned ones, at s = 4–16
  (answer_sep r1 s = 8: 0.97 unmentioned). NLDR overshoots for DAS (2.7–3.7 at s = 1–2) while top-1 stays A.
* E10 (s = 1/2/4, independent random draws because the Haar draw shape depends on max rank): random mean at s = 1 0.22–0.27,
  p95 0.58–0.73; same picture.

INTERPRETATION: in OT, DAS and error-PCA directions are 13–17× (DAS) and 9–17× (pca, answer_sep) more effective per unit push
than random directions at s = 0.25, so U ≫ random holds at low dose. But the flip target is the fallback that any destruction of
"the queried object moved at the changed step" produces, so OT steering demonstrates that U carries the *move event* (or its
read-out), not which box. A test that can separate the two needs pairs where A ≠ prefix default ("moved" pairs), which the
model fails (clean-B 18 %, mechanism.md) and which are absent from the eligible set.

### F3 (strong, negative). PC L12: steering never flips; DAS pushes toward fixed cities

FACT: max flip over all U and s = 0.027 (das8, s = 8–16; random 0 at every s; `dose_summary.csv`). NLDR ≤ 0.23 (das4 s = 16).
At s = 16 DAS moves top-1 off B in 37–49 % of pairs (out_B 0.51–0.63) but to a near-constant city: das8 "Tokyo" 0.93 of 70
other-candidate landings, das4 "Paris" 0.57 of 54 (`pc_landing.csv`). Random steering's rare landings are "Oslo" (100 % at
s ≤ 2, 4–22 rows of 3000). The first-300 train candidates' A-minus-B city counts favour Delhi +13, Paris +7, Tokyo +6, Oslo +4.
Under a one-hot city code, |mean(e_A − e_B)| / |e_A − e_B| = 0.073 / 1.414 = 0.05 (proxy fit set).
DAS fit validity: das4 train acc 0.14, das8 0.44 (both < 0.5, invalid), das8_e10 0.54 (valid) — it does not steer better
(0.020 vs 0.027 at s = 16).
INTERPRETATION: the A–B contrast in PC is a pair-specific identity (which city); its train average is ≈ 5 % of a pair's contrast
under one-hot coding and points to over-represented cities. Mean-difference steering cannot work here even if U is right.

### F4 (moderate). A good patching subspace is a good steering direction only where the contrast is shared across pairs

FACT (`patch_link.csv`; patching = E18 full_along_U f_pilot, same U bitwise — `dirs_identity.csv`; das8_e10 from E17):
| task | U | patch flip / NLDR | best steer flip (s) | steer NLDR at s = 1 | mean-share bound |
|---|---|---|---|---|---|
| RS | das4 | 0.760 / 0.748 | 0.180 (4) | 0.014 | ≤ 0.019 |
| RS | das8 | 0.867 / 0.878 | 0.220 (16) | 0.007 | ≤ 0.008 |
| RS | das8_e10 | 0.900 / 0.903 | 0.167 (8) | 0.011 | ≤ 0.012 |
| PC | das4 | 0.100 / 0.307 | 0.013 (16) | 0.003 | ≤ 0.009 |
| PC | das8 | 0.313 / 0.488 | 0.027 (8) | −0.004 | ≤ 0 |
| OT | das4 | 1.000 / 1.853 | 1.000 (0.5) | 2.830 | uninformative |
| OT | pca r1 | 0.910 / 0.427 | 1.000 (1) | 0.652 | uninformative |
Across the 25 (task, U) cells with both: Spearman(patch flip, best steer flip) 0.913 (p = 2e-10), (patch flip, best spec) 0.933,
(patch NLDR, max steer NLDR) 0.867; within task (patch NLDR vs max steer NLDR): RS 0.65 (n = 9, p = 0.058), OT 0.95 (n = 8,
p = 0.0003), PC 0.83 (n = 8, p = 0.010). 7 correlations, uncorrected.
**Mean-component bound** (`steering_link.py` docstring): P_U(x_A − x_B) = P_U m + P_U δ_i; the mean part alone is steering at
s0 = |P_U m|/|m| ≤ 1. The measured steering NLDR is non-decreasing (within 0.005) over s = 0.25 → 1 in every cell, so if it is monotone
on (0, 1] the mean part produces NLDR ≤ NLDR_steer(1). For RS DAS that is ≤ 0.8–1.9 % of the patching NLDR, for PC DAS ≤ 0.9 %;
≥ 98 % of the patching effect therefore requires the pair-specific component δ_i. For OT the bound is > 1 (steering at s = 1
already exceeds patching) and says nothing.
CAVEATS: |P_U m| and |P_U(x_A − x_B)| cannot be computed: neither m nor activations are stored (directions/*.npz contain U only;
no activation caches on this machine). The bound assumes monotonicity below s = 0.25, m_train ≈ m_test, and treats a nonlinear
model's effects as comparable. The cross-task correlation is mostly the OT-vs-rest split.
INTERPRETATION: causal sufficiency for patching (pair's own x_A) and steerability (one shared vector) are different properties.
They coincide in OT because the relevant contrast ("one more move") is the same sign in every pair; they diverge in RS (signed
value difference) and PC (identity), where the patching success of DAS is almost entirely pair-specific.

### F5 (moderate). Longer DAS training does not improve steering; steering does not track DAS fit validity

FACT (`das_epochs.csv`, paired exact McNemar per scale, 21 tests, BH): RS das8_e10 vs das8 equal at s ≤ 8 (e.g. s = 4 0.147 vs
0.140, q = 1), worse at s = 16 (0.120 vs 0.220; 2 vs 17 discordant, q = 0.007). OT: e10 better at s = 1 (1.000 vs 0.876, 11 vs 0,
q = 0.007) and s = 2 (0.854 vs 0.640, 20 vs 1, q < 0.001), equal elsewhere; the difference is less overshoot to other boxes, not
more flips at low dose (s = 0.25: 0.978 vs 1.000). PC: equal (q ≥ 0.5). Patching for comparison (E17/E18, RS): das8_e10 0.900
vs das8 0.867. DAS meta (`das_meta.csv`): RS das8_e10 train acc 0.98 vs 0.89, loss 0.33 vs 0.70.
E10 (RS, s = 4): das1 (train acc 0.25, an invalid fit) flips 0.167, das2 (0.33) 0.200, das4 (0.73) 0.180; E10b best: das1
0.227 / 0.147 / 0.107 vs das4 0.220 / 0.167 / 0.147 (rs_plus / story / symbolic).
INTERPRETATION: steering success in RS is set by the direction of P_U m ("bigger number"), which even a poorly-fitted
low-rank DAS captures; interchange accuracy is not the relevant property.

### F6 (moderate). Error-derived U (pca_pooled, answer_sep) never steer specifically in RS or PC

RS: flips ≤ 0.027 up to s = 4, ≤ 0.133 at s = 16 where random p95 is 0.074–0.15; 0 of 42 pca/answer_sep RS cells pass BH in
f_pilot (in f_none, answer_sep r1 s = 8 / 16 and pca r1 s = 8 pass with spec 0.03–0.06). PC: 0 flips. Matches their
patching failure (E18 full_along_U RS ≤ 0.07 for r ≤ 8; PC 0). In OT they behave like weaker DAS (F2).

### F7 (supporting). Reproducibility

* U matrices are bitwise identical across E9, E10, E17, E18, E19 for every shared (task, estimator, rank) (`dirs_identity.csv`,
  max |ΔU| = 0), including DAS rotations retrained on gnode058.
* E10 (gnode084) vs E19 (gnode058) estimator-steered outputs: identical (max |ΔD| = 0, top-1 agreement 1.000 on every shared key;
  `repro_E10_E19.csv`). Random rows differ because the Haar draws differ (draw shape depends on max rank: 4 in E10, 8 in E19);
  their flip rates agree closely (e.g. OT s = 1 pooled shared seeds 0.244 vs 0.249; s = 2 0.468 vs 0.458), an independent
  replication of the null.

### F8 (weak / descriptive). Logistic fits in log2 s: where an effective-and-specific window could be

Model flip = top·σ(β(log2 s − c)), binomial MLE on the rising part only (s ≤ scale of peak flip); random seeds fitted one by one;
pair-bootstrap 200 (`fit_params.csv`, `fit_window.csv`).
* OT pca_pooled / answer_sep: top = 1, s50 = 0.20–0.23 (CIs 0.17–0.26); random seeds: median s50 3.44 (r = 1), 1.17 (r = 4),
  0.93 (r = 8) with 17 / 14 / 18 of 20 seeds reaching 0.5 → error-U are ≈ 4–16× more potent in dose than random U. At
  s = 0.125 (extrapolated) the fitted random p95 is 0.026–0.030 but the fitted error-U flip is only 0.09–0.17: **no window**
  where error-U flip ≥ 0.5 and random p95 ≤ 0.05, even by extrapolation.
* OT DAS: flips are already 0.98–1.00 at the lowest dose (0.25), so s50 is below the measured range and NOT identifiable
  (fits hit parameter bounds; das8's fitted "window" 0.03–0.15 is an extrapolated step and not credible; das4 / das8_e10 give no
  window on the same data). SPECULATION: the only candidate for a Q2-type regime is OT DAS at s ≈ 0.06–0.125; it would still
  carry the prefix-fallback caveat (F2). A run at s ∈ {0.03, 0.06, 0.125} would settle it.
* RS: tops 0.08–0.36 (das4 0.36), s50 undefined; PC: tops ≤ 0.28 with flips ≤ 0.027, undefined. Random RS: s50 > 59 in the few
  seeds where defined.

---

## Answers to the brief's questions

| question | answer (all exploratory) |
|---|---|
| Any regime effective (≥ 50 %) and specific (random p95 ≤ 5 %)? | No measured cell (0 / 702). OT is effective but not specific at s ≥ 0.25 (best: DAS s = 0.25, 0.98–1.00 vs p95 0.11–0.15); RS is specific but not effective (≤ 0.22); PC is neither. |
| U ≫ random? | Yes in OT at s = 0.25 (DAS spec 0.83–0.86, 13–17× the random mean) and in RS for DAS at s = 2–8 (spec 0.05–0.17, BH q ≤ 0.02). Not for error-U in RS/PC. |
| Where does the mass go? | RS: up the number line (DAS), overshooting to larger digits; never to non-numbers. OT: to A at low dose; DAS overshoot to non-box tokens (s ≥ 8), error-U and random to other boxes (s ≥ 4). PC: stays on B, or goes to a fixed city. |
| OT random flips = prefix fallback? | Yes: 99–100 % of random non-B outcomes at s ≤ 1 are A (null 1/6); flip propensity tracks the clean-B margin (ρ = −0.85). |
| das8_e10 better than das8? | No for RS and PC; in OT only less overshoot at s = 1–2. |
| Patching sufficiency ⇒ good steering? | Only when the contrast is shared (OT). In RS / PC the mean component explains ≤ 2 % of the DAS patching NLDR (bound); patching success is pair-specific. |

## Multiplicity and tests
Specificity: 189 bootstrap tests per filter in E19 (39 BH-significant at q < 0.05 in f_pilot, 42 in f_none), 81 each in E10 and
E10b (17 / 15 in f_pilot). das8 vs das8_e10: 21 McNemar tests, BH. Steering–patching Spearman correlations (7) and the OT
fragility correlations (21) are uncorrected; all quoted p-values are far below any reasonable correction except RS within-task
(p = 0.058). OT role shares use a pair-cluster bootstrap (1000); seeds share pairs, so rows are not independent.

## Caveats and alternatives
* m and activations are not stored: the |P_U(x_A − x_B)| vs |P_U m| comparison asked for cannot be done directly; F4 uses a
  behavioural bound instead and F1/F3 use task-structure proxies (train Δ / city counts from the first 300 train candidates,
  not the exact f_pilot fit set, whose flags are not on disk).
* Steering adds |m| (the FULL mean difference's norm) along U at every patched row, so s = 1 is not "the size of the A–B
  difference along U"; scales are comparable across U and random but not to patching magnitudes.
* OT eligible pairs are 89 (f_pilot), almost all "stay" pairs; OT results are about this template (mechanism.md §1: weaker on
  fresh templates). PC DAS fits at rank 4 / 8 are invalid (train acc 0.14 / 0.44).
* E10b uses fresh families but U fitted on pilot templates; its "f_pilot" rows are the 150-pair subsets used in the tables.
* No E23 (norm-matched noise): the random-U envelope is the only null here, and it is a direction null at matched norm.

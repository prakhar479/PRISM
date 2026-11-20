# Geometry and SAE width (role: geometry), 2026-11-19

Everything below is **post hoc and exploratory**. It uses existing run outputs only; nothing was run on a GPU. Paths are relative
to `$SP` = `.layout`.

- Scripts: `team/scripts/geometry_*.py`.
- Outputs: `team/geometry_out/*.csv`.
- Bootstraps: 2000 pair resamples, seed 0, using the same resamples at every width, so all cross-width differences are paired.
- Every CI below is an uncorrected 95% percentile CI unless stated otherwise.

**Testing load.**
- The width comparisons cover 2 tasks × {gap, DAS capture, error-U repair} × ranks × widths. That is 30 repair-share
  differences and 12 DAS-capture differences.
- Most of their CIs sit far from 0, but they are not BH-corrected.
- The Spearman tests in finding 4 are BH-corrected (24 tests).

**Sanity and provenance checks** (`geometry_width_capture.py`, `geometry_width_subspace.py`):
- E9 and E20 share the DAS rotations (das4/das8): max |ΔU| = 0 for RS L18, OT L12 65k and OT L12 262k.
- E9's U equal the first 32 columns of E18's (max |Δ| = 0).
- E20's error U equal E4's for the same SAE (max |Δ| ≤ 1.4e-14).
- The clean_A/clean_B logits are identical between the E9 and E20 per-pair files (max |Δlogit| = 0), with 0 filter
  disagreements.
- The E20 feat_A_errB NLDR gaps reproduce E4's to 4 decimals: RS 65k 0.1129, OT 65k 0.2412, OT 262k 0.1466
  (`hw/tables/E4_width_l0.csv`).
- In short, E20 adds no new error geometry beyond E4. What it adds is the along-U, repair and DAS conditions at the wider widths.

---

## Findings, ranked by strength

### 1. A wider SAE carries more of the DAS causal subspace, but a large part stays in the error (strong, paired)

**Setup.**
- E9 (16k) and E20 (wider widths) use the same DAS U and the same test pairs.
- Because of that, `full_along_U` is identical across widths: NLDR 0.748 for RS das4 and 1.853 for OT das4.
- `feat_A_along_U = x̂_B + P_U(x̂_A − x̂_B) + ε_B` therefore isolates how much of the causal subspace each SAE's reconstruction
  carries.

**Measures.**
- *capture* = NLDR(feat_A_along_U) / NLDR(full_along_U).
- Paired differences are wide minus 16k (`geometry_width_capture.py`, `width_capture.csv`).

Results (f_pilot, n = 150; CI in brackets):

| cell | DAS | capture 16k | capture 65k | capture 262k | Δ feat_A_along_U NLDR (65k − 16k) | Δ (262k − 16k) |
|---|---|---|---|---|---|---|
| RS L18 | das4 | 0.754 [0.722, 0.785] | 0.827 [0.799, 0.854] | – | +0.055 [0.035, 0.075] | – |
| RS L18 | das8 | 0.763 [0.734, 0.791] | 0.842 [0.814, 0.868] | – | +0.069 [0.048, 0.091] | – |
| OT L12 | das4 | 0.532 [0.515, 0.548] | 0.574 [0.557, 0.590] | 0.591 [0.574, 0.607] | +0.078 [0.063, 0.093] | +0.110 [0.092, 0.128] |
| OT L12 | das8 | 0.578 [0.562, 0.593] | 0.622 [0.606, 0.637] | 0.632 [0.617, 0.647] | +0.087 [0.073, 0.101] | +0.108 [0.088, 0.127] |

- **f_none gives the same result** (RS n = 215, OT n = 249): capture rises +0.077/+0.084 for RS and +0.042/+0.056 for OT
  das4 at 65k/262k.
- **Flip rate:**
  - RS das4 feat_A_along_U goes 0.547 → 0.620 (Δ +0.073 [0.013, 0.133]).
  - RS das8 goes 0.680 → 0.727 (Δ +0.047 [−0.013, 0.107], CI includes 0).
  - OT flip is at ceiling (0.989–1.000) at every width.
- **Frozen E9 labels at the wider widths:**
  - RS das4/das8: still "illusion" and "feature-mediated" at 65k. The full − feat flip difference is 0.140 [0.073, 0.207] at
    das4 and 0.140 [0.080, 0.200] at das8, against 0.213 and 0.187 at 16k.
  - OT DAS: still "feature-mediated, no illusion" (flip ceiling).
  - Read from `team/geometry_out/e20_table_print.txt` (from `hw/tables/final/E9_illusion.csv` and `r5/tables/E20_illusion.csv`).

**Interpretation.**
- Going 4× wider (RS) moves about 30% of the DAS subspace's missing share (1 − capture) into the features: 0.246 → 0.173
  for das4.
- Going 4× and 16× wider (OT) moves about 9% and 13% of it: 0.468 → 0.426 → 0.409.
- The OT gain from 65k to 262k (+0.017 capture) is smaller than from 16k to 65k (+0.042), so the returns diminish.
- L0 is slightly lower at the wider widths (RS 74 → 64, OT 82 → 72 / 67). L0 therefore cannot explain the higher capture;
  if anything it works against it.

**Caveats.**
- OT NLDR overshoots 1 along DAS (full_along_U 1.85–1.98), so OT capture is a ratio of overshooting quantities.
- Capture is set-level (ratio of means).
- The gain may reflect generally lower reconstruction error at larger width, not anything task-specific. Without stored
  activations, this analysis cannot normalise by per-prompt fidelity on the task prompts.

### 2. More capture of the causal subspace does not mean a smaller total gap at matched L0, except OT 262k (strong)

`geometry_width_capture.py`: wide − 16k NLDR gap differences, f_pilot.

| cell | gap 16k | gap wide | Δ gap |
|---|---|---|---|
| RS feat_A_errB, 65k | 0.124 | 0.113 | −0.011 [−0.032, 0.011] |
| RS feat_A_meanPos, 65k | 0.117 | 0.115 | −0.002 [−0.020, 0.017] |
| OT feat_A_errB, 65k | 0.252 | 0.241 | −0.010 [−0.027, 0.005] |
| OT feat_A_errB, 262k | 0.252 | 0.147 | −0.105 [−0.122, −0.088] |
| OT feat_A_meanPos, 65k | 0.060 | 0.046 | −0.014 [−0.025, −0.003] |
| OT feat_A_meanPos, 262k | 0.060 | 0.022 | −0.038 [−0.052, −0.024] |

- In f_none, the RS errB gap shrinks slightly: −0.025 [−0.049, −0.003].
- **Flip-rate gaps:**
  - RS: 0.080 → 0.113 (Δ +0.033 [−0.020, 0.087]).
  - OT: 0.000 / 0.000 / 0.011 (flip-saturated).
- **Fact.** At 65k, both tasks gain DAS-subspace capture (finding 1) with no detectable change in the total errB gap.
- **Interpretation.** The error's contribution *inside* the DAS subspace shrinks with width. The parts of ε that matter
  outside that subspace (or interactions with it) do not.
  - In RS the in-subspace error contribution (full − feat along das4, NLDR 0.184 at 16k) is larger than the whole errB gap
    (0.124), so the contributions are not additive.
  - OT at 262k is the one cell where the gap itself falls, by 40%. E4 reported the same value, so this replicates.

### 3. Error-U repair changes with width in opposite directions for the two tasks (moderate-strong, uncorrected)

`geometry_width_repair_ci.py`, `width_repair_ci.csv`.

- **Measure.** Share of the NLDR gap closed by repair (x̂_A + ε_B + P_U(ε_A − ε_B)). Each width uses its own fitted error U.
- **Random-U null.** Over 20 draws per r, the p95 of the random share is ≤ 0.027 everywhere (`width_repair_nldr.csv`).
  Every non-random error-U share in the table below is far above it, except RS 65k pca r1 (−0.033).
- **Why NLDR.** The frozen flip-based rule cannot be met for RS repair: full_A flip is 1.0 and feat_A_errB 0.92/0.89, so the
  "+10 pts over random p95" bar exceeds 1. NLDR share is the informative metric here.

| cell | U | 16k | 65k | Δ (65k − 16k) | 262k | Δ (262k − 16k) |
|---|---|---|---|---|---|---|
| RS L18 | pca r1 | 0.225 [0.159, 0.288] | −0.033 [−0.058, −0.013] | −0.259 [−0.330, −0.184] | | |
| RS L18 | pca r8 | 0.730 | 0.628 | −0.102 [−0.188, −0.020] | | |
| RS L18 | answer_sep r1 / r8 | 0.444 / 0.756 | 0.316 / 0.579 | −0.128 / −0.177 (both CI < 0) | | |
| OT L12 | pca r1 | 0.519 [0.482, 0.553] | 0.642 | +0.123 [0.088, 0.159] | 0.608 | +0.089 [0.036, 0.152] |
| OT L12 | pca r8 | 0.547 | 0.734 | +0.187 [0.150, 0.221] | 0.663 | +0.115 [0.059, 0.181] |
| OT L12 | answer_sep r1 | 0.377 | 0.407 | +0.030 [0.004, 0.058] | 0.523 | +0.147 [0.097, 0.209] |

- **Facts.**
  - In RS the 65k error U is a worse repair basis than the 16k error U. Its top PC no longer repairs at all.
  - In OT, the wider SAEs' error U close more of a (same-size at 65k, smaller at 262k) gap.
  - The OT error-U illusion stays at every width: full_along_U is effective and feat_A_along_U is not at r1/r2.
- **Link to geometry (finding 4).**
  - The RS 65k top error PC has rotated: top cosine with 16k r1 is 0.402, while r4 is 0.831.
  - So the variance-dominant 65k axis is a different axis, and that axis does not carry the task variable.
  - This fits the earlier H6 null: the variance rank of an error axis does not predict its causal use.

### 4. The 16k error subspace persists in wider SAEs' errors but drifts gradually with width, with no convergence (strong geometry; trend moderate)

`geometry_width_subspace.py`, `width_subspace.csv`, `width_pairwise_*.csv`, `geometry_width_trend.py`. Monte Carlo Haar p99:
r1 0.054, r8 0.129, r32 0.246.

**OT L12, 10 wider SAEs (E4; E20 = 65k_l072 and 262k_l067):**
- Top cosine with the 16k U (pca_pooled): 0.57–0.76 (r1), 0.81–0.89 (r8), 0.87–0.93 (r32). answer_sep: 0.64–0.72, 0.78–0.85,
  0.87–0.92.
- Mean cos² (the share of the subspace that is shared): 0.47–0.64 at r8, 0.44–0.55 at r32 (Haar about 0.003 / 0.014).
- Capture of the 16k top PC in the wide r32 error PCA: 0.45–0.62 (Haar 0.014).

**RS L18, 65k:**
- r1 0.402, r2 0.586, r4 0.831, r32 0.904.
- Mean cos² r32 0.467.

**Rank-k variance share of the error (evr), 16k vs wide:**
- OT: evr1 0.215 vs 0.189–0.218; evr sum over 8 PCs 0.542 vs 0.513–0.556.
- RS 65k: 0.218 → 0.185 (evr1), 0.714 → 0.673 (sum of 8).
- IOI: 0.254 vs 0.235–0.258.
- So the concentration of the error in its top task directions barely changes with width (ranges are over the wide SAEs).

**Trend with width (Spearman over the wide SAEs, BH over 24 tests):**
- OT persistence falls with width:
  - pca mean cos² r32: ρ = −0.86, q = 0.008.
  - pca top r1: ρ = −0.74, q = 0.027.
  - answer_sep similar, ρ −0.76 to −0.86, q ≤ 0.02.
  - It does not fall with L0 (ρ −0.04 to 0.35, q ≥ 0.40).
- IOI (n = 5): width and L0 are collinear (ρ −0.9 to −1.0 vs +0.975), so the two cannot be separated.

**Pairwise matrix** (mean cos² r8, pca, `width_pairwise_*.csv`):
- IOI has a clean band structure. Adjacent widths agree about 0.64–0.68, two octaves apart 0.58–0.63, 16k vs 524k 0.50.
- Wide–wide pairs are no more similar than narrow–narrow pairs (262k–524k 0.64; 16k–32k 0.65).
- A descriptive OLS over all OT pairs (n = 55, not independent): similarity ≈ 0.72 − 0.031·|Δlog₂ width| − 0.072·|Δlog₂ L0|
  − 0.013·log₂(min width).
- So similarity falls with dictionary distance, and the minimum width contributes nothing: there is no convergence toward a
  "width-limit" error subspace.

**Interpretation.**
- The task-relevant error subspace is not absorbed by width up to 524k (32×). It rotates gradually as the dictionary changes,
  about −0.03 shared-subspace fraction per octave for OT at r8 under the OLS.

**Caveats.**
- The same prompts underlie every width, and prompt-driven variance could inflate all the cosines alike. It cannot create the
  band structure.
- There are no 65k+ decoders, so absorption at the atom level is untested.

### 5. E18 high rank: learned DAS sits partly in the error subspace for RS and OT, barely for PC; DAS fits of different ranks are nested (strong geometry)

`geometry_e18_rank.py`: `e18_das_vs_err.csv`, `e18_das_nesting.csv`. The DAS init is removed exactly as in
`make_final.das_learned`. The Haar p99 values come from a Monte Carlo for each (dim, m).

**Init capture** (share of DAS energy still in the shared seed-0 init; Haar k/2304):

| cell | das1 | das2 | das4 | das8 | das16 |
|---|---|---|---|---|---|
| RS | 0.41 | 0.54 | 0.63 | 0.64 | 0.76 |
| OT | 0.14 | 0.56 | 0.65 | 0.73 | 0.78 |
| PC12 | 0.24–0.45 across ranks | | | | |
| PC16 | 0.28–0.38 across ranks | | | | |

- At higher rank, more of each rotation is untouched init. That is consistent with 3 epochs moving a fixed budget of
  directions.

**Learned DAS vs error U** (top principal cosine; capture in the 64-dim error PCA, Haar 0.028):

| cell | das | acc | top vs pca r=k (p99) | top vs pca r64 (p99) | capture in pca64 |
|---|---|---|---|---|---|
| RS18 | das4 | 0.733 | 0.348 (0.098) | 0.618 (0.224) | 0.240 |
| RS18 | das8 | 0.890 | 0.450 (0.129) | 0.608 (0.239) | 0.211 |
| RS18 | das16 | 0.830 | 0.53 at m=16 (0.184) | 0.593 (0.263) | 0.167 |
| OT12 | das4 | 1.000 | 0.396 | 0.528 | 0.161 |
| OT12 | das8 | 1.000 | 0.466 | 0.552 | 0.123 |
| OT12 | das16 | 0.997 | 0.511 at m=16 | 0.596 | 0.113 |
| PC12 | das16 | 0.810 (first valid PC fit) | 0.282 at m=16 | 0.395 | 0.064 |
| PC16 | das16 | 0.780 | 0.336 at m=16 | 0.477 | 0.073 |

- The full principal-angle spectra are in the csv. For example, RS das16 vs pca16:
  0.53 0.49 0.47 0.41 0.38 0.35 0.28 0.25 0.20 …
  So several learned DAS directions have moderate overlap with the error PCs, not just one.
- PC das1–das8 are invalid (train accuracy 0.007–0.44). PC das16 is the first valid PC DAS (L12 0.81, L16 0.78), but only at
  k = 16.
- **Fact.**
  - The causal subspace of the people-cities task is mostly outside its 64-dim error PCA: capture 0.064, about 2× Haar.
  - For RS and OT it is 4–9× Haar.
- **Interpretation.** This matches the E10/E11 result that PC error-U repair closes about 0%. The PC binding variable is not
  where the PC error varies.
- **Nesting.**
  - Learned das_k lies largely inside learned das8/das16: top cosine 0.73–0.97 (RS/OT 0.82–0.96) and capture 0.35–0.83
    (Haar 0.003/0.007).
  - The inits for different k are independent random draws: init_k captured in init_16 is 0.003–0.008, about Haar. So the
    nesting is not an init artefact.
  - The DAS fits of different ranks therefore find one shared causal subspace.
- **Spectra limitation.** Only the first 8 evr values are saved: `harness/run.py` line 465 truncates `evr[:8]`. The requested
  spectra up to r = 64 cannot be recovered from the outputs. Rank-8 cumulative evr:

  | cell | RS18 | OT12 | PC12 | PC16 |
  |---|---|---|---|---|
  | cumulative evr (8 PCs) | 0.714 | 0.542 | 0.569 | 0.473 |
  | evr1 | 0.218 | 0.215 | 0.283 | 0.203 |

### 6. OT vs PC at L12: a partly shared error subspace that grows with rank, but disjoint causal subspaces (moderate)

`e18_err_overlap.csv`.

| comparison | U | r1 | r4 | r8 | r16 | r32 | r64 |
|---|---|---|---|---|---|---|---|
| OT12 vs PC12 | pca top cosine | 0.160 | 0.417 | 0.460 | 0.531 | 0.612 | 0.707 |
| | Haar p99 | 0.054 | 0.098 | 0.129 | 0.184 | 0.246 | 0.342 |
| | principal cosines above p99 | | 3 of 4 | 6 of 8 | 9 of 16 | 14 of 32 | 30 of 64 |
| | mean cos² | | | 0.065 | | | 0.144 (Haar 0.028) |
| OT12 vs PC12 | answer_sep top cosine | 0.10 | | | | | 0.69 |
| PC12 vs PC16 (cross-layer) | pca top cosine | 0.205 | | 0.306 | | | 0.589 |
| cross-task, cross-layer reference: OT12/PC12 vs RS18 | pca top cosine | | | below p99 at r ≤ 16 | | | 0.40–0.43 (just over p99) |

- **Learned DAS, OT12 vs PC12:** das8 0.150 (p99 0.129), das16 0.247 (p99 0.184). So only a marginal overlap.
- **Learned DAS, PC12 vs PC16:** 0.883 (das8) and 0.856 (das16), mean cos² about 0.5. The PC causal subspace is largely the
  same at L12 and L16, while its error PCA rotates (0.31 at r8).
- **Interpretation.**
  - OT and PC share a sizeable part of their L12 error variance: at r = 64, 30 of 64 principal cosines are above p99. This is
    plausibly entity/name-token variance common to both prompt formats.
  - Their causal subspaces barely overlap, which matches E11's ≤ 0.11 transfer.
- **Caveat.** PC12 das8 is invalid (0.437). Two fits sharing an init that keep 35–45% init energy could leave correlated
  learned parts, because the same init gives similar early gradients. A seed-varied PC DAS would test this; none was run.

### 7. Decoder atoms (16k L12/L18)

See the section "Decoder atoms" below. It is filled in from `geometry_atoms.py`.

---

## Alternative explanations and limits (all findings)

- **Feature capture is behavioural, set-level, NLDR-based.**
  - OT along-DAS NLDR overshoots 1, and OT flip is saturated.
  - A general fidelity gain at larger width could produce finding 1 without anything task-specific. A per-prompt FVU on the
    task prompts would be needed to separate the two; it is not stored.
- **The DAS rotations are 3-epoch fits that keep 41–78% (RS/OT) of their energy at init.** All DAS geometry uses the learned
  component, which is a partial-training object.
- **E23 (norm-matched noise controls) was not run.** None of the conclusions here depend on it, but the "error-U illusion" for
  OT has no norm-matched control.
- **No 65k+ decoders.** Width absorption is tested decoder-free: error-PCA persistence plus behavioural capture.
- **No evr beyond rank 8.**

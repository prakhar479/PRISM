# Causal subspaces: what E17, E18 and E20 say about the task variable (role: das)

Everything here is post hoc and exploratory. It uses only the saved directions (`ana_runs/E{9,17,18,20}/*/directions`), the
per-pair parquet files, the make_final tables in `r5/tables/` and the public 16k decoders (L12 l082, L18 l074) in `deep/hf`.
There is no 16k decoder for L16 on disk, so people-cities L16 has no atom analysis. E23 (norm-matched noise) was not run.
Unless stated otherwise, the test split uses f_pilot with n = 150 eligible pairs (OT L12: 89).

**Definitions**
- *Learned component*: a rotation with its own initialisation span projected out (harness `Rotation(d, k, seed)`).
- *Capture*: the mean squared projection of one subspace's orthonormal columns onto another subspace. The Haar level is m/2304.
- *OMP-m*: the share of a subspace's energy inside the span of the m best 16k decoder atoms, chosen greedily. Haar p99 from 40
  random draws per k, for 20 atoms: k=1 .117, k=2 .070, k=4 .046, k=8 .032, k=16 .023.
- *Residual gap closed by repair*: (NLDR_repair − NLDR_featA_errB) / (1 − NLDR_featA_errB). Its null is the same quantity at the
  p95 of 20 random U.

**Scripts** are in `team/scripts/`: das_lib.py, das_e17.py, das_atoms.py, das_e18.py, das_overlap.py, das_consensus_flip.py.
**Outputs and logs** are in `team/das_out/`.

**Tests and multiplicity**
- 13 McNemar tests on E17 training changes, BH-corrected.
- 12 McNemar tests on E20 width, BH-corrected.
- The Spearman correlations are uncorrected and descriptive (n = 7–20).

---

## F1. Running sum has a reproducible rank-1 causal direction, but the DAS optimiser does not isolate it (strong)

**FACT**
- **Validity.** In E17, no rank-1 or rank-2 fit reaches train interchange accuracy ≥ 0.5. That covers 8 rank-1 and 8 rank-2
  fits across seeds 0/1/2, 3/10/30 epochs and lr 1e-2/3e-2.
  - Rank 1: best 0.34 (das1_e10_s2).
  - Rank 2: best 0.473 (das2_e30).
  - The lr 3e-2 rank-1 fit collapsed: acc 0.000, 95% of its energy still at init, held-out flip 0.
- **Held-out effect is large anyway.** full_along_U flip, against a random-U p95 of 0.000 at every rank:

  | fit | flip | 95% CI |
  |---|---|---|
  | das1 (range over 7 fits, lr 1e-2) | 0.087 to 0.347 | best fit (das1_e10): [0.27, 0.43] |
  | das2 (range over 7 fits) | 0.31 to 0.50 | das2_e30: [0.42, 0.58] |

  - These fits would pass the "effective" margin (≥ p95 + 10 points, lower CI > p95) if they were not gated by validity.
  - Train accuracy tracks held-out flip: Spearman 0.95, n = 20 fits at lr 1e-2, uncorrected.
- **More training barely helps.** Paired exact McNemar on held-out flips, BH over 13 tests:
  - Only das2 → das2_e30 is significant: 0.347 → 0.500, 37 vs 14 discordant pairs, p_BH = 0.022. das2_e10 → e30 is too
    (p_BH = 0.022).
  - Every rank-1 change is n.s. (das1 → e10: +0.073, p_BH = 0.18; e10 → e30: −0.047).
  - Rank 4 and rank 8 are n.s. (das4 → e10: −0.027; das8 → e10: +0.033).
- **Learning curves** (only the first- and last-epoch mean losses are saved, so these are not full curves):
  - Rank 1: the loss does not fall. das1 goes 2.104 → 2.078 at 3 epochs, → 2.305 at 10 epochs and → 2.242 at 30 epochs. Seed 1:
    2.128 → 2.412 at 3 epochs.
  - Rank 2: it falls slowly, 2.088 → 1.525 at 30 epochs.
  - Rank 4 and 8: it falls clearly. das4 1.669 → 1.109 (3 epochs) → 0.730 (10 epochs); das8 1.280 → 0.329 (10 epochs).
  - The rotations never leave their random initialisation. Init capture for rank 1 is 0.30–0.52 even at 30 epochs (das1_e30:
    0.476). Ranks 2/4/8 stay at 0.50–0.67.
- **The learned component is seed-stable.** Learned-component top cosine between fits with *different* random initialisations
  (Haar p99: k=1 .052, k=2 .072):

  | rank | median | range | n pairs | raw cosine (median) |
  |---|---|---|---|---|
  | 1 | 0.74 | 0.60–0.86 | 16 | 0.43 |
  | 2 | 0.80 | 0.72–0.88 | 16 | — |
  | 4 | 0.87 | — | 2 | — |
  | 8 | 0.92 | — | 2 | — |

  My recomputation matches make_final's `cos_top_learned` to 5e-10 on all 36 rows.
- **Consensus direction.** Take the top singular vector of the 7 learned rank-1 directions (lr 1e-2):
  - It holds 81% of their summed energy. Each fit's cos² with it is 0.67–0.89.
  - Leave-one-seed-out consensus vs the other seeds: |cos| 0.77–0.88.
  - It lies inside the learned span of every rank-2/4/8 fit, including the seed-1 fits: cos² 0.68–0.82, against a Haar level of
    k/2304.
  - So the rank-1 direction is the leading component of every running-sum rotation.
- **Behavioural agreement.** Fits from different initialisations flip overlapping pairs: for rank 1, phi median 0.38
  (0.13–0.54).
- **Alignment with the consensus predicts outcome.** The share of a rotation lying on the consensus tracks held-out flip:
  - Rank 1: Spearman 0.95, n = 8. This includes the collapsed fit and is partly circular, because the consensus is built from
    the same fits.
  - Rank 2: Spearman 0.77, n = 8.
  - Script: das_consensus_flip.py.

**INTERPRETATION**
- The rank-1/2 "failures" in E9 are optimisation-limited fits, not evidence that no rank-1 variable exists.
- Every seed moves toward the same direction, but 30–50% of the patched energy stays on a random init axis. That dilutes the
  patch, since the effective component along the consensus is about cos² ≈ 0.3–0.58 of what is needed.
- The 0.5 validity gate is therefore measuring the optimiser as much as the representation.

**SPECULATION**
- A rank-1 patch along the consensus vector alone (saved as `das_out/rs18_das1_consensus.npy`) might exceed 0.5.
- That needs a GPU run, which was not done.

**Alternatives and caveats**
- The consensus could be the dominant first-step gradient direction, for example the mean of x_A − x_B, rather than "the
  variable". It is still a data-determined, reproducible causal direction.
- There are only 3 seeds.

## F2. Cross-machine, cross-run and cross-SAE reproducibility of DAS is bit-exact (strong; methodological)

**FACT**
- E9 (gnode084, commit 14e7e1f) and E17/E18/E20 (gnode058, commit 240e5f6) produce *bit-identical* rotations: `np.array_equal`
  is True, max |diff| 0. This holds for das1/2/4/8 at RS L18 and das1–8 at OT L12. The E17 logs show the rotations were
  retrained.
- Losses and train accuracies are identical.
- Per-pair held-out flips agree on 150/150 pairs for every rank.
- E20 trains DAS on the residual x, so its rotations are byte-identical to the 16k ones at 65k and 262k.
- The error-U is not identical across widths:
  - RS pca r1 top cosine 0.40. r8: top 0.86, mean 0.70.
  - OT 65k: r8 mean 0.75. OT 262k: r8 mean 0.66.

**INTERPRETATION**
- The DAS pipeline is deterministic, so DAS variability comes only from seed, schedule and learning rate (F1), not from the
  hardware.

## F3. The running-sum causal subspace is partly feature-made: a few recurring atoms plus a separate slice of the error subspace (moderate-strong)

**FACT: atoms** (learned components, L18, das_atoms.py)
- **20 atoms explain** (Haar p99 in brackets):
  - Rank 1: 0.37–0.49 [0.117].
  - Rank 2: 0.32–0.47 [0.070].
  - Rank 4: 0.33–0.37 [0.046].
  - Rank 8: 0.29–0.33 [0.032].
  - Rank 16: 0.19 [0.023].
- **Nearest single atom |cos|**: 0.25–0.37 for rank 1, and up to 0.76 for rank 8 (atom 14691). The null max is ≈ 0.10–0.14.
- **Recurring atoms recur across seeds**, which rules out the shared init as the cause:
  - All 7 previously named atoms (15376, 7686, 8352, 11219, 13950, 411, 14691) sit in the top 10 of 16k for das4, das4_e10, das8,
    das8_e10, das8_s1 and das16. The one exception is 14691 in das4_s1, at rank 85.
  - 14691 has cos² 0.48–0.58 with every rank-8 fit (s0, s1, e10).
  - Rank-1 fits are dominated by 411 and 8352 instead: 411 is top-3 in 7/7 lr-1e-2 rank-1 fits, cos² 0.05–0.13. 4092 and 7479 are
    frequent. 14691 is outside the top 700 atoms in all 7; 15376 is top-5 in only 2/7 (e30, s1).
  - The consensus vector: 20 atoms 0.46; top atoms 411 (cos² 0.10), 4092, 8352, 7479, 11219.
- **Training does not make rotations more feature-made.** Learned OMP-20:
  - das1: 0.44 → 0.49 (e10) → 0.46 (e30).
  - das2: 0.43 → 0.40 → 0.32.
  - das4: 0.33 → 0.37.
  - das8: 0.30 → 0.29.
  - On the full U, OMP-20 rises only where the init share falls (das1_e10: 0.35 vs 0.27).

**FACT: error subspace** (same run's pca_pooled; Haar 32/2304 = 0.014)
- The learned component in pca_pooled 32: rank 1 0.21–0.31, rank 2 0.18–0.25, rank 4/8 0.15–0.19.
- In the r = k error PCs only: 0.00–0.09.
- **Feature part and error part are mostly different directions** (das_overlap.py). For RS fits, joint capture in [atoms20 ∪
  pca32] is close to the sum of the parts:
  - das4: 0.333 + 0.177 = 0.51 vs joint 0.47.
  - das8: 0.295 + 0.166 = 0.46 vs joint 0.42.
  - The atoms20 span itself sits only 0.07–0.10 in pca32 (Haar 0.014).
  - About 50% of each subspace is in neither.
- **Behavioural ratios.** NLDR feat_A_along_U / full_along_U is 0.61–0.65 at rank 1/2 and 0.75–0.79 at rank 4–16 (E18). The
  illusion gap (full − feat) is 0.033–0.287, with lower CI > 0, for every rank 1–8 E17 fit except the collapsed lr3e-2 one. For
  das16 (E18) it is 0.073, lower CI −0.007.

**INTERPRETATION**
- The running-sum variable at L18 is low-dimensional.
- Its causal subspace is roughly one third "a handful of SAE features" (411/8352 for the leading axis; 14691/15376/7686/11219/
  13950 for the higher axes), plus a smaller, separate piece in the SAE error subspace.
- The error piece is what full − feat measures. This matches E9 flagging both "feature-mediated" and "illusion".

**Caveats**
- OMP greedy selection overstates "few atoms" relative to an exact search only slightly, and the null uses the same OMP.
- Atom labels are not checked (Neuronpedia not consulted).

## F4. E18 rank curves: three tasks, three geometries (strong for RS/PC, moderate for OT)

FACT (E18 f_pilot, das_e18.py → `das_out/e18_rank.csv`).

**Running sum L18**
- DAS becomes valid at rank 4 (acc 0.73).
- Error-U (pca / answer_sep) full_along_U becomes effective only at r = 32: 0.153 / 0.173, p95 0.000. At r = 64: 0.247 / 0.267.
  r ≤ 16 is ≤ 0.08.
- Repair (error only) still closes the residual NLDR gap. featA_errB is 0.876. Share of gap closed (null p95 ≤ 0.016 at r ≤ 8):
  - pca: r8 0.73, r64 0.92.
  - answer_sep: r4 0.72.
  - DAS U: das2 0.79, das4 0.90, das8 1.06.
- So the error along the *DAS* subspace repairs better than the top error PCs of equal rank.

**Object tracking L12**
- DAS is valid at rank 1 (0.97) with flip 1.00, and feat_A_along_U 0.99 (feature-mediated, no illusion).
- Error PC1 alone is also effective: full 0.91 vs p95 0.045, but its feature component flips only 0.11, an illusion gap of 0.80
  (lower CI 0.71).
- The learned DAS-1 direction sits 0.13 in error PC1 (Haar 0.0004) and 0.28 in pca64. 20 atoms explain 0.27 [null 0.117].
- Higher-rank OT DAS is diffuse: 20 atoms explain 0.09–0.19.

**People-cities L12 and L16**
- Error-U is *never* effective: full_along_U = 0.000 on 150/150 pairs at every r up to 64, for both estimators and both layers.
- Error-U repair closes ≤ 0.043 (L12) and ≤ 0.16 (L16) of the residual gap, against a null p95 of 0.037 (L12) and 0.031 (L16):
  L12 is null and L16 is small.
- DAS is invalid at rank ≤ 8:

  | layer | rank 1 | rank 2 | rank 4 | rank 8 | rank 16 |
  |---|---|---|---|---|---|
  | L12 train acc | 0.007 | 0.017 | 0.143 | 0.437 | **0.810** (valid) |
  | L16 train acc | 0.007 | 0.037 | 0.047 | 0.327 | **0.780** (valid) |

- DAS16 results:
  - full_along_U 0.793 (L12) and 0.827 (L16).
  - feat_A_along_U 0.420 / 0.273, so both "feature-mediated" and "illusion" flags are set. Gap 0.373 [lo 0.30] and 0.553 [lo
    0.47].
  - Repair with the DAS16 U closes 0.60 (L12) and 0.58 (L16) of the residual gap, against a null p95 of 0.008 / 0.009.
  - Its learned subspace lies only 0.064 in pca64 (Haar 0.028). 20 atoms explain 0.185 [0.023]; the top atoms (205, 10931, 4339)
    each have cos² of about 0.2.

**INTERPRETATION**
- **Running sum**: a low-dimensional, mostly feature-made variable, with a small error-carried slice that lies along the
  causal subspace rather than along the top error-variance axes.
- **Object tracking**: the variable (or, given the prefix-fallback confound in deep_analysis C, the "erase B's update"
  direction) is one-dimensional and *redundantly* accessible. The features carry it along the DAS axis, and the error carries it
  along error PC1. The latter explains why error-U "works" for OT and not for the other tasks.
- **People-cities**: binding is high-rank and distributed (rank 16 needed). The part of the SAE error that matters lies in this
  causal subspace but carries almost none of the error *variance*. That is why every variance- or answer-based error estimator
  fails, while repair along DAS16 recovers 60% of the shortfall. This extends geometry.md H6 ("the variance-dominant error axis
  is not the causal one") from correlation to intervention.

**Caveats**
- PC DAS16 validity is modest (0.78–0.81) at 3 epochs.
- The OT flip ceiling is reached at feat_A_errB (1.00), so OT conclusions rely on NLDR, which DAS overshoots (> 1.5).

## F5. E20: wider SAEs leave the DAS subspace unchanged and shift little of it from error to features (weak)

**FACT**
- With the identical rotation, RS feat_A_along_U goes from 16k to 65k as follows:
  - das4: 0.547 → 0.620 (17 vs 6 discordant pairs, p = 0.035 uncorrected, p_BH = 0.42).
  - das8: 0.680 → 0.727 (p = 0.21).
  - full_along_U is identical (0 discordant pairs).
  - The illusion gap shrinks from 0.213 to 0.140 (das4) and from 0.187 to 0.140 (das8), but is still flagged.
- OT is unchanged at the ceiling for 65k and 262k.
- The learned DAS share inside each SAE's own error pca32 does not fall with width:
  - RS das4: 0.177 at 16k vs 0.222 at 65k.
  - OT das4: 0.117 at 16k, 0.110 at 65k, 0.125 at 262k.

**INTERPRETATION**
- The causal subspace is a property of the model, not of the SAE.
- A 4× wider SAE at matched L0 may move a little of the RS variable from ε to x̂, but this does not survive correction.
- The error slice of the causal subspace persists, which is consistent with geometry.md finding 2.

## Nulls and failures (as findings)
- No E17 configuration produced a valid rank-1/2 fit. lr 3e-2 destroyed the rank-1 fit (acc 0).
- More epochs did not change held-out rank-1 flips (all rank-1 McNemar tests n.s. after BH).
- Training did not increase feature-made share (F3).
- Seed 1 is consistently worse at every rank (das1_s1 0.087, das4_s1 0.593, das8_s1 0.733 vs seed 0: 0.273 / 0.760 / 0.867). Its
  rotations also align least with the consensus. An n of 3 seeds cannot separate an unlucky init from a real basin.
- People-cities error-U: zero flips up to r = 64 at two layers.

## Paper-ready one-liners (exploratory)
- "Rank-1/2 DAS failures on the running sum are optimisation-limited: rotations from different seeds converge on the same
  direction (learned-component cosine 0.60–0.86 vs Haar 0.05), but retain 30–50% of their random initialisation even after 30
  epochs."
- "The running-sum causal subspace is about one third spanned by ≤ 20 SAE atoms that recur across seeds (14691, 15376, 8352,
  411, …). A separate slice (about one fifth) lies in the top-32 error PCs. People-cities requires rank 16, and its causally
  relevant error lies outside the top-64 error PCs, yet carries 60% of the residual shortfall."

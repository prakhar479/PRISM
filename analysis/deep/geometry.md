# Geometry of the task-relevant SAE error: width, depth, task, language, DAS (post hoc, exploratory)

All analyses are post hoc and exploratory, from existing run outputs on DEV (no GPU, no new runs). The one exception is that the
two public Gemma Scope 16k decoders (L12 l082, L18 l074; 577 MB) were downloaded at the pinned revision into
`deep/hf/` for H1.

Definitions:
- "Error directions" are `pca_pooled` (centered PCA of last-position SAE errors of A and B train rows) and `answer_sep`
  (answer-supervised), as saved in `ana_runs/E*/*/directions/*.npz`.
- `top` is the top principal cosine between two subspaces. `capture(A in B)` is the mean squared projection of A's
  orthonormal columns onto span(B).
- Null for top cosine: independent Haar subspaces of R^2304; p99 is 0.052 (k=1), 0.097 (k=4), 0.131 (k=8). Null for capture
  is m/2304 (m=32: 0.014).
- Scripts are in `deep/scripts/geo_*.py`, outputs in `deep/geo_*.csv`.

## Findings, ranked by strength

### 1. Task-relevant error directions are task-specific and stable across templates and languages (strong)
- **Across tasks:** at the same layer (E8, 4 tasks × layers 8–20, 66 pairs), pca r=1 top cosine has median 0.016 and max
  0.120. Only 8% of pairs exceed the Haar p99 (answer_sep: median 0.026, 18%). At r=4 the median is 0.115, with 68% above p99,
  so there is a small shared component.
- **Within task, across template families** (E3 fit-pilot vs fit-fresh):
  - Running sum: r1 0.78 (pca) / 0.90 (answer_sep); r2 0.95.
  - Object tracking: r1 0.08 / 0.00 (unstable top direction), but r4 0.71 / 0.45. The subspace is stable, not its first axis.
- **Across languages** (E14, native fit vs English fit):
  - Running sum: French 0.86, Hindi 0.92, Hinglish 0.89 at r1 (pca). That matches the English template-change reference
    (0.87).
  - People-cities: French 0.57, but Hindi 0.02, romanized Hindi 0.00 and Hinglish 0.11 at r1 (r8 ≤ 0.31), against an English
    cross-template reference of 0.996.
  - So the running-sum error subspace is language-invariant, and the people-cities subspace changes in the Hindi family.
- **Paper claim:** the SAE-error subspace that carries task state is task-specific (cross-task cosine ≈ Haar) yet reproducible
  across template families and, for running sum, across French, Hindi and Hinglish.
- **Caveats:**
  - Directions are fitted on last-position errors, which share token context within a task. Cross-task orthogonality argues
    against a generic "last-token" direction, but within-task stability may partly reflect a shared prompt format.
  - The people-cities Hindi family has Devanagari prompts with Latin answers, a different token context.

### 2. The task error direction is off-dictionary at 16k and persists in the errors of 32×-wider SAEs (strong)
- **Off-dictionary (H1):** the 16k decoders are L12 l082 and L18 l074. For every error direction (4 E8 tasks + people-cities,
  first two columns of pca and answer_sep):
  - The nearest atom has |cos| 0.10–0.25 (null p50 0.085, p99 0.10).
  - The best 20 atoms (OMP) explain 14–30% of its energy (null p99 0.12).
  - The encoder sees it no better: max |cos| with W_enc columns 0.09–0.20.
  - So it is not a missing or under-activated single feature; it is largely off-dictionary.
- **Width persistence (decoder-free; E4: same prompts, SAEs 16k → 524k at bracketing L0):**
  - The 16k object-tracking L12 direction stays in the wider SAEs' error PCA, with r1 top cosine 0.57–0.76 at every width
    (32k–524k), r8 0.81–0.89. The Haar p99 is 0.052.
  - Its variance share stays at 0.19–0.22 of the error. IOI is the same (0.71–0.85). Running sum L18 65k: r1 0.40, r4 0.83.
  - A dictionary up to 32× larger does not absorb the direction. This fits the E4 gap persisting
    (NLDR gap 0.13–0.25 at every width).
  - Spearman(gap, persistence) = 0.49 (p = 0.15, n = 10): suggestive only.
- **Paper claim:** the task-relevant dark matter is an off-dictionary direction (nearest of 16k atoms |cos| ≤ 0.25) that
  survives in the error of SAEs up to 524k latents.
- **Caveats:**
  - The 65k/262k decoders were not downloaded, so "not absorbed" rests on the error-PCA persistence, not on atom cosines at
    the wider widths.
  - The same prompts underlie every width, so shared prompt-driven variance can inflate the persistence.
- Consistent with published reports that SAE error is partly predictable and persists with scale (e.g. Engels et al. 2024,
  "Decomposing the dark matter of SAEs"). Verify this citation before using it.

### 3. Methodological: DAS rotations remain largely at their shared initialisation (strong; affects E9/E16/E17 geometry)
- Every DAS fit starts from the same seed-0 orthonormal initialisation (harness/model/das.py `Rotation`, seed = config seed).
  After training (3 epochs, lr 1e-2), each rotation keeps 14–73% of its energy in its initial span; the chance level is
  k/2304 ≈ 0.002. Running sum L18: das4 0.63, das8 0.64; object tracking L12: das8 0.73.
- As a result, raw DAS-to-DAS comparisons are init artefacts:
  - Running-sum das4 across L12–L20: top cosine 0.77–0.99 raw, 0.05–0.85 learned-only.
  - Object tracking vs people-cities das4: 0.85 raw, 0.08 learned-only.
- DAS vs error-U cosines (E9 `cos_top_das`) are not inflated: the init is random with respect to U. On the learned component
  they are similar (0.24–0.45 vs 0.15–0.42), so E9's "partial" label stands.
- **Action:**
  - Any figure or claim comparing DAS subspaces across layers, tasks or seeds must use the learned component (projection off
    the init span) or seed-varied inits.
  - E17's s1/s2 seeds change the init, so E17 stability must be computed after removing each run's own init. make_final's
    `das_stability` does not do this yet.
  - Also reconsider "das1/2 failed": with most of the energy at init, three epochs may simply be too short (E17 tests 10/30).

### 4. The running-sum causal (DAS) subspace at L18 is spanned by a few SAE features; error directions are not (moderate-strong)
- **Running sum L18 (valid fits):** on the learned DAS directions, the nearest single atom has |cos| 0.32–0.60, and 20 atoms
  explain 44–59% of the energy (null p99 0.12).
- **Recurrent atoms:** 15376, 7686, 8352, 11219, 13950, 411, plus 14691 for das8 (cos² 0.44). Each takes 0.13–0.44 of the
  learned span (null best atom in a random 2-dim span: p99 0.014). The top-5 atom sets of das4 and das8 share 4 atoms.
- **Object tracking L12 DAS:** diffuse. The nearest atom has |cos| 0.16–0.27, 20 atoms explain 22–29%, and the top atoms
  change across ranks.
- **Error directions at the same layers:** nearest atom 0.11–0.22 (finding 2).
- **Paper claim:** in running sum the causal variable lives in a handful of SAE features (geometric support for E9's
  "feature-mediated" flag), while the error directions that repair the residual shortfall are off-dictionary.
- Atom ids are given for Neuronpedia labelling (gemma-2-2b, 18-gemmascope-res-16k).
- **Caveat:** the learned component of a partly trained rotation (finding 3); das1/das2 fits are invalid.

### 5. Depth: the learned DAS subspace enters the error subspace at L16–L20 (moderate, n = 5 layers)
- Running sum, capture of the learned DAS in the 32-dim error PCA (Haar 0.014):

  | layer | capture |
  |---|---|
  | L12 | 0.02 |
  | L14 | 0.02–0.03 |
  | L16 | 0.13–0.18 |
  | L18 | 0.18–0.30 |
  | L20 | 0.19–0.24 |

- The E16/E9 NLDR gap is 6.6 / 5.3 / 5.4 / 14.2 / 13.7 at the same layers.
- The causal variable starts to overlap the SAE error one layer band before the shortfall rises. Descriptive only.
- The learned DAS rotates gradually across layers: adjacent pairs 0.47–0.85; L12 vs L18 0.05.

### 6. Error directions rotate with depth: half-life about 1.5–2 layers (moderate)
- Mean r1 top cosine by layer distance (E8, L8–16):

  | distance | cosine |
  |---|---|
  | 1 | 0.41–0.58 |
  | 2 | 0.23–0.45 |
  | 4 | 0.09–0.31 |
  | 8 | 0.01–0.16 |

- IOI and running sum decay slower than object tracking and var_binding. E11 confirms this for object tracking and
  people-cities.
- Running sum shows a dip at L16→L18 (r1 0.10 vs about 0.39 expected at distance 2; r4 0.49 is normal), where the gap emerges.
- There is no single direction carried through depth; each layer's error axis is a new, partly overlapping one.

### 7. Object tracking and people-cities share part of their error subspace, but not causally (weak/moderate)
- Same layer L8–16: pca r1 0.10–0.24 and r4 0.27–0.42, against a cross-task median of 0.115 at r4.
- E11 transfer of U between the two tasks closes ≤ 0.11 of the gap, so the shared part is not interchangeable causally. It may
  be entity-name / position variance.

## Nulls (report them)
- **H6:** error concentration does not predict repair.
  - Variance share of the first error PC vs share of gap closed by same-task repair (E8, NLDR, f_pilot): ρ = 0.15 (n = 36),
    and 0.02 among cells with gap ≥ 5 points. answer_sep r1 among gap ≥ 5: ρ = 0.51, p = 0.06.
  - People-cities error is as concentrated as object tracking (variance share 0.20–0.28; answer-aligned 0.19 at L12), yet
    repair closes about 0%. The variance-dominant error axis is not the causally used one.
- **H2:** width vs L0 (release-2 analysis E, OLS on n = 11): R² 0.28; −0.011 NLDR per doubling of width, −0.027 per doubling
  of L0. Underpowered and non-monotone; no width effect separable from L0.
- **H5 for people-cities:** the DAS fits are invalid (train accuracy 0.007–0.14), so there is no geometric conclusion.

## Scripts and outputs
| script | output |
|---|---|
| geo_lib.py | shared helpers |
| geo_layers_tasks.py | geo_crosslayer_E8.csv, geo_crosstask_E8.csv |
| geo_decay.py | geo_crosslayer_E11.csv |
| geo_templates_langs.py | printed |
| geo_width.py | geo_width_E4.csv |
| geo_das.py | geo_das_capture.csv |
| geo_das_init.py | init artefact check |
| geo_das_learned.py | geo_das_learned.csv |
| geo_spectrum.py | geo_spectrum_E8.csv |
| geo_atoms.py | geo_atoms.csv |
| geo_atom_ids.py | atom ids |
| geo_crosstask_summary.py | printed |

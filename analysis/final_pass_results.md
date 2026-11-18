# Final pass (queue complete-server) results — DEV analysis, 2026-11-18

Source: branch `results/final-run-20261117-2-1-g14e7e1f` (commit 21dd985), gnode084, 4 GPUs, 2026-11-17/18.
Tables: `tables/final/*.csv` (make_e12, make_e13, make_final with n_boot = 10 000). Decision rules: PREREGISTRATION_ADDENDUM.md
(frozen 2026-11-16); constants in harness/final.py. Numbers are copied from the tables; nothing here is interpreted beyond the
frozen rules except where marked *exploratory note*.

## Provenance and integrity

- 31 runs published, all `complete`, none smoke/tiny, all on one commit `14e7e1f`, tree clean (inventory_runs.py).
- `14e7e1f` = tag `run-20261117-2` (6afd6af) + one server commit changing only the `#SBATCH --nodelist/--gres` lines of
  scripts/slurm_final.sbatch. Harness code identical to the tag; ingest needs `--allow-untagged` (to be logged in DECISIONS.md).
- Hook checks: max |dD| = 0 in every run. CellSkips (E14, < 20 train pairs after f_pilot): OT hi, OT hinglish, OT fr, RS hi_latn.
- Reproducibility vs release 2 on the same pairs: RS L18 feat_A_errB NLDR 0.876 (f_pilot) vs 0.874; OT L12 0.748 vs 0.741;
  feat_A_meanPos flip (f_A_correct, RS) 87.7 vs 88.2 (one pair). E2r (seed 1) noise controls within 2 points of seed 0.

## E9 — illusion checks + DAS (confirmatory: RS L18; OT L12 exploratory)

RS L18, f_pilot, n = 150 (same pattern for f_none 215 and f_A_correct 208):

| estimator | r | full_along_U flip | feat_A_along_U flip | effective | illusion | feature-mediated | DAS fit valid | cos(U, DAS_k) |
|---|---|---|---|---|---|---|---|---|
| pca_pooled | 1/2/4/8 | 0.01/0.01/0.01/0.06 | 0.00 | no | no | no | – | 0.15–0.40 (partial) |
| answer_sep | 1/2/4/8 | 0.03/0.03/0.07/0.06 | ≤0.01 | no | no | no | – | 0.26–0.39 (partial) |
| das1 / das2 | 1 / 2 | 0.27 / 0.35 | 0.12 / 0.20 | – | – | – | **no** (train acc 0.25 / 0.33) | – |
| das4 | 4 | 0.76 [0.69, 0.83] | 0.55 [0.47, 0.63] | yes | yes (diff 0.21, lo 0.15) | yes | yes (0.73) | – |
| das8 | 8 | 0.87 [0.81, 0.92] | 0.68 [0.60, 0.75] | yes | yes (diff 0.19, lo 0.11) | yes | yes (0.89) | – |

Random-U p95 for full_along_U = 0.00 (20 seeds). Repair (features + U) is never effective: the features-only baseline already
flips 0.92 and random-U repair p95 is 0.92–0.93.

- By the frozen rules: the error-derived subspaces (pca_pooled, answer_sep) are **not effective** at any rank ≤ 8 for RS L18.
  DAS-4 and DAS-8 are effective, and both the `illusion` and the `feature-mediated` flags are true (the addendum's two
  definitions are not mutually exclusive; report both). U–DAS alignment is `partial` at every rank (above the random p99, below 0.5).
- DAS-1/2 are failed fits (they are not evidence). *Exploratory note:* their training loss barely moved (2.10 → 2.08 for k = 1;
  3 epochs, lr 1e-2), so this may be an optimisation failure rather than the absence of a rank-1/2 variable.
- OT L12 (exploratory): pca_pooled/answer_sep full_along_U 0.87–0.96 vs feat_A_along_U 0.09–0.20 → `illusion` at every
  rank (diff 0.75–0.85, lo ≥ 0.66); feature-mediated too at r ≥ 4 (pca) / r = 8 (answer_sep). DAS 1–8: valid fits (0.97–1.00),
  effective, feature-mediated, not illusion (feat ≈ full ≈ 0.96–1.00).
- E9b (RS L18, U/DAS fitted on pilot templates, evaluated on 3 fresh families: plus 205, story 194, symbolic 156 pairs):
  the same pattern in every family (error-U never effective; DAS-4/8 effective, illusion + feature-mediated; DAS-1/2 failed fits).

## E10 — steering, Q2 (confirmatory: RS L18)

`q2_both` is false in every row (E10: 243 rows; E10b: 243 rows).
- RS L18: steer flip ≤ 0.03 for pca/answer_sep at ×1/×2/×4; DAS 0.03 (×1) to 0.20 (×4); matched-norm random p95 ≤ 0.01.
  The Q2 criterion (≥ 0.50) is not met. E10b (fresh families): the same, max 0.22.
- OT L12 (exploratory): steering flips 0.94–1.00 at ×1, but random directions of matched norm also flip (p95 0.58–0.73 at ×1,
  up to 0.90 at ×2): the random-p95 ≤ 5 % arm fails. Consistent with release 2's OT B-run fragility note.
- People-cities L12 (exploratory): 0 flips for every estimator and scale, and for random; NLDR moves ≤ 0.08 (DAS ×4).

## E16 — RS layer sweep (exploratory)

f_A_correct, n = 208. Gap (1 − NLDR of feat_A_errB): L12 6.6, L14 5.3, L16 5.4, L18 14.2 (E9), L20 13.7 points.
Error-U (pca, answer_sep, r ≤ 4) not effective at any layer (full_along_U ≤ 0.03). DAS-4 valid and effective at every layer
(full_along_U 0.85 / 0.80 / 0.63 / 0.72 at L12/14/16/20); DAS-1/2 failed fits everywhere.

## E11 — people-cities gap and specificity, L8–16 (exploratory)

f_pilot, n = 150, NLDR metric:
- People-cities: gap rises with depth, 13.1 (L8) → 23.7 (L12) → 33.5 (L15) → 32.3 (L16) points. Same-task U (r = 1–4) closes
  0–6 % of it (the best: answer_sep r = 4 at L16, 0.06 [0.05, 0.06]). Flip-rate gap 2.7 → 18.7 points; share closed ≈ 0.
- OT: NLDR gap peaks at L12 (25.2 points; 10.1 at L11, 9.6 at L13, ≤ 5 elsewhere); same-task U closes 0.36–0.64 at L11–13
  with other-task / random controls ≤ 0.11 (diff lower bounds 0.23–0.50). Shares at L14–16 are unstable (gap ≈ 0).

## E12 — M1, SAE fidelity per language

M1 (mean-ablation CE-recovered upper CI < English lower CI, same layer) is **supported for every non-English set at every layer**
(L6, L12, L18, L24), French included (its deficit is small: 0.3–2.6 points). Parity check: en_pilot100 = 0.993 / 0.986 / 0.979 /
0.963 (L6/12/18/24). Zero ablation agrees except fr_web (L6, L12, L18) and ta_latn_dakshina (L12), which are not below English.

| set | L12 | L18 | | set | L12 | L18 |
|---|---|---|---|---|---|---|
| en_pile | 0.986 | 0.979 | | hi_deva_web | 0.950 | 0.932 |
| fr_web | 0.981 | 0.971 | | hinglish_phinc | 0.966 | 0.949 |
| hi_deva_dakshina | 0.932 | 0.912 | | hi_latn_dakshina | 0.966 | 0.927 |
| ta_taml_dakshina | 0.897 | 0.869 | | ta_latn_dakshina | 0.965 | 0.886 |
| bn_beng_dakshina | 0.905 | 0.879 | | bn_latn_dakshina | 0.939 | 0.926 |
| ta_web | 0.912 | 0.885 | | bn_web | 0.909 | 0.889 |

Script vs language (same Dakshina sentences): native script is lower than romanized for hi, ta and bn at L12 and L18
(e.g. ta L12: −6.9 points). *Exploratory note:* across the 12 sets, CE-recovered falls with tokens-per-word (Spearman −0.93 at
L12, −0.97 at L18); fertility, script and genre are confounded (Addendum A).

## E13 — capability gates (test split governs)

| task | en | hi | hi_latn | hinglish | fr |
|---|---|---|---|---|---|
| OT acc / Gate A / B | 0.927 ✓ | 0.309 ✗ ✗ | 0.656 ✗(keep 0.29) ✗ | 0.264 ✗ ✗ | 0.632 ✗(keep 0.01) ✗ |
| RS acc / Gate A / B | 0.916 ✓ | 0.536 ✗ ✗ | 0.324 ✗ ✗ | 0.678 ✓ ✗ (−23.8) | 0.802 ✓ ✗ (−11.3) |
| PC acc / Gate A / B | 0.958 ✓ | 0.780 ✓ ✗ (−17.8) | 0.376 ✗ ✗ | 0.713 ✓ ✗ (−24.4) | 0.996 ✓ ✓ |

Only people-cities French passes both gates. The train split agrees except RS hi (train 0.636 passes Gate A; test 0.536 fails).

## E14 — M2 / M3

Readable rows (both gates + validated): pc_fr only (plus English references).
- M2 (pc_fr): flip gap 2.7 [0.7, 5.3] vs English 7.3 [3.3, 11.8]; NLDR gap 24.1 vs 23.7 → not larger than English: **not supported**.
- M3: `m3_usable` is false in every row. pc_fr is readable but its native share's lower bound is 0 (gap 2.7 points); every other
  language is not readable. **M3 cannot be evaluated** under the frozen rule.
- *Exploratory note (not readable rows):* Hindi people-cities flip gaps are large (pc_hi 47 %, pc_hinglish 42 %, pc_hi_latn 58 %
  vs English 7 %), while NLDR gaps are the same as English (22–24 %): the flip gap tracks the lower clean accuracy / margins.

## Answers to the questions as preregistered

| question | answer |
|---|---|
| RQ1 RS L18 (release 2) | GO; reproduced here (shortfall 12.3 points f_A_correct vs 11.8) |
| E9 RS L18: does the error subspace carry the missing variable? | Error-derived U: not effective (r ≤ 8). DAS-4/8: effective; illusion + feature-mediated flags both true; alignment partial |
| E10 Q2 RS L18 | Not met (max steer flip 0.20 vs 0.50 needed) |
| M1 | Supported for all languages and layers (CI rule) |
| M2 | Testable only for pc_fr: not supported |
| M3 | Not evaluable (no usable row) |

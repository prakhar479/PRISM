# Claims audit of `hw/analysis/team_analysis_2026-11-19.md` (role: claims auditor), 2026-11-19

Every number in the sections Provenance, Preregistered answers, C1–C8, Other robust patterns and Corrections was recomputed
from raw data (per_pair.parquet, directions npz/json, E12/E21 per-window npz, prompt files, run manifests/logs/summaries, the
16k L18 decoder in `deep/hf`) with **new code** (`team/scripts/audit_*.py`). No script under `team/scripts` (other than `audit_*`),
`deep/scripts` or `hw/analysis/*/scripts` was imported or run; they were read only for definitions. No harness module was
imported either: metrics (flip, ratio-of-means NLDR, pair bootstrap) and the DAS initial rotation (QR of a seeded
`torch.randn(d,k)` + torch's orthogonal parametrisation, as in `harness/model/das.py Rotation`) were re-implemented in
`audit_lib.py` (init check: max |param − QR| = 1.2e-7). Bootstraps: B = 2000 (pairs) unless stated, 500–1000 for windows.
OMP_NUM_THREADS=2. The multilingual window-level part (C7 text fidelity + Corrections row 1) was done by a sub-audit in
`audit_ml.py` and merged here unchanged. Logs/outputs: `team/audit_out/`.

**Verdict counts (88 numeric claims or claim parts; 73 audited here + 15 C7 window rows from the sub-audit):** match 78 · mismatch 5 · not reproducible 2 · not audited 3.
"match" means the stated value is reproduced to the stated precision under a definition that I could pin down; where the
match needs a specific pair set, filter or definition, the note says which. Wording flags (claim stronger than the evidence)
are listed separately at the end; they do not change the match count.

| script | covers | log |
|---|---|---|
| `audit_rq1.py` | RQ1 gate, errB shortfall, margin split, families | `audit_out/rq1.log` |
| `audit_das_geom.py` | E17 validity, init capture, learned-component cosines, atoms, OMP, error slice | `das_geom.log`, `das_atoms.csv` |
| `audit_das_flips.py` | E17 flips, Spearman, seed McNemar+BH; E18 RS/OT/PC rank curves, repair, PC | `das_flips.log`, `e17_*.csv` |
| `audit_steer.py` | C3 (Δ distribution, one-sided flips, trajectories, E10b, specificity), C4 (OT fallback, margin ρ, noise) | `steer.log` |
| `audit_width.py` | C6 width (E9 vs E20, paired) and the lead's capture-median claim | `width.log`, `capture.log`, `capture_all.csv` |
| `audit_misc.py` | C1 filters, C5 city eta², Athens, Tokyo; C7 E22 per-pair; C8; start-0; E22 replication; profiles; graded resample | `misc1-3.log`, `gain_slopes.csv` |
| `audit_prov.py` | logit bit-identity across runs/nodes, laptop vs cluster, DAS rotation identity | `prov.log` |
| `audit_extra.py` | donor-order bug share, cost vs patchability, E10 Q2 | `extra.log` |
| `audit_m1.py` | preregistered M1 44/44 | `m1.log` |
| `audit_ml.py` (sub-audit) | C7 E12/E21 windows, Corrections row 1 | `ml_*.csv`, `ml_regress.log`, `team/audit_ml_ledger.txt` |

Abbreviations: f_pilot/f_A_correct/f_none = filters; "elig" = clean_B top-1 = B; "both" = clean A and clean B correct
(f_none: RS 203, OT 98, PC 151). NLDR = ratio of means unless "per-pair".

## Ledger
| # | claim (section) | stated | audit value | verdict | data / code | note |
|---|---|---|---|---|---|---|
| P1 | Final run on gnode058 at commit 240e5f6 (Prov.) | gnode058, 240e5f6 | manifests E17/E18/E19/E20/E22 + 9 E21 E12 runs: hostname gnode058, commit 240e5f614…, tags [] | match | ana_runs/*/manifest.json; audit_prov.py (manual listing) | |
| P2 | 5 experiment runs + 9 E21 runs, all complete (Prov.) | 5 + 9 | 5 + 9, status complete | match | manifests | |
| P3 | Hook checks are 0 (Prov.) | 0 | max_abs_dD = 0.0 in every hook_checks entry (E17 6, E18 24, E19 18, E20 18, E22 48 fields) | match | summary.json | |
| P4 | Bit-identical across three RTX 2080 Ti nodes gnode077/084/058 incl. per-pair logits (Prov.) | identical | clean_A/clean_B/full_A/feat_A_errB/meanPos logits: max abs diff 0 over every pairwise join of E4, E8, E9, E2r, E17, E18, E20 (1,075–3,565 shared rows per pair of runs), top-1 agreement 1.000 | match (identity) | audit_prov.py | E3/E4/E5/E8 manifests say hostname "Poseidon" (finalized on DEV) and never name gnode077, see N1 |
| P4b | ...one of the three nodes is gnode077 (Prov.) | gnode077 | no manifest names gnode077: E3/E4/E5/E8 (the only runs not on gnode084/058 or the laptop) record hostname "Poseidon" (finalize_note: finalized on DEV from committed parts) | not reproducible | manifests | the GPU node of those runs is not recorded on disk (N1) |
| P5 | DAS rotations retrained bit-identically (Prov.) | identical | E9 (gnode084) vs E17/E18/E20 (gnode058): np.array_equal True for RS das1/2/4/8, OT das1/2/4/8, and the E20 copies; train acc identical | match | directions npz; audit_prov.py --das | |
| P6 | "39,560 shared values" (Prov.) | 39,560 | the count depends on which runs and conditions are joined; my joins give 2,150–7,130 logit values per run pair | not reproducible | audit_prov.py | the identity holds; the specific count cannot be rebuilt without the original join |
| P7 | gnode058 logs show stage 1 recomputed and DAS retrained (Prov.) | yes | E17 log.stage1.txt: 15 shards of 96 pairs, about 27 s each; log.wE17_s3-*.txt: "DAS: training das1 (k=1, 3 epochs, lr 0.01, seed 0)…" | match | ana_runs/E17/*/log.* | |
| P8 | Different GPU (RTX 5060 laptop) within about 0.5 points (Prov.) | ≤ ~0.5 pts | E0/E2-5ad7 (laptop) vs E9/E18/E8/E2r: RS L18 flip gap 0.0867 vs 0.0800 (0.67 pts), NLDR gap 0.1256 vs 0.1235 (0.21); OT L12 NLDR gap 0.2587 vs 0.2516 (0.71) | **mismatch** (minor) | audit_prov.py --gaps | write "within about 0.7 points". Laptop runs on other commits (E2 2c7b/ffa4, f0c41753e/e89fd4c08) give RS NLDR gap 0.081 (different code version; not comparable) |
| P9 | Independent recomputation of 96 table rows reproduced exactly (Prov.) | 96/96 | not re-done as a list; every table value I recomputed (RQ1, E17, E18, E19, E20 rows) agrees to 3 dp | not audited | – | meta-claim |
| R1 | RQ1 GO for RS L18; gate 0.147 [0.113, 0.180], n = 450 (prereg.) | 0.147 [0.113, 0.180], 450 | 0.1467 [0.1133, 0.1800], 450 pairs / 450 eligible; full_A 1.000, meanPos 0.853 | match | E3 7bf755d092fa, RS L18 fit fresh, eval fresh, f_pilot; audit_rq1.py | |
| R2 | E9: error-U not effective for RS L18; DAS-4/8 effective with both flags (prereg.) | – | pca/answer_sep r1–8 full flip ≤ 0.067 (no row clears p95 + 0.10); das4 0.760 / feat 0.547, das8 0.867 / 0.680, both effective; full − feat 0.213 [0.140, 0.287], 0.187 [0.120, 0.260] | match | E18 rows, bit-identical to E9 (P4/P5); audit_das_flips.py | |
| R3 | E10 Q2 not met (prereg.) | not met | E10 RS best steering flip 0.200 (das2 s=4, random p95 0.007); no row has flip ≥ 0.5 with p95 ≤ 0.05 | match | E10 9eb5f4d77046; audit_extra.py | |
| R4 | M1 supported for every non-English set, 44/44 at 16k and 65k (prereg.) | 44/44 | 44/44 (recovered_mean upper CI < en_pile lower CI; 500 window bootstraps) | match | E12 per_window npz; audit_m1.py | |
| R5 | M2 only testable for French PC and not supported; M3 not evaluable (prereg.) | – | – | not audited | – | non-numeric |
| R6 | With B's actual error the RS shortfall is 0.098 (red team) | 0.098 | 0.0978 | match | audit_rq1.py | |
| R7 | Shortfall 0.204 for low-margin pairs, 0.089 for high-margin pairs | 0.204 / 0.089 | 0.2035 (n 226, median lead 1.76) / 0.0893 (n 224, median 3.23); median split of clean-A lead of A over the best other candidate | match | E3 cand_logits; audit_rq1.py | |
| R8 | Family-cluster CI rests on 3 clusters | 3 | 3 families: rs_plus 0.087, rs_story 0.213, rs_symbolic 0.140 (n 150 each); cluster CI [0.087, 0.213] | match | audit_rq1.py | |
| R9 | E9 labels depend on the DAS seed | – | see C2.1 | match | – | |
| C1.1 | No rank-1/2 DAS fit is valid in E17: 16 variants, best train acc 0.473 | 16, 0.473 | 16 fits (8 rank-1, 8 rank-2), max 0.473 (das2_e30), none ≥ 0.5 | match | E17 directions json; audit_das_geom.py | in-sample train accuracy |
| C1.2 | Ranks 4–8 valid in all 6 fits, reaching 0.98 (das8_e10) | 6/6, 0.98 | 6/6 (min 0.647 das4_s1), max 0.980 das8_e10 | match | same | |
| C1.3 | Rank-1 learned direction seed-stable: top cosine 0.74 [0.60–0.86] over 16 pairs; random p99 0.05 | 0.74, 0.60–0.86, 16 | median 0.740, range 0.595–0.862, 16 different-seed pairs of the 7 lr-1e-2 fits; raw (init included) median 0.434; Haar p99 (2000 MC) 0.052 | match | audit_das_geom.py | brackets are a min–max range, not a CI; 3 seeds, 7 fits, 16 non-independent pairs (W2) |
| C1.4 | The rank-1 direction flips 27–35% of pairs vs random p95 0 | 0.27–0.35 | seed-0 rank-1 fits 0.273/0.300/0.347; all 7 lr-1e-2 rank-1 fits **0.087–0.347** (das1_s1 0.087, das1_e10_s1 0.147, seed 2 0.307/0.320); random p95 0.000 | **mismatch** | E17 f_pilot n=150; audit_das_flips.py | the range covers seed 0 only |
| C1.5 | Gate ≈ threshold on outcome: train acc vs test flip Spearman 0.96 | 0.96 | 0.961 (p 1.3e-12, n 22 fits incl. collapsed lr3e-2); 0.948 for the 20 lr-1e-2 fits | match | e17_flips.csv | |
| C1.6 | Seven atoms (15376, 7686, 8352, 11219, 13950, 411, 14691) in the top 10 of 16k in every rank-8 and rank-16 fit, incl. seed 1 | every rank-8/16 fit | rank by cos² with the learned span: all 7 in the top 10 for das8, das8_e10, das8_s1 (worst 9th) and das4_e10; **das16: atom 411 is 11th**; das4: 14691 is 11th; das4_s1: 14691 86th | **mismatch** (minor) | 16k L18 W_dec (unit rows); E17 + E18 das16; das_atoms.csv | das.md's ranks appear to be 0-based, so "top 10" there means top 11. Write "in the top 11 of every rank-8/16 fit" or "top 10 of every rank-8 fit" |
| C1.7 | 20 atoms explain 0.29–0.49 of each learned subspace vs random p99 0.03–0.12 | 0.29–0.49; 0.03–0.12 | greedy subspace OMP-20 on learned components: ranks 1–8 0.290–0.485 (lr3e-2 rank 2 0.317); **das16 0.192**; Haar p99 (20 draws/k) k1 0.112, k2 0.069, k4 0.046, k8 0.032, k16 0.023 | match (ranks 1–8) | audit_das_geom.py | "each" holds only for ranks 1–8; the rank-16 fit is 0.19 |
| C1.8 | Error slice 0.15–0.31 of the learned subspace in the 32-dim error PCA (random 0.014) | 0.15–0.31 | 0.148–0.305 (lr 1e-2 fits; collapsed lr3e-2 fit 0.326); Haar 32/2304 = 0.0139 | match | E17 pca_pooled[:, :32]; audit_das_geom.py | |
| C1.9 | Error-U effective only at r ≥ 32 (E18): flips 0.15–0.28; never feature-mediated even at r = 64 | 0.15–0.28 | f_pilot: pca r32 0.153, r64 0.247; answer_sep 0.173, 0.267 (p95 0.000); f_A_correct 0.158–0.276; f_none 0.167–0.282. feat_A_along_U at r64 0.053/0.067 vs p95 0.007, not effective | match | E18; audit_das_flips.py, audit_misc.py c1 | the upper end 0.28 comes from f_none; under the primary f_pilot the range is 0.15–0.27 |
| C1.10 | Repair along DAS closes 0.90–1.06 of the residual NLDR gap vs 0.28 for error PCA r = 4 | 0.90–1.06; 0.28 | das4 0.901, das8 1.057 (das1 0.733, das2 0.791, das16 0.949); pca r4 0.282; featA_errB NLDR 0.876 | match | E18 RS f_pilot; (NLDR_repair − NLDR_errB)/(1 − NLDR_errB) | the range covers das4/das8 only |
| C2.1 | Seed 1 vs 0: das4 full 0.760 → 0.593, feat 0.547 → 0.307 (BH q < 0.001) | as stated | full 0.760 → 0.593 (31 vs 6 discordant, q 2e-4); feat 0.547 → 0.307 (39 vs 3, q < 1e-4) | match | exact McNemar, BH over 20 seed contrasts; e17_seed_tests.csv | das8 full 0.867 → 0.733 (q 0.003) |
| C2.2 | Fits with the same (k, seed) share their init; 3-epoch fits keep 41–76% of energy in it | 41–76% | E17 3-epoch fits 0.405–0.673; E18 das16 0.758 | match | audit_das_geom.py, das_flips.log | 0.76 comes from das16 (E18), not from E17 |
| C2.3 | Fixed-seed bootstrap CI half-width about 0.07 | ~0.07 | median 0.075 (range 0.043–0.080 over non-degenerate fits) | match | e17_flips.csv | |
| C2.4 | RS illusion flag turns false at DAS-16 | false | full − feat 0.073 [0.000, 0.153], lower bound not > 0 | match | E18 RS f_pilot | the table gives lower −0.007; bootstrap noise at the boundary |
| C2.5 | Seed-robustness of the illusion flag cannot be shown at the 10-point bar | – | – | not audited | – | TOST/power claim (stats.md); not recomputed |
| C3.1 | das4 moves answers up in 100% of trajectories and never below B | 100%, never | 203 both-correct pairs (f_none): 100% moved, 100% of them monotone up, 0% below B at any dose; mean change s0.25→16 +2.37 | match | E19; digit tokens from prompt cand_ids; audit_steer.py | output "1" coded as 10 |
| C3.2 | pca r1/r4 move answers down in 86–96% | 0.86–0.96 | 0.857 / 0.962 of moved pairs monotone down | match | same | only 45% / 39% of pairs move at all |
| C3.3 | das4 s = 4 flips 0.32 of A > B and 0.00 of A < B (Fisher p = 1e-11) | 0.32 / 0.00, 1e-11 | **both-correct, f_none (n 106 / 97): 0.321 / 0.000, p 1.3e-11**; f_none eligible (109/100): 0.312/0.000, p 1.3e-11; f_pilot (86/64): 0.314/0.000, p 4.1e-8 | match | E19 RS; audit_steer.py | the claim needs its pair set stated (W4) |
| C3.4 | E10b replicates on fresh templates | replicates | das1/2/4 at s = 4: flips on A < B pairs 0.000 in rs_plus, rs_story, rs_symbolic; A > B 0.17–0.44 | match | E10b f_none | |
| C3.5 | Ceiling max_k P(A − B = k) = 0.353 | 0.353 | f_pilot eligible n = 150: Δ = +1 0.353 (−3 .053, −2 .207, −1 .167, +2 .173, +3 .047); f_none eligible 0.359, both 0.355 | match | E19 + add_chain digit ids | the ceiling assumes a shift by one integer k; it is a design bound, not a measured one |
| C3.6 | Specific but weak: 0.18 flips vs random p95 0.007 (q = 0.005) | 0.18, 0.007, q .005 | das4 s=4 0.180; random r4 mean 0.0023, p95 0.0070; spec 0.173 [0.113, 0.233], bootstrap p 0.001 (B = 1000) | match | E19 f_pilot | BH q over 189 tests not rebuilt |
| C4.1 | At s ≤ 1, 99–100% of random-steered non-B outcomes land on A's box (chance 1/6) | 0.99–1.00 | r1: 1.000 / 1.000 / 0.9915; r4: 1.000 / 0.996 / 0.991; r8: 1.000 / 1.000 / 0.999 (s = 0.25 / 0.5 / 1; 65–748 outcomes) | match | E19 OT f_pilot, 89 eligible, 20 seeds; 7 candidate boxes from the prompt file | |
| C4.2 | Random-flip propensity tracks the clean-B margin, Spearman −0.85 | −0.85 | r4 −0.850, r8 −0.849, r1 −0.842 (s = 1, n 89) | match | same | |
| C4.3 | Noise in place of ε_B gives A in 89–92% of pairs | 0.89–0.92 | OT L12 pilot templates, feat_B_noise: E8 0.927 (f_pilot) / 0.918 (f_A_correct) / 0.882 (f_none); E1 0.899 / 0.892 / 0.855 | match (subset) | E8, E1; audit_steer.py | full range over runs/filters 0.86–0.93; on fresh templates the noise control is ~0.50 (RQ1 table) (W5) |
| C4.4 | DAS steering at s = 0.25 flips 0.98–1.00 vs random p95 0.11–0.15; error-U 0.58–0.67 | as stated | das4 0.978, das8 1.000, das8_e10 0.978; random p95 0.112 / 0.114 / 0.146 (r1/4/8); pca 0.607–0.663, answer_sep 0.584–0.674 | match | E19 OT f_pilot | |
| C4.5 | Features alone erase B along DAS-1 (0.989) | 0.989 | feat_A_along_U das1 0.989 (88/89) | match | E18 OT f_pilot | |
| C5.1 | PC error-U flips 0/150 at every rank up to 64, L12 and L16 | 0/150 | 0 flips for pca and answer_sep, r = 1…64, both layers | match | E18 PC f_pilot | |
| C5.2 | DAS valid only at rank 16 (0.81 / 0.78) | 0.81 / 0.78 | L12 0.810, L16 0.780; ranks ≤ 8 ≤ 0.437 | match | E18 json | |
| C5.3 | DAS16 full 0.79 / 0.83, feat 0.42 / 0.27; full − feat 0.37 / 0.55, "largest of any cell" | as stated | full 0.793 / 0.827; feat 0.420 / 0.273; diff 0.373 [0.300, 0.453] / 0.553 [0.473, 0.633] | match (numbers) | E18 PC f_pilot | **"largest of any cell" is false**: OT error-U r1 full − feat = 0.910 − 0.112 = 0.80 (r2 0.85). It holds among DAS fits (W6) |
| C5.4 | A's city explains 57% (L12) and 71% (L16) of per-pair DAS variance | 0.57 / 0.71 | median eta² over the 10 DAS keys (full/feat × das1–16), per-pair NLDR clipped [−1, 2], 151 both-correct pairs, 16 cities: 0.571 / 0.705 (range 0.24–0.68 / 0.48–0.83); permutation p95 ≈ 0.16 | match | E18 + people_cities_en prompts; audit_misc.py c5 | median over keys; das16 keys lower |
| C5.5 | das1 acts as an "Athens" direction | Athens | das1 full NLDR: Athens 0.43 at both layers; next city 0.14 (L12 Mumbai) / 0.11 (L16 Cairo) | match | same | |
| C5.6 | Steering pushes toward a fixed city: Tokyo in 93% of landings, das8 s = 16 | 0.93 | 0.929 of 70 non-A/B candidate landings; das4: Paris 0.574 of 54 | match | E19 PC f_pilot | |
| C5.7 | No PC seed replicates; all PC fits at rank ≤ 8 invalid | – | max acc rank ≤ 8: 0.437 (L12), 0.327 (L16); only seed 0 exists | match | E18 json | |
| C6.1 | Feature capture of the DAS subspace rises with width: RS das4 0.754 → 0.827 | 0.754 → 0.827 | 0.754 → 0.827; Δ feat NLDR +0.055 [+0.036, +0.075] (same U, same pairs, paired); das8 0.763 → 0.842 | match | E9 vs E20 f_pilot; audit_width.py | full_along_U is identical across widths by construction |
| C6.2 | OT das4 0.532 → 0.574 (65k), 0.591 (262k); paired CIs exclude 0 | as stated | 0.532 → 0.574 / 0.591; Δ feat NLDR +0.078 [0.062, 0.093] / +0.110 [0.091, 0.128] | match | same | OT NLDR overshoots 1 (full 1.85) |
| C6.3 | Total feat_A_errB gap does not change at 65k: RS +0.020 [−0.035, 0.074] | +0.020 [−0.035, 0.074] | flip gap, f_A_correct (n 203 elig): +0.020 [−0.030, +0.074]; f_pilot +0.033 [−0.013, +0.080]; f_none +0.024 [−0.033, +0.081] | match | audit_width.py | **NLDR gap shrinks under f_A_correct −0.024 [−0.046, −0.002] and f_none −0.025 [−0.049, −0.004]** (f_pilot −0.011 [−0.032, +0.012]); "does not change" is metric-dependent (W7) |
| C6.4 | Gap falls only for OT at 262k (−0.105) | −0.105 | NLDR f_pilot −0.105 [−0.122, −0.088]; 65k −0.010 [−0.027, +0.005] | match | same | |
| C6.5 | Error-U repair: RS pca r1 0.225 → −0.033; OT pca r8 0.547 → 0.734 | as stated | 0.225 → −0.033; 0.547 → 0.734 (262k 0.663) | match | share = (NLDR_rep − NLDR_errB)/(NLDR_fullA − NLDR_errB) | |
| C6.6 | NEW lead claim: capture median across valid native f_pilot DAS fits RS16k 0.754 (16 fits), 65k 0.834; OT 16k 0.565, 65k 0.598, 262k 0.612; PC L12 0.690, L16 0.659 | as stated | values reproduced exactly (RS 16k rows 16 → 0.754; 65k 0.834; OT 0.565 / 0.598 / 0.612; PC 0.690 / 0.659) | match (values) | E9/E17/E18/E20/E22; capture_all.csv; audit_width.py | **"16 fits" counts 4 duplicated rotations** (E9 and E18 das4/das8 are bit-identical to E17's): 12 unique fits, median **0.745**; 7 unique pilot-template fits (no E22) 0.754. OT "fits" are 5 unique (9 rows). The 65k/262k medians are over 2 fits and PC over 1; comparing medians across widths mixes fit sets; the paired per-fit numbers (C6.1–2) are the right comparison (W8) |
| C7.a | Task-state loss language-invariant: RS 0.116–0.148, PC 0.221–0.244 | as stated | feat_A_errB NLDR gap, f_A_correct, native-fit rows: RS en 0.148, fr 0.125, hi 0.123, hinglish 0.116; PC en 0.237, fr 0.241, hi 0.244, hinglish 0.221 | match | E22; audit_misc.py c7 | |
| C7.b | English-fit RS DAS transfers: flips 0.50–0.63 vs 0.54–0.61 at home | as stated | Eng-fit full_along_U in fr/hi/hinglish 0.503–0.632; English home 0.539 (das4) / 0.611 (das8) | match | E22 f_A_correct | |
| C7.c | Equivalence only to within about ±15 points (underpowered) | ±15 | Eng-fit − home CIs span −0.139 to +0.161 (half-widths ≈ 0.10–0.11) | match | same, unpaired bootstrap | |
| C7.d | PC transfer fails (−0.23 to −0.27) | −0.23 to −0.27 | das8 Eng-fit − English home: fr −0.231, hi −0.271, hinglish −0.258 | match | same | every PC fit ≤ rank 8 is invalid, including the English one |
| C8.1 | IOI L20 loses 36% of the logit difference but flips only 1.3% | 0.36 / 0.013 | NLDR gap 0.358, flip gap 0.013 (n 150) | match | E1 854a2370f878 f_pilot | |
| C8.2 | OT L12 loses 26% and flips none | 0.26 / 0 | 0.259 / 0.000 (94 eligible) | match | same | |
| C8.3 | RS per-pair loss is additive: slope −0.005 [−0.067, 0.054] | as stated | −0.005 [−0.062, 0.048], intercept 0.84 (B = 1000) | match | E8 RS L18 f_pilot both-correct; loss = D_A − D_errB vs swing = D_A − D_B | |
| C8.4 | Gain shape (slope CI > 0) in only 7 of 36 E8 cells | 7/36 | 7/36: RS L20, OT L12, IOI L12/14/18/20, VB L20 | match | gain_slopes.csv | |
| O1 | Hard RS pairs: start-0 pairs least patchable, DAS4 NLDR 0.65 vs 1.00 | 0.65 vs 1.00 | per-pair das4 full NLDR by start: 0 → 0.65 (n 78), 1 → 0.77, 2 → 0.85, 3 → 1.00 (n 22), 4 → 0.89 | match | E18 RS f_none both-correct; add_chain_pilot prompts | 1.00 is the best single group (start 3, n = 22), not "the other pairs" (W9) |
| O2 | Replicates on independent E22 pairs (15/16 cells positive) | 15/16 | ρ(full_along_U NLDR, start) > 0 in 15/16 family × U cells; the negative one is hi pca r8 (−0.01), hi pca r4 +0.03 | match | E22 + add_chain_e13 prompts | rows pool native- and English-fit U; positivity, not significance |
| O3 | Per-pair responses low-rank: first mode 39–72%, stable across halves | 0.39–0.72 | PC1 share RS 0.385, OT 0.715, PC12 0.456, PC16 0.548 (86 columns); split-half abs cos of V1 0.90–0.99 (one random split) | match | E18 f_none both-correct; audit_misc.py modes | |
| O4 | First mode unrelated to per-pair error cost | unrelated | Spearman(featErrB per-pair NLDR, das4 full) RS −0.14, OT −0.04, PC12 +0.07, PC16 +0.21 | match | audit_extra.py | PC16 +0.21 is not nil |
| O5 | RS error carries graded magnitude: same − diff +0.16 to +0.21 at ±3/±4, about 0 at ±1 | as stated | red-team definition (both-correct, common denominator, donors rebuilt with the harness rule over the fresh cell's pairs sorted by (family, rank)): ±1 +0.021 [−0.006, 0.050] (n 222), ±2 +0.028, ±3 +0.166 [0.117, 0.218], ±4 +0.217 [0.143, 0.297] | match | E3 RS fresh; audit_misc.py resamp | donor rebuild validated only through the 'none' status (3/3 agree). Per-pair NLDR (f_A_correct) gives ±1 +0.005, ±2 +0.060 [0.003, 0.133], ±3 +0.182, ±4 +0.288 |
| X1 | Donor-pull bug: donors rebuilt in the wrong order (1.9–2.8% match) (Corrections) | 1.9–2.8% | pair_id vs (family, rank) order, same diff-answer donor: E2 1.9%, E2r 2.3%, E3 eval-pilot 2.3% | **mismatch** (minor) | audit_extra.py | 2.8% not reproduced; write 1.9–2.3%. "Recomputed, the conclusion is unchanged" not audited |
| X2 | "Error acts as uniform gain" weakened; RS additive (Corrections) | – | see C8.3–4 | match | – | |
| X3 | Bit-exact across 3 nodes; independent hardware within about 0.5 points (Corrections) | – | see P4, P8 | (counted in P4/P8) | – | |

### C7 text-fidelity rows (sub-audit, `audit_ml.py`; merged unchanged from `team/audit_ml_ledger.txt`)
Data: `ana_runs/E12/*/per_window_<set>_L<layer>.npz` (16k gnode084, 65k = E21 gnode058; en_pilot100 excluded). CEM CIs B = 500,
overlap/width B = 1000 window bootstraps within set; wild-cluster p enumerates all 2^12 sign vectors.

| # | claim | stated | audit value | verdict | data / code | note |
|---|---|---|---|---|---|---|
| ML1 | C7 raw native/English splice-cost ratio | ~2.7x (geo mean; reg 2.74) | regression (cell FE + native, log cost>0) 2.738, CR1 [2.26, 3.31], wild p 0.0002; plain geometric mean of the 24 cell ratios 2.82 (range 2.12-3.88) | match | audit_ml.py regress / match | "about 2.7x" ok; it is a 6-vs-6-set (12-cluster) contrast |
| ML2 | C7 matched on clean NLL (CEM 0.25 nat) | 2.95x, 24/24 > 1 | median 2.953 [2.317, 4.077], 24/24 point > 1, 24/24 CI > 1 | match | audit_ml.py match (ml_matched.csv spec P, ref en_pile) | |
| ML3 | C7 matched on clean NLL + abs(eps)/abs(x) (CEM) | 1.47x (1.23-1.75; 19/21 cells > 1) | supported cells n=21: median 1.469 [1.228, 1.751]; point > 1 in 21/21, CI lower bound > 1 in 19/21 (all 24: median 1.475, 21/24 CI > 1) | match (wording) | ml_matched.csv spec PE | "19/21 > 1" is the count of CIs excluding 1, not of point estimates (21/21). 3 of 6 L12-65k cells unsupported (<50 % matched). Single-cell check vs team/stats_out/mm_matched.csv: all 28 native/en PE cells identical to 3 dp (max abs diff 0.0), e.g. hi_deva_web L12 16k 1.228 [1.017, 1.607]; ta_taml_dakshina L12 16k 1.530 [1.131, 2.567]; bn_web L18 16k 1.501 [1.289, 1.723]; ta_web L18 65k 1.751 [1.521, 2.078]; hi_deva_web L12 65k 1.263 [1.096, 1.518] |
| ML4 | C7 vs pooled Latin sets | 1.56x | supported n=23: median 1.558 [1.311, 2.386], 23/23 CI > 1 | match | ml_matched.csv ref latin_other | |
| ML5 | C7 regression with set clusters | 1.41-1.53 | rcs4 + log abs(eps)/abs(x): no language FE 1.527 (CR1 [1.36, 1.71], wild p 0.001); with language FE 1.412 ([1.10, 1.82], wild p 0.117) | match | audit_ml.py regress | the lower end (1.41) is NOT significant (wild p 0.12); 12 clusters |
| ML6 | C7 holds at 16k and 65k | yes | CEM native/en PE: L12 16k median 1.41, L12 65k 1.49 (3/6 supported), L18 16k 1.49, L18 65k 1.45; all supported cells > 1 | match | ml_matched.csv | |
| ML7 | C7 larger at L12 | larger at L12 | CEM medians: NOT larger at L12 (L12 1.41/1.49 vs L18 1.49/1.45). Per-cell regression (rcs4+eps, no FE): L12 16k 1.77, L12 65k 1.51, L18 16k 1.46, L18 65k 1.43 (larger at L12 16k only) | partial / mismatch for CEM | ml_matched.csv; ml_regress.csv cell_* rows | true only for the regression and mainly for L12 16k; drop or qualify |
| ML8 | C7 reverses at L6 (L24 1.94-1.95) | L6 0.70-0.74, L24 1.94-1.95 | L6 hi_deva_dakshina 0.735 [0.625, 0.993], hi_deva_web 0.701 [0.565, 0.883]; L24 1.953, 1.938 | match | ml_matched.csv layer 6/24 | only the 2 Hindi native sets exist at L6/L24 (16k only): "reverses at L6" is a Hindi-only, 2-set statement |
| ML9 | C7 placebos between Latin sets | 0.64-2.28 | PE spec, Latin target vs pooled other Latin: 0.638-2.280 (median 0.965, 9/24 CI > 1); nllc-only spec 0.648-2.254 | match | ml_matched.csv native==0 | |
| ML10 | C7 with language FE | 1.14-1.92, significant in 5/16 specs | 16 specs (lin/rcs4/rcs6/bins x FVU/eps/raw/FVU+eps): 1.140-1.921, wild p < 0.05 in 5/16 (no FE: 1.46-3.03, 16/16) | match | ml_regress.csv | |
| ML11 | C7 dropping English and French | 1.05 on common support | rcs4 + log FVU, no FE: 1.046 on common support (n=15113 windows); 1.069 on all windows | match (1.05); note | ml_regress.csv *_drop_en_fr | stats.md says 1.25 on all windows; I get 1.07 (support defined on the full sample). Synthesis only quotes 1.05, which matches |
| ML12 | C7 width: native cost falls 15-22 % vs English 27 % (L12) | 15-22 % vs 27 % | English L12 -27.3 % [ratio 0.727, CI 0.713-0.740]; native L12 -15.0 % to -21.0 %, L18 -15.0 % to -22.1 %; English L18 -21.7 % | match (wording) | audit_ml.py width (ml_width.csv) | the contrast holds at L12; at L18 English (-21.7 %) lies inside the native range (hi_deva_dakshina -22.1 %); the sentence mixes layers |
| ML13 | C7 width removes less native FVU, all 12 CIs exclude 0 | 12/12 | native - English change in log pooled FVU: +0.050 to +0.075, 12/12 CIs exclude 0 | match | ml_width.csv | window-bootstrap CIs only (no SAE-training replicate); very narrow |
| ML14 | Corrections: within-Hindi overlap, adj. for nllc, L12 16k | 1.34 [1.09, 1.71] | overlap nllc [3.579, 4.445], 79 vs 188 windows; 1.341 [1.087, 1.695] (B=1000) | match | audit_ml.py overlap (ml_overlap.csv) | 4 % tail of native windows; adding eps gives 1.21 [0.97, 1.55], CI includes 1 |
| ML15 | Corrections: within-Hindi overlap elsewhere | 0.6-1.0 | L6 0.586 [0.37, 0.89]; L12 65k 0.996 [0.74, 1.32]; L18 16k 0.708 [0.56, 0.89]; L18 65k 0.641 [0.53, 0.77]; L24 0.814 [0.73, 0.92] | match | ml_overlap.csv | at L6/L18/L24 native is significantly CHEAPER (CI < 1); "0.6-1.0" understates that the sign reverses |

## Notes
- **N1 (provenance).** Logit identity across runs is real (P4) and the gnode058 runs recomputed stage 1 and retrained DAS (P7), so
  this is deterministic recomputation. But the "three RTX 2080 Ti nodes" cannot be fully checked: the E3/E4/E5/E8 manifests carry
  the DEV finalizer's hostname (Poseidon), not a GPU node.
- **N2 (definitions I had to pin down).** C3.3 holds on the f_none both-correct pairs (n 106/97); under f_pilot it is 0.314 vs
  0.000 with p 4e-8. C5.4 is a median over 10 DAS keys of eta² on clipped per-pair NLDR. O5 matches only under the red team's
  common-denominator definition; the claim's direction holds under per-pair NLDR as well. C1.6 depends on 0- vs 1-based ranking.
- **N3 (digit tokens).** The add_chain digit token ids are not contiguous (cand_ids '0'..'9' = 235276, 235274, 235284, 235304,
  235310, 235308, 235318, 235324, 235321, 235315). Any analysis that takes `ans_A_id − ans_B_id` as A − B gets nonsense (my
  first pass did; fixed in `audit_steer.py`). Worth a check in any script that infers values from token ids.
- **N4 (scope).** Statements already flagged in the synthesis as single-seed, post hoc or E23-dependent were not re-flagged
  unless the number itself was off.

## Wording I would change (claim stronger than the evidence, or imprecise)
- **W1 (C1, "It flips 27–35% of pairs").** That range is seed 0 only. Across the 7 lr-1e-2 rank-1 fits it is 0.09–0.35. Write
  "rank-1 fits flip 9–35% (27–35% for seed 0)".
- **W2 (C1, "0.74 [0.60–0.86]").** Square brackets read as a CI. It is a min–max over 16 non-independent pairs from 7 fits and 3
  seeds. Write "median 0.74, range 0.60–0.86".
- **W3 (C1, atoms).** Write "in the top 10 of every rank-8 fit and the top 11 of das16". The "20 atoms explain 0.29–0.49" range
  excludes the rank-16 fit (0.19).
- **W4 (C3, one-sided flips).** State the pair set: "das4 at s = 4 flips 0.32 of A > B pairs (n 106) and 0 of A < B pairs (n 97),
  both-correct test pairs" (f_pilot: 0.31 vs 0, p 4e-8). The 0.353 ceiling is a design bound under a single integer shift.
- **W5 (C4, noise 89–92%).** That is pilot templates only, with a range of 0.86–0.93 over runs/filters. On fresh templates the
  noise control gives A in about 50% (RQ1 table), so the fallback is template-specific.
- **W6 (C5, "largest full − feature gaps of any cell").** False: the OT error-U r1/r2 gaps are 0.80/0.85. Write "of any DAS fit".
- **W7 (C6, "total gap does not change at 65k").** That holds for the flip gap (+0.020 [−0.030, 0.074], f_A_correct). The RS NLDR
  gap shrinks slightly but with a CI excluding 0 under f_A_correct and f_none (−0.024 [−0.046, −0.002]). Write "no change in flip
  gap; at most a 2-point NLDR reduction".
- **W8 (new lead capture claim).** Report unique fits: RS 16k is 12 unique fits (median 0.745) or 7 pilot-template fits (0.754),
  not 16. 65k/262k medians are over 2 fits, PC over 1 fit. Make the cross-width statement with the paired same-U numbers (RS das4
  0.754 → 0.827, das8 0.763 → 0.842; OT das4 0.532 → 0.574 → 0.591).
- **W9 (Other, "DAS4 NLDR 0.65 vs 1.00").** 1.00 is the start-3 group (n 22). Starts 1–4 give 0.77–1.00. Write "0.65 at start 0
  vs 0.77–1.00 for starts 1–4".
- **W10 (Provenance).** Write "different GPU architecture within about 0.7 points". Drop "39,560", or give the join. gnode077 is
  not recorded in any manifest.
- **W11 (C7, "larger at L12").** It holds only for the per-cell regression at L12 16k. The CEM medians are 1.41/1.49 at L12 vs
  1.49/1.45 at L18. "Reverses at L6" rests on 2 Hindi sets at 16k only. "19/21 cells > 1" counts CIs excluding 1 (all 21 point
  estimates are > 1). The lower end of the regression range (1.41, with language FE) has wild p 0.12. The within-Hindi "0.6–1.0
  elsewhere" hides that native is significantly *cheaper* at L6/L18/L24 (CI < 1).
- **W12 (C1, "0.15–0.28").** Under the primary filter f_pilot the range is 0.15–0.27; 0.28 is f_none. Repair "0.90–1.06" covers
  das4/das8 only (das1 0.73, das2 0.79, das16 0.95).
- **W13 (Corrections, donor bug).** Write 1.9–2.3%, not 1.9–2.8%.

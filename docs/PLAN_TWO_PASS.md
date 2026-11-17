# Two-pass completion plan (2026-11-16)

Decided by the humans on 2026-11-16: running sum L18 = GO; templates approved; finish the project in **one compute pass** that runs
every remaining experiment (including redundant variants) and **one analysis pass** that applies the preregistered rules and rejects
what does not qualify. Release 3 (E12, E13) runs first, separately, because it is ready and cheap.

## State before the passes

| item | state |
|---|---|
| Release 1 (E0, E1, E2, E6, E7) | done, ingested |
| Release 2 (E3, E4, E5, E8) | computed on gnode074 (3 x 2080 Ti, ~16 h), finalized on DEV from parts; tables: RQ1 gate, E4 width persistence, exploratory digests; E8 table being computed |
| Release 3 (E12 fidelity, E13 behaviour) | built and tagged `run-20261116-1` (with template sign-offs) |
| Final pass (E9, E10, E11, E14, E4_1m) | built in `harness`; queue `queue/final-server.yaml`; tiny-model end-to-end runs on DEV |

## Pass 1: compute (gnode074, keeps the release-2 caches)

Operational steps: docs/FINAL_PASS_RUNBOOK.md and scripts/slurm_final.sbatch (one job: release 3, then the final pass).

1. Release 3: `sbatch --nodelist=gnode074 --gres=gpu:3 --export=ALL,QUEUE=queue/release3-server.yaml,CLEAN_CACHE_ON_SUCCESS=0 scripts/slurm_release2.sbatch`
2. Final pass (after release 3, or right away if the node is free; the two queues share nothing that conflicts, but one at a time keeps
   the GPUs uncontended): `sbatch --nodelist=gnode074 --gres=gpu:3 --export=ALL,QUEUE=queue/final-server.yaml,CLEAN_CACHE_ON_SUCCESS=0 scripts/slurm_release2.sbatch`
3. Each run publishes to `results_server/<tag>/`. Push it as a results branch (as for release 2). If a finalize fails again, push the
   raw `runs/` as before: DEV can finalize from parts.

| experiment | cells | conditions per cell | role | est. GPU-h |
|---|---|---|---|---|
| E9 illusion checks + DAS (rank 1, 2, 4, 8) | RS L18, OT L12 | 335 | RS confirmatory; OT exploratory | ~4 |
| E10 steering (x1, x2, x4; pca, answer_sep, DAS; 20 random) | RS L18, OT L12, PC L12 | 243 | RS confirmatory; others exploratory | ~3.5 |
| E11 people-cities L8-16 + OT L8-16, U transfer both ways | 18 | 83 | exploratory | ~7 |
| E14 multilingual gap, native and English-fit | 27 (3 tasks x 5 languages + 4 x English-fit) | 146 | exploratory; readable only past Gate A, Gate B and validation | ~24 |
| E4_1m 1m SAE at L12 | 3 | 12 | exploratory (redundant width point) | ~2 |
| stage 1 for the new sets, preflight | | | | ~3 |
| E9b, E10b (bonus: fresh-template generalization of DAS / U and of steering, RS L18) | 1 + 1 | 335, 243 | exploratory | ~10 |
| E2r (redundant: E2 with seed 1) | 2 | 196 | exploratory | ~2 |
| **total** | | | | **~55 GPU-h: ~18 h on 3 x 2080 Ti** (extrapolated from release 2, +-50%) |

## Pass 2: analysis (DEV, no model)

Order, each step a script that reads only ingested runs and writes CSV/MD:

1. Ingest (or finalize from parts) and `scripts/inventory_runs.py`: every run complete, non-smoke, single commit, no claims left.
2. Release 2 (already scripted): `make_gate.py`, `make_e4.py`, `make_e8.py`, `make_tables.py`, `explore_r2.py`.
3. Release 3: `make_e12.py` (M1: CI non-overlap vs English per layer), `make_e13.py` (Gate A, Gate B).
4. Final pass: `make_final.py` -> `E9_illusion.csv`, `E10_steering.csv` (Q2 rule columns), `E11_spec.csv`, `E14_spec.csv`,
   `E14_lang_gap.csv` (M2 flags; `readable` = Gate A and Gate B and validated), `<EXP>_conditions.csv`; `make_e4.py` again for E4_1m.
5. Figures from CSVs only; `scripts/make_results_branch.py` -> `results-consolidated`.
6. Report: confirmatory statements only from rows the preregistration + addendum allow (RS L18 for E9/E10; E14 rows with
   `readable`); everything else is labelled exploratory.

## Decisions the humans must still take (before pass-2 results are read)

1. Freeze `PREREGISTRATION.md` and `PREREGISTRATION_ADDENDUM.md` (both still unfrozen).
2. E9: what counts as an illusion (for example: full_along_U recovers the flip while feat_A_along_U does not, by >= X points) and
   the cos(U, DAS) threshold. The table reports the numbers; the rule is yours.
3. E10: the Q2 criterion says "move to the other box"; for running sum the default reading in `make_final.py` is "the top-1 answer
   becomes A's answer" (flip rate >= 50% with random p95 <= 5%). Confirm or change.
4. M3: what "similar share" means for English-fit vs native-fit directions (for example within 10 points).

## Redundancy and rejection rules (fixed now, applied in pass 2)

- OT L12 rows of E9/E10 are exploratory because OT is not GO; they are reported as such or omitted, never as confirmatory.
- E14 rows enter M2/M3 statements only if `readable` (Gate A viable, Gate B within 10 points of English, template validated).
- Random-U nulls with fewer than 20 seeds are flagged underpowered (E8 rule) and are not used for a claim.
- DAS rotations whose train interchange accuracy is below the clean model's own train accuracy on A by > 20 points are reported as
  a failed fit (the rotation did not learn the interchange) rather than as evidence.
- Cut lines if pass 1 slips: drop E4_1m first, then E11, then E14's English-fit cells; E9/E10 RS L18 are never dropped.

## Timeline

| when | step |
|---|---|
| Nov 17 | push tags; start release 3 on gnode074; humans: decisions 1-4 |
| Nov 17-18 | release 3 finishes; final pass starts (~15 h) |
| Nov 18-19 | DEV: release-3 analysis while the final pass runs |
| Nov 19-21 | final pass results -> pass 2 |
| Nov 22-30 | figures, report draft, `results-consolidated` final snapshot |
| Dec 1-7 | report and slides |

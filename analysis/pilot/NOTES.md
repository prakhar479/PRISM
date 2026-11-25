# Pilot results: how to read analysis/pilot/VERDICTS.md (2026-11-26)

Sources: results/pilot-run-20261124-pilot-b (breadth pilot, gnode084 account) and results/pilot-g3-run-20261125-g3e
(second pilot). All numbers are exploratory and post hoc; every run passed scripts/verify_pilot_runs.py.

## Coverage
- Done: P0, P1, P5, P7, P8, P11 (breadth); P12, P13 (second). The four trained dictionaries (PD) are present.
- **Missing: P2 (dictionary sweep) and P4 (logit fixes / state) have no complete run**, nor PF_P2 / PF_P4 (CE per dictionary).
  They decide papers B and E. Resubmit on the breadth account: `sbatch --export=ALL,ONLY=P2,P4 scripts/slurm_pilot.sbatch`.
- P14 (the P12 dictionaries on object tracking) is queued on the second account, not yet run.
- P5 / P7 / PD ran on an **untagged commit c566c86485** that is not in this repository (before the review fixes: its resampU rows have
  donor_status n/a, which make_pilot backfills from the feat_A_resampC rows). Find out what it changed (`git show --stat c566c86485` on the
  breadth account) before using P5 / P7 numbers in a paper.

## Caveats on the verdicts
- **P0 (NO-GO as is):** capture moves by 0.06-0.13 across DAS objective x seed, and the code-level write-back differs from the subspace
  write-back by up to 0.4 (people-cities: up to 1.8). People-cities rank 4: 0 % of fits reach held-out IIA >= 0.5. Any capture difference
  below ~0.15 is inside the protocol noise; use kl + held-out validity and report the spread.
- **P1 / P5 (NO-GO):** no binding blind spot (content = swap within +-0.04, query higher); magnitude carries <= 0.13 of the resampled-error effect.
  Papers A (taxonomy) and C (magnitude) lose their main evidence.
- **P7 (GO, overstated):** the "share of the gap closed" divides by (full - feature-only). On object tracking L12 feature-only already
  reaches NLDR 0.966, so the gap is ~0.03 and grad_err "closes 2.5x" because its repair overshoots the clean state (NLDR 1.05): that is
  steering, not restoring. The people-cities rows (0.21-0.95 vs ~0.00 for error PCA) and running sum r<=4 are the meaningful ones.
- **P8 / P12 / P14:** Gemma 3 1B is at chance on the running sum (0 of 60 pairs pass the filter; P8's 4 and P12's 43 running-sum cells all
  skipped with "1 test pairs"). People-cities and object tracking pass.
- **P12 (GO, confounded):** capture tracks reconstruction error: Spearman(capture, relative error) = -0.80 at L13 over 35 cells (-0.73 within
  residual SAEs). Crosscoders are the one family below that trend (-0.26 vs the error-size fit) and the most robust family effect; replacement
  models with attention SAEs sit above it (n = 1 each). Composite "L0" is summed over every sublayer dictionary (up to ~1000), so it is not
  comparable to a residual SAE's L0. tc_single replaces one MLP of 14 (capture 0.98: near-trivial). CE / FVU per composite is not measured.
- **P13 (GO, permissive rule):** features carry most of the refusal state (diffmean capture 0.82-1.0 for residual SAEs at L7-22; feature-only
  gap 0.04-0.15), the error carries a minority that is largest early (L7 16k error-only 0.51; L22 ~0). Crosscoder 65k is lowest (0.59-0.68).

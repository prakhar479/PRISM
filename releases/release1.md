# Release 1 (DRAFT, not tagged yet): E0, E1, E2, E6, E7

Tag on approval: `run-YYYYMMDD-1`. Status: all DEV tests pass (T0: 130). T1 golden tests skip until R2's golden
export is returned. Every config dry-runs on the `run` profile and fits the 7.3 GB cap.

## What changed since the pilot

The new harness (`harness/`) runs the pilot's experiment design as three cached stages. It adds probability and
flip-rate metrics, error-slot baselines, filter flags in place of deletion, more direction estimators, and full
provenance. The pilot code (`src/`) is unchanged.

## Experiments in this release

| job | config | what | cells |
|---|---|---|---|
| E0 | configs/experiments/E0.yaml | pilot regression (acceptance: NLDR within ±1 point of the pilot) | OT L12, RS L18 |
| E1_split | E1_split.yaml | pilot features/error/noise conditions, full metric set, 3 filters | 4 tasks × L4, 8, 10, 12, 14, 16, 18, 20 |
| E1_repair | E1_repair.yaml | pilot repair cells (Table B.2) with same-task / other-task / generic / random directions | 13 cells |
| E2 | E2.yaml | all error-slot baselines + repair / repair_mean, r ∈ {0,1,2,4,8}, 20 random U | OT L12, RS L18 |
| E7 | E7.yaml | four direction estimators; PC1 variance and cosines in `directions/*.json` | OT L12, RS L18 |
| E6 | E6.yaml | OT every layer 8–16; L12 directions applied at every layer | 9 cells |

## Run machine: exact steps

Do steps 1–2 first, in the **existing pilot environment**, if RUN_REQUESTS R1–R3 have not been run yet. The golden
export they produce is used by this release's preflight.

```bash
git fetch --tags && git checkout <TAG> && git status --short          # clean
export HF_TOKEN=hf_...
# 1-2. RUN_REQUESTS R1-R3 with the pilot environment ($PY as in RUN_REQUESTS.md) -> probe/R1, golden/R2, hf_agreement/R3

# 3. separate, locked environment for the harness (the pilot env stays untouched)
uv venv .venv-harness --python 3.12
uv pip install --python .venv-harness/bin/python -r requirements/harness.lock \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
H=.venv-harness/bin/python
export HL_PROFILE=run            # optionally HL_CACHE_ROOT=/path/with/40GB  HF_HOME=...
$H scripts/check_env.py          # must print 0 errors

# 4. plan (memory per job; time estimates appear after preflight has measured throughput)
$H scripts/plan_runs.py --queue queue/release1.yaml --profile run

# 5. the queue: prefetch -> preflight -> E0 -> E1 -> E2 -> E7 -> E6 -> bundle (safe to kill and rerun)
$H scripts/run_queue.py --queue queue/release1.yaml
cat runs/_queue/release1/queue_status.md                              # progress
```

If preflight fails, the E jobs are skipped, because they depend on it. Run
`$H scripts/bundle_results.py --tag release1` and send the bundle back; do not fix code on RUN.

## Expected cost (rough; preflight's throughput measurement replaces this)

| | estimate |
|---|---|
| downloads | model at revision c5ebcd40 (cached already if the pilot cache has that revision) + 16k SAEs at layers 4, 8–20 (~0.3 GB each) |
| disk (cache root) | ~15–20 GB of activation and SAE caches; leave 40 GB free |
| GPU | ≤ 5 GB per job (model bf16 with the embedding on CPU, as in the pilot) |
| wall time | E0 ~0.5 h; E1_split ~1.5 h; E1_repair ~2 h; E2 ~1.5 h; E7 ~0.5 h; E6 ~1 h; ≈ 7 h in total |

## What to send back

- `bundles/release1_<stamp>.tar.gz` + `.sha256`, written by the last queue job: runs, preflight reports, queue state.
  It contains no caches.
- From R1–R3 (if not sent yet): `probe_R1.tar`, `golden/R2.tar`, `hf_agreement_R3.tar` with their `.sha256` files.
- A short report: start and end times, which jobs completed, failed or were skipped, any profile change (only
  `max_batch_tokens` may be lowered, in `profiles/local.yaml`), and anything unusual.

## Known limitations of this release

- The pilot's noise rows came from the pilot's own test-set order. E0's noise uses the same seeds over the harness's
  rows, so it is statistically, not bitwise, identical.
- E1 is split into two configs (split conditions on 32 cells; repair on the 13 Table B.2 cells) to keep run time
  bounded. Both are tagged exp_id E1.
- Figures are not produced yet (Phase 2, from tables/ only).

# Release 2 (run-20261103-2): E3 (RQ1 gate), E4, E5, E8 fill

Tag: `run-20261103-2`. It has the same code as `run-20261103-1` and also contains the release-1 `runs/` that the GPU machine committed (`ef95eaa`). Check out -2: checking out -1 would delete those tracked run directories from the working tree. DEV status: 165 tests pass (1 skipped). Every queued config dry-runs on the `run` profile within
the 7.0 GB cap (max stage peak 5.87 GB). Nothing in this release has run on the GPU box yet.

## What changed since release 1

| area | change |
|---|---|
| E3 | three fresh template families per task (harness/families.py), `<task>_e3` prompt sets, fit/eval family cells |
| E4 | several SAEs per (task, layer); matched-L0 widths from configs/sae_inventory.json; block-wise stage 2 for 131k-524k |
| E5 | `subthresh(α)` condition; sparse near-threshold codes kept in stage 2; greedy latents fitted on train pairs only |
| E8 | 36 cells with all three repair controls (other task, generic, 20 random seeds) in one run |
| memory | TransformerLens' unembed no longer copies the 1.1 GiB W_U on every call (bitwise-identical logits) |
| analysis (DEV) | harness/e8.py, scripts/make_e8.py, scripts/make_gate.py, scripts/make_figures.py |

Release-1 code paths are unchanged: E0-E7 configs run exactly as before.

## Queue (queue/release2.yaml)

prefetch → preflight → **E3** → E4 → E5 → E8 → bundle. Each experiment depends only on preflight.

| job | cells × conditions | what | rough time* |
|---|---|---|---|
| E3 | 6 × 196 | E2 conditions on fresh families: fit pilot → eval fresh, fit fresh → eval fresh (**the RQ1 gate**), fit fresh → eval pilot | 6-9 h (+ stage 1 for 16 200 new candidates) |
| E4 | 19 × 12 | OT L12 every width 16k-524k at matched + bracketing L0; RS L18 16k/65k; IOI L12 | ~1 h + stage 2 for large SAEs |
| E5 | 4 × 56 | subthreshold add-back, α ∈ {0.5 … 0.95}, ε_B / ε̄_pos slot, greedy k ∈ {1, 2, 5, 10} | ~1 h |
| E8 | 36 × 138 | repair specificity with every control on the full pilot grid + E6 layers | 10-17 h |

\*Scaled from release 1's measured 0.12-0.20 min per cell-condition at 150 pairs (E3's fresh-family cells have 450
test pairs). Preflight re-measures throughput; `plan_runs.py` then prints estimates. Total roughly 20-28 h, so run
it in tmux; the queue is safe to kill and restart.

## Decisions taken as defaults (the humans may change these before running)

- **E5 greedy objective is model-free**: least squares on ε_A − ε_B over train span rows. A model-based objective (train
  flip rate through patched forwards) would cost about k × candidates extra forward passes.
- **1m SAE is not queued** (`configs/experiments/E4_1m.yaml` exists; ~9 GB host RAM, block-wise on the GPU).
- **E8 keeps ranks 0, 1, 2, 4, 8.** Dropping 4 and 8 would save roughly a third of E8's time; the preregistered criterion
  needs only r ≤ 2.

Fixed by the humans on 2026-11-03 (DECISIONS.md): the RQ1 gate uses filter `f_pilot`, mean errors fitted on fresh
families, and is reported with the noise-control flip rate alongside.

## Run machine: exact steps

```bash
cd ~/Coding_folder/interpretability/PRISM
git status --short                               # clean; send it to DEV if not
git fetch origin --tags && git checkout run-20261103-2
H=.venv-harness/bin/python
export HL_PROFILE=run HF_TOKEN=hf_...
$H scripts/check_env.py                          # 0 errors (lockfile unchanged since release 1)
df -h "$($H -c 'from harness.profile import load_profile as l; print(l("run").root("cache_root"))')" ~/.cache/huggingface
#   needs ~35 GiB for new SAE downloads (two 524k = 18 GiB, two 262k = 9 GiB) plus activation caches: keep >= 60 GB
#   free. If there is not enough space, stop and report to DEV; do not delete caches.
$H scripts/plan_runs.py --queue queue/release2.yaml --profile run
$H scripts/run_queue.py --queue queue/release2.yaml
cat runs/_queue/release2/queue_status.md          # progress, from another pane
```

`golden/R2` must still be in place (preflight uses it). Only `max_batch_tokens` may be lowered, in `profiles/local.yaml`.

## What to send back

- `bundles/release2_<stamp>.tar.gz` + `.sha256` (written by the last queue job): runs, preflight reports, queue state,
  no caches. Use the same `results/...` branch recipe as before, splitting anything over 95 MB.
- R5 if not sent yet (RUN_REQUESTS.md).
- A short report: start/end times, jobs completed/failed/skipped, any profile change, `skipped_saes.json` in E4/E5 run
  directories if present, and anything unusual (OOM backoff lines in the job logs).

## Known limitations

- The W_U layout fix keeps logits bitwise identical on DEV (CPU, fp32 and bf16); on CUDA it is verified only by the
  preflight's golden and hook checks.
- E3's family-clustered CIs rest on 3 families per task: they are reported next to pair-level CIs, not instead of them.
- Cross-family `feat_A_meanPos` falls back to offset matching for frame words (coverage recorded per cell).

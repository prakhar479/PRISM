# Final exploratory run: runbook (last compute, 2026-11-18)

One Slurm job runs one queue, `queue/final-explore-server.yaml`, on **gnode084** (4 x RTX 2080 Ti), where the final pass left its
activation caches and DAS rotations. Everything here is **exploratory and post hoc** (designed after reading the final pass); it
cannot change any preregistered answer (DECISIONS.md 2026-11-18). Each experiment answers one open point of
`analysis/final_pass_results.md`.

| exp | question (from the final-pass results) | cells | est. GPU-h |
|---|---|---|---|
| E17 | DAS rank 1/2 failed their fits for running sum L18 (loss barely moved in 3 epochs): absent, or under-trained? Seeds 1-2, 10 and 30 epochs, lr 3e-2; rotation agreement across seeds | RS L18 | ~7 |
| E18 | Is the missing information higher-rank? Error-U up to r = 64, DAS up to 16; DAS on people-cities for the first time (E11: gap 13-34 points, r <= 4 closes <= 6 %) | RS L18, OT L12, PC L12, PC L16 | ~14 |
| E19 | Steering dose 0.25-16: does RS reach 50 % at some scale; does OT U beat random at small scale (random p95 was 58-90 % at x1)? | RS L18, OT L12, PC L12 | ~8.5 |
| E20 | Does the E9 picture (error-U ineffective, DAS feature-mediated) hold for wider SAEs at matched L0? | RS L18 65k, OT L12 65k / 262k | ~7 |
| E21 | Does the multilingual fidelity deficit (M1) shrink with a 4x wider SAE at matched L0? | 12 language sets + English, L12 and L18, 65k | ~6.5 |
| E22 | Is the subspace DAS finds in English the one the model uses in fr / hi / Hinglish? (descriptive: M2/M3 were untestable) | RS L18, PC L12, native vs English fit | ~8 |
| E23 | On identical sentences, native script and romanized text have the same SAE error size (FVU) but native script costs the model 1.5-4x more loss (deep analysis 2026-11-18): is that the error's DIRECTION or the context's general fragility? Norm-matched noise controls (x_hat + eta, x + eta, \|eta\| = \|eps\| per token) and per-token NLLs. E21 gets the same controls at 65k | 7 language sets + English, 16k, L12 and L18 | ~8 (+3 for E21's controls) |
| | **total** | | **~62 GPU-h, ~16-18 h on 4 GPUs** (estimate from the final pass's per-condition cost, +-50 %; 72 h limit) |

New downloads: ~8 GB (65k SAEs at L12 and L18, 262k at L12). Everything else is a cache hit from the final pass.

## Before (login node)

```bash
cd /home2/raveeshvyas1410/prism-run
git fetch origin --tags
git checkout <TAG>                       # the tag given with this release
git status --short                       # must print nothing (do NOT commit on the server; override sbatch options instead)
bash scripts/slurm_final.sbatch --check  # tag, clean tree, token, queue, sign-offs, golden fixtures, disk
```

**Do not run `--clean-full` on gnode084 before this run**: it would delete the caches (spans, codes, DAS rotations) this run reuses.
gnode074's caches are not used any more; `bash scripts/slurm_final.sbatch --clean-full` may be run there (on gnode074) at any time.

## Run

```bash
sbatch scripts/slurm_final.sbatch                                                  # defaults: gnode084, 4 GPUs, this queue
sbatch --export=ALL,PRISM_BULK=/scratch/$USER/prism scripts/slurm_final.sbatch     # if /ssd_scratch has < ~25 GiB free
```

Node or GPU count are changed on the command line (`sbatch --nodelist=... --gres=gpu:N ...`), never by editing and committing
the script on the server (that makes every run untagged).

## During

```bash
squeue -u $USER
tail -f slurm-prism-final-<jobid>.out
cat runs/_queue/final-explore-server/queue_status.md
```

Expect DAS training lines early (`DAS: training das1_e10 (k=1, 10 epochs, ...)`): E17 trains 18 rotations, E18 ~12, E19/E22 a few.
A 30-epoch rank-1 rotation takes about an hour. Time limit or cancel: resubmit the same command (resumes).

## After

```bash
git switch -c results/explore-$(git describe --tags --always)
git add -f results_server && git commit -m "final exploratory results"
git push -u origin HEAD
```

If a `*_fin` job failed, also `git add -f runs/E17 runs/E18 runs/E19 runs/E20 runs/E22 runs/E12 runs/_queue` before committing
(E21 and E23 write E12-style runs under `runs/E12/`; the per-token arrays make those run folders ~10-40 MB each).
Send DEV the branch name. After DEV confirms: `bash scripts/slurm_final.sbatch --clean-full` on gnode084.

## What DEV does with it

`ingest_published.py` -> `make_e12.py` (E21 rows sit next to the 16k rows, keyed by SAE) -> `make_final.py --only
E17,E18,E19,E20,E22` (E17_das_stability.csv, E18/E20/E22 illusion tables with the E9 rules, E19_steering.csv, E22_transfer.csv)
-> analysis write-up -> `make_results_branch.py`.

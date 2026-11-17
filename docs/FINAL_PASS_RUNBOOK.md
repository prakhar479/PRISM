# Final pass runbook (release 3 + final compute pass)

One Slurm job runs two queues back to back on gnode074 (3 x RTX 2080 Ti, where the release-2 activation caches live):

| queue | experiments | role | est. time |
|---|---|---|---|
| `queue/release3-server.yaml` | E12 SAE fidelity by language (9 jobs), E13 task behaviour by language (5 jobs) | exploratory, M1 + capability gates | a few hours (not measured on this node) |
| `queue/final-server.yaml` | E9 illusion checks + DAS, E10 steering (RS L18 confirmatory), E11 people-cities binding, E14 multilingual gap + M3, E4_1m; bonus E9b, E10b (fresh-template generalization), E2r (seed replicate) | see PREREGISTRATION_ADDENDUM.md | ~55 GPU-h, ~18 h on 3 GPUs (extrapolated from release 2, +-50%) |

Both queues are resumable: resubmitting the same job skips finished units. A failed queue does not stop the next one.

## Before (once, on the login node)

```bash
cd /home2/raveeshvyas1410/prism-run
git fetch origin --tags
git checkout <TAG>                       # the tag given with this release, e.g. run-20261117-1
git status --short                       # must print nothing
ls ~/.hf_token || (read -s -p "HF token: " T && echo "$T" > ~/.hf_token && chmod 600 ~/.hf_token)
bash scripts/slurm_final.sbatch --check  # tag, clean tree, token, queues, sign-offs, disk; no GPU work
```

Expect `checkout: <TAG>`, `HF token: found`, `template sign-offs: present` and no ERROR line. Do not delete
`/ssd_scratch/$USER/prism/cache` on gnode074 before this pass ends: it holds the release-2 caches the pass reuses.

## Run

```bash
sbatch scripts/slurm_final.sbatch                                          # release 3, then the final pass
# variants:
sbatch --export=ALL,QUEUES=queue/final-server.yaml scripts/slurm_final.sbatch        # final pass only (release 3 already done)
sbatch --nodelist=gnode077 --gres=gpu:4 scripts/slurm_final.sbatch                   # another node (no cache reuse: slower stage 1)
```

## During

```bash
squeue -u $USER
tail -f slurm-prism-final-<jobid>.out
cat runs/_queue/release3-server/queue_status.md      # per-unit status, attempts, durations
cat runs/_queue/final-server/queue_status.md
.venv-harness/bin/python scripts/storage_report.py   # disk; the queue pauses itself if the cache disk runs low
```

If the job hits the time limit or is cancelled: just `sbatch scripts/slurm_final.sbatch` again (same tag, same node).

## After

1. The job's last lines print a summary per queue (every experiment's run hash and status) and the push commands. Every row
   should say `complete`; none should say `SMOKE`.
2. Push the results as a branch (DEV fetches and analyses it):
   ```bash
   git switch -c results/final-$(git describe --tags --always)
   git add -f results_server
   git commit -m "final pass results"
   git push -u origin HEAD
   git switch --detach <TAG>
   ```
3. If any `*_fin` job failed (`queue_status.md`), the compute is still usable: also add the raw run directories before
   committing (`git add -f runs/E9 runs/E9b runs/E10 runs/E10b runs/E11 runs/E14 runs/E2r runs/E4 runs/_queue`), as for release 2;
   DEV finalizes from `parts/`.
4. Only after DEV confirms the results are complete: `bash scripts/slurm_final.sbatch --clean-full` on gnode074 frees the caches.

Send DEV: the results branch name, and the job's `.out` file if anything failed.

## What DEV does with it (pass 2, no GPU)

`inventory_runs.py` (completeness) -> ingest -> `make_e12.py`, `make_e13.py` (writes the Gate A/B table that E14 needs) ->
`make_final.py` (E9/E9b illusion, E10/E10b steering, E11/E14 specificity, E14 language gap + M3, E2r) -> `make_e4.py` (with
E4_1m) -> figures -> `make_results_branch.py`. Decision rules are frozen in PREREGISTRATION_ADDENDUM.md (2026-11-16).

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ERROR: working tree is dirty` | a file was added or edited in the checkout | `git status`; remove it (job logs are excluded automatically) |
| `HEAD is not a run-* tag` | checked out a branch | `git checkout <TAG>` |
| `HF_TOKEN not set` | no token | create `~/.hf_token` (Before) |
| queue says `paused: only N GiB free` | cache disk low | `storage_report.py`; free space; the queue resumes by itself |
| a unit `failed` after 2 attempts | look at `runs/_queue/<tag>/<unit>.log` | resubmit (failed units are retried); send DEV the log if it fails again |
| `*_fin` failed | finalize problem | push the raw runs (After, step 3); DEV finalizes from parts |

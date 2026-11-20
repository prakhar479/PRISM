# Final pass runbook: every remaining experiment in one run

One Slurm job runs one queue, `queue/complete-server.yaml`, on node-A (3 x RTX 2080 Ti, where the release-2 activation caches
live). The scheduler interleaves cheap script jobs with GPU experiments; confirmatory work has the highest priority.

| group | experiments | role | est. GPU-h |
|---|---|---|---|
| release 3 | E12 SAE fidelity by language (L12, L18; 9 jobs), E13 task behaviour by language (test split; 5 jobs) | M1 + capability gates | ~3 (not measured on this node) |
| release-3 extensions | E12 at L6 and L24 (5 jobs), E13 on the train split (5 jobs) | exploratory | ~2 |
| final pass | E9 illusion checks + DAS, E10 steering (RS L18 confirmatory), E11 people-cities binding, E14 multilingual gap + M3 | see PREREGISTRATION_ADDENDUM.md | ~40 |
| exploratory / redundant | E9b, E10b (fresh-template generalization), E16 (running-sum layer sweep with DAS), E2r (seed replicate) | exploratory | ~14 |
| **total** | | | **~59 GPU-h, ~20 h on 3 GPUs** (extrapolated from release 2, +-50%; 72 h limit) |

Not in this queue: E4_1m (two 1m SAEs, ~36 GiB of downloads): `queue/extra-1m-server.yaml`, optional, only where the disk allows.

The queue is resumable: resubmitting skips finished units. Units whose output depends on the code commit (stage 3, finalize, script
jobs, bundle, publish) are redone after a re-tag; caches (stage 1/2) are reused. Bundle and publish run again on every submit.

## Before (once, on the login node)

```bash
cd /home/user/prism-run
git fetch origin --tags
git checkout <TAG>                       # the tag given with this release, e.g. run-20261117-1
git status --short                       # must print nothing
ls ~/.hf_token || (read -s -p "HF token: " T && echo "$T" > ~/.hf_token && chmod 600 ~/.hf_token)
bash scripts/slurm_final.sbatch --check  # tag, clean tree, token, queues, sign-offs, disk; no GPU work
```

Expect `checkout: <TAG>`, `HF token: found`, `template sign-offs: present`, `golden fixtures: present` and no ERROR line. Do not
delete `/ssd_scratch/$USER/prism/cache` on node-A before this pass ends: it holds the release-2 caches the pass reuses.

Disk: the new caches need ~16 GiB beyond what is cached. The job checks the free space over all cache tiers after setup and
stops in seconds if it is below 25 GiB (instead of pausing for hours). If `/ssd_scratch` is short, add the HDD as a spill tier
before submitting: `export PRISM_BULK=/scratch/$USER/prism` (and `PRISM_MIGRATE_HF=1` once, to move the model download there).

## Run

```bash
sbatch scripts/slurm_final.sbatch                                          # the complete queue
# variants:
sbatch --export=ALL,PRISM_BULK=/scratch/$USER/prism scripts/slurm_final.sbatch      # with the HDD spill tier
sbatch --export=ALL,QUEUES=queue/extra-1m-server.yaml scripts/slurm_final.sbatch    # optional E4_1m, afterwards
sbatch --nodelist=node-B --gres=gpu:4 scripts/slurm_final.sbatch                  # another node (no cache reuse: slower stage 1)
```

## During

```bash
squeue -u $USER
tail -f slurm-prism-final-<jobid>.out
cat runs/_queue/complete-server/queue_status.md      # per-unit status, attempts, durations
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
   committing (`git add -f runs/E9 runs/E9b runs/E10 runs/E10b runs/E11 runs/E14 runs/E16 runs/E2r runs/E12 runs/E13 runs/_queue`), as for release 2;
   DEV finalizes from `parts/`.
4. Only after DEV confirms the results are complete: `bash scripts/slurm_final.sbatch --clean-full` on node-A frees the caches.

Send DEV: the results branch name, and the job's `.out` file if anything failed.

## What DEV does with it (pass 2, no GPU)

`inventory_runs.py` (completeness) -> ingest -> `make_e12.py`, `make_e13.py` (writes the Gate A/B table that E14 needs) ->
`make_final.py` (E9/E9b/E16 illusion, E10/E10b steering, E11/E14 specificity, E14 language gap + M3, E2r) -> figures -> `make_results_branch.py`. Decision rules are frozen in PREREGISTRATION_ADDENDUM.md (2026-11-16).

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ERROR: working tree is dirty` | a file was added or edited in the checkout | `git status`; remove it (job logs are excluded automatically) |
| `HEAD is not a run-* tag` | checked out a branch | `git checkout <TAG>` |
| `HF_TOKEN not set` | no token | create `~/.hf_token` (Before) |
| queue says `paused: only N GiB free` | cache disk low | `storage_report.py`; free space; the queue resumes by itself (it gives up after 60 min paused with nothing running) |
| `ERROR: only N GiB free for caches` at start | not enough disk | free space or set PRISM_BULK (Before), resubmit |
| a unit `failed` after 2 attempts | look at `runs/_queue/<tag>/<unit>.log` | resubmit (failed units are retried); send DEV the log if it fails again |
| `*_fin` failed | finalize problem | push the raw runs (After, step 3); DEV finalizes from parts |

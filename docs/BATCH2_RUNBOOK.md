# Batch 2 runbook (both branches)

What and why: docs/NEXT_BATCH_PLAN.md. Where the results go and how to read them: docs/FINDINGS_CATALOGUE.md.
Two branches, two accounts, each one Slurm job of about 36 h on 4 x RTX 2080 Ti (limit 96 h):

| branch | queue | runs | analysis |
|---|---|---|---|
| `batch2-atlas` | `queue/pilot-x1-server.yaml` | P20 (Gemma 3 depth atlas), P21 (Gemma 2 depth atlas), P22 (refusal atlas), P28 (width x sparsity grid), fidelity of every dictionary | `scripts/make_atlas.py` |
| `batch2-repair` | `queue/pilot-x2-server.yaml` | S_* (generic-text Fisher), T_klft_*, P23-P27 (sensitivity-subspace repair, decoder vs information, replicate on Gemma 3), fidelity of every add-on | `scripts/make_repair.py` |

Nothing in either branch depends on the other, or on the breadth pilot / second pilot still running. Both reuse downloads and caches
of their own account only.

## Before (once per account)

1. SSH key with write access to the repository (for `--record`); `~/.hf_token` from an account that accepted the licences of
   **google/gemma-2-2b, google/gemma-3-1b-pt and google/gemma-3-1b-it** (the atlas needs all three; the repair branch needs the first two).
2. Clone and check out the tag of the branch you run:
   ```bash
   git clone git@github.com:prakhar479/PRISM.git prism-b2 && cd prism-b2 && git checkout <run-tag>
   git archive origin/release golden_parts | tar x && echo 'golden_parts/' >> .git/info/exclude     # preflight check 5
   git status --short                                         # must print nothing
   bash scripts/slurm_batch2.sbatch --check
   ```
3. Disk: about 300 GB of caches plus 30-80 GB of downloads. Put them on the bulk disk if the fast one is small:
   `export PRISM_BULK=/scratch/$USER/prism` (and pass `--export=ALL,PRISM_BULK=...` to `sbatch`).

## Smoke, then run

```bash
sbatch --time=04:00:00 scripts/slurm_batch2.sbatch --smoke
grep -E "SMOKE (OK|FAIL)|smoke (finished|passed)" slurm-prism-b2-<jobid>.out       # every line must be OK
sbatch scripts/slurm_batch2.sbatch                                                  # resubmit the same command after a time limit
```
The smoke runs a subset of every experiment on the real models with tiny sizes and throwaway caches (about 1-2 h; it also does the
downloads, which the full run reuses). On the repair branch it runs two phases: the Fisher estimate and the KL fine-tunes first, then
one subset of each experiment covering every add-on kind (`var fis sens rnd`, their oracles, `lin clin mlp cmlp`), both models and the
fidelity job. If a line says `SMOKE FAIL`, send the tail of the log named on it.

Progress: `squeue -u $USER`, `tail -f slurm-prism-b2-<jobid>.out`, `cat runs/_queue/<tag>/queue_status.md`
(`<tag>` = `pilot-x1-server` or `pilot-x2-server`). A partial pass is useful: units run in priority order (atlas: P20, P22, P28, P21;
repair: P23, P27, P25, P26, P24), so a cut run lacks the least important cells first. Resubmitting skips finished units.

A **re-tag re-runs stage 3**: run hashes and the stage-3 markers include the commit (stage 1 / 2 caches do not). Keep one tag per
account for the whole batch; if you must move to a new tag, use `ONLY=<ids>` for the experiments that still need to run.

## After

```bash
.venv-harness/bin/python scripts/verify_pilot_runs.py --queue queue/pilot-x?-server.yaml      # provenance, config, coverage, nulls
bash scripts/slurm_batch2.sbatch --record                  # results branch results/b2-x?-server-<tag>, results_server/ committed
git push -u origin HEAD
```
Send the branch name. Then `--clean-full` only after the results are confirmed on DEV. Expect verify warnings, not failures, for:
cells skipped by the capability gate (Gemma 3 1B cannot do the running sum; none of these experiments uses it on Gemma 3), and tasks whose
base gap is below 0.05 (their closure is not computed).

On DEV:
```bash
python scripts/make_atlas.py  --runs <results_server/pilot-x1-server/runs> --out analysis/batch2_atlas
python scripts/make_repair.py --runs <results_server/pilot-x2-server/runs> --out analysis/batch2_repair
```
then update docs/FINDINGS_CATALOGUE.md (add rows; change statuses; never delete).

## What each experiment answers (one line)

- **P20 / P21** capture of the causal subspace by depth, task, sparsity (small / big L0) and width (16k / 65k), 3 DAS seeds each.
  **P28** 36 dictionaries at 4 depths: width vs sparsity. **P22** the same question for the refusal direction. Fidelity jobs give CE
  recovered, FVU, relative error, L0 and the splice / noise ratio of each dictionary, to ask which cheap metric predicts capture.
- **P23** gap closure of a k-dimensional linearly predicted error correction whose subspace comes from generic text only:
  by variance, sensitivity, both, or at random; oracle versions bound it. **P24** at 65k, **P25** across depth, **P26** on Gemma 3.
  **P27** is the missing part recoverable from the codes (decoder refit, network on codes) or only from x (network on x), and does the
  repair add to KL+MSE fine-tuning.

# PRISM run checkout (multi-GPU server)

This branch (`run`) holds only what a GPU server needs to run the experiment queue. It is generated from the
development branch `harness` by `scripts/make_run_branch.py`; `RUN_SOURCE.json` names the source commit. Do not
edit files here: a dirty tree stops the runs (put machine-specific overrides in `profiles/local.yaml`, git-ignored).

## Run release 2

```bash
git clone --branch run --single-branch <repo-url> prism-run && cd prism-run
git checkout <tag>                            # the run-* tag you were given
bash scripts/setup_server.sh                  # venv, locked packages, env check, golden fixtures, machine probe
export HL_PROFILE=server HF_TOKEN=hf_...      # gemma-2-2b is gated
H=.venv-harness/bin/python
$H scripts/plan_runs.py --queue queue/release2-server.yaml --profile server
tmux new -s prism
$H scripts/run_queue.py --queue queue/release2-server.yaml --gpus auto
```

Progress: `cat runs/_queue/release2-server/queue_status.md`. The queue is safe to kill and restart: finished units
are skipped. `--gpus 0,1` limits it to some GPUs; `--workers-per-gpu 2` puts two workers on each large GPU.

## Older GPUs (e.g. RTX 2080 / 2080 Ti, Turing, compute capability 7.5)

- Needs NVIDIA driver >= 580 (CUDA 13 wheels); `setup_server.sh` stops early if the driver is older.
- Turing has no bf16 tensor cores. The model still runs in bf16 (same numbers as the pilot and the golden export),
  but bf16 is emulated, so expect each GPU to be several times slower than an RTX 30xx/40xx/50xx. Setup step 5
  prints the measured fp32/bf16 GEMM speed per GPU (`runs/gpu_compat.json`). Do not switch to fp16: Gemma-2
  overflows in fp16, and a dtype change makes a different run.
- 8-11 GB per GPU: the embedding stays on the CPU, budgets are measured by preflight, and SAEs of 131k latents and
  up stay in host RAM and are streamed to the GPU in blocks (the queue serialises their loads against a RAM budget).

## What the harness does with the machine

- Measures every GPU (nvidia-smi), host RAM, CPU cores and disk first (`runs/_queue/<tag>/machine.json`).
- `profiles/server.yaml` sizes everything from that: memory cap per GPU, tokens and rows per forward, embedding
  placement, CPU threads per worker. Preflight then runs one real forward per GPU at that size and keeps the largest
  budget that fits (`gpus.json`); a GPU that fails is left out.
- Work is spread over all GPUs: stage-1 caches are filled once per shared prompt set, stage 2 is split into disjoint
  SAE groups, stage 3 runs one claiming worker per GPU (the RQ1 gate, E3, first), and a CPU finalize step writes each
  run's results. Every number in one run uses one pinned token budget.

## Send results back

The last queue job writes `bundles/release2-server_<stamp>.tar.gz` + `.sha256`. Return them on a results branch,
never by committing `runs/`:

```bash
git switch -c results/<tag>
mkdir -p incoming/RT2 && cp bundles/release2-server_*.tar.gz* incoming/RT2/
find incoming -type f -size +95M -exec sh -c 'split -b 90m -d "$1" "$1.part" && rm "$1"' _ {} \;
git add -f incoming/RT2 && git commit -m "release 2 (server) results" && git push -u origin results/<tag>
```
The bundle already contains the queue state (`machine.json`, `gpus.json`, `queue_status.md`) and the preflight reports.

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

Where things live: `setup_server.sh` puts the heavy intermediates on the scratch disk (`/ssd_scratch/$USER/prism`, or
`PRISM_WORK=<dir>`): the venv (`.venv-harness` is a link to it), the uv cache, Hugging Face downloads and the
activation caches (`profiles/local.yaml`, git-ignored, points the harness there). Everything finished stays in the
repo directory, which only needs ~1 GB: `runs/` (one folder per run with manifest, summary, per_pair.parquet),
`runs/_queue/<tag>/` (queue status, machine.json, gpus.json, logs), `preflight/` (reports) and `bundles/`.

Progress: `cat runs/_queue/release2-server/queue_status.md`. If the cache disk drops below a quarter of `min_free_gb`
(profiles/server.yaml, 10 GB, so 2.5 GB) the queue starts nothing new and prints `paused: ...`; free space and it
resumes by itself. The queue is safe to kill and restart: finished units
are skipped. `--gpus 0,1` limits it to some GPUs; `--workers-per-gpu 2` puts two workers on each large GPU.

## Older GPUs (RTX 20xx Turing cc 7.5, GTX 10xx Pascal cc 6.1)

- Needs NVIDIA driver >= 560 and GPUs of compute capability >= 5.0. `setup_server.sh` picks the torch build:
  `requirements/harness.lock` (torch 2.14.1+cu130) on driver >= 580 with Turing or newer;
  `requirements/harness-cu126.lock` (torch 2.14.1+cu126, kernels sm_50 to sm_90) on driver 560-579, or on any
  machine with a GPU older than Turing (CUDA 13 has no Pascal code). Both pin the same torch version and
  identical other packages; the build used is recorded in every manifest (`torch_cuda`).
- Neither generation has bf16 tensor cores. The model still runs in bf16 (same numbers as the pilot and the golden
  export, to bf16 tolerance; preflight check 5 tests it), but bf16 is emulated: expect each GPU to be several
  times slower than an RTX 30xx/40xx/50xx, Pascal slower than Turing. Setup step 5 measures fp32 and bf16 GEMM
  speed per GPU and checks a bf16 GEMM against fp32 (`runs/gpu_compat.json`). Do not switch to fp16: Gemma-2
  overflows in fp16, and a dtype change makes a different run.
- ~10-11 GB per GPU: the embedding stays on the CPU, budgets are measured by preflight, and SAEs of 131k latents and
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

The last two queue jobs write `bundles/release2-server_<stamp>.tar.gz` + `.sha256` (what DEV ingests) and
`results_server/release2-server/` (the same finished files, readable, nothing from `parts/` or the caches; files over
90 MB come as `.part-NN` pieces, `PUBLISHED.json` lists every file with its sha256 and the source commit). Both are
small. Push either on a results branch, never by committing `runs/`:

```bash
git switch -c results/<tag>
git add -f results_server/release2-server && git commit -m "release 2 (server) results" && git push -u origin results/<tag>
git switch --detach <run tag>                 # back to the run checkout
```
Or push the bundle instead (what `scripts/ingest_results.py` expects):

```bash
git switch -c results/<tag>
mkdir -p incoming/RT2 && cp bundles/release2-server_*.tar.gz* incoming/RT2/
find incoming -type f -size +95M -exec sh -c 'split -b 90m -d "$1" "$1.part" && rm "$1"' _ {} \;
git add -f incoming/RT2 && git commit -m "release 2 (server) results" && git push -u origin results/<tag>
git switch --detach <run tag>
```
The bundle already contains the queue state (`machine.json`, `gpus.json`, `queue_status.md`) and the preflight reports.

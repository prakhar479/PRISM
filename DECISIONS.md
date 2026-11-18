# DECISIONS (harness)

Dated log of non-trivial choices. Pilot run verdicts live in `results/*/DECISIONS.md`; this file is about the harness.

## 2026-10-30 (Phase 0)

- **Pilot code = `src/` at HEAD**, identical to `430aa27`, which produced `results/paper_20261026_001341`. It is frozen
  as the reference, and the new code goes in `harness/` (proposed; see docs/IMPLEMENTATION_PLAN.md).
- **Hook site recorded:** `blocks.{L}.hook_resid_post` (TransformerLens, `from_pretrained_no_processing`). Model
  runs in bf16 on CUDA. SAE weights are fp32 below 524k. D uses post-softcap float32 last-position logits.
- **Golden export** (`scripts/export_golden.py`):
  - The 16 golden test pairs are drawn with `np.random.default_rng(20261030)` from the pilot test order.
  - Sparse z keeps entries with `z > 0` and `z >= 0.5·threshold` (E5's smallest α is 0.5).
  - B's residual is stored only from `first_diff` on, because earlier positions equal A's; the stored
    `prefix_A_minus_B_maxabs` checks this.
  - Condition logits are stored as candidate-set logits + logsumexp + top-50. Full-vocab logits are kept only for
    clean A/B, to stay under 200 MB.
  - Tolerances used there are provisional and become central in `harness/tolerances.py` in Phase 1:
    - D vs pilot: 0.05;
    - batch-1 vs batched D: 0.25 (bf16);
    - U columns: |cos| > 0.999;
    - NLDR: ±1 point.
- **HF agreement** (`scripts/hf_agreement.py`): "near-tie" = disagreement whose top-1 margin is < 0.25 logits in
  either model. HF runs eager attention, same dtype, batch 1, `logits_to_keep=1`.

## 2026-10-30 (Phase 1)

- **Environment.** `requirements/harness.{in,lock}` pin the stack: torch 2.14.1+cu130, transformers 5.19.0,
  transformer-lens 3.9.0, numpy 2.5.4 (Python 3.12). The humans report that DEV and RUN share one environment.
  RUN's freeze (R1) is checked against this lock by `check_env.py` in preflight. On RUN the harness gets its own
  `.venv-harness`, so the pilot environment (needed for R2/R3) is not modified.
- **Pilot scripts untouched.** The new tools are `scripts/run_preflight.py` and `scripts/run_prefetch.py`. The
  pilot's `preflight.py` hashes its own path into its fingerprint, and `run_server_gpu.sh` calls both pilot scripts
  by name.
- **Reuse, not copies.** Pilot math is imported from the frozen `src/`: JumpReLU loading, `fit_unsupervised`,
  `fit_supervised`, `random_subspace`, `fit_cov_eig`, `rescaled_noise`, generators, tokenization, `valid_mask`,
  forward primitives. The harness bootstrap with seed 0 and 1000 resamples uses the pilot's index matrix, so pilot
  CIs are reproduced exactly (tested).
- **SAE choice.** "Canonical" = the `average_l0` variant closest to 100. This matches all 12 SAEs the pilot
  resolved through sae_lens (tested), so layers the pilot never used (9, 11, 13, 15) are resolved the same way.
  Revision `fd571b47` of `google/gemma-scope-2b-pt-res`.
- **Model revision.** `google/gemma-2-2b@c5ebcd40` (main on 2026-10-30). The pilot used TL's default (main at run
  time); R1's cache listing must show the same commit.
- **Pilot pairs.** The candidate pools are rebuilt unfiltered in pilot order. The pilot's 300/150 are the first n
  candidates passing `f_pilot`. R2's dumps verify this (T1).
- **Noise.**
  - `feat_B_noise` keeps the pilot definition: norm-matched to ε_A, seed + 1000·s + L.
  - `feat_A_noise` is matched to ε_B, the error it replaces.
  - `feat_A_covnoise` draws N(0, Σ_train ε) and rescales each row to ‖ε_B‖, so only the direction distribution
    differs from `feat_A_noise`.
- **Mean errors.** ε̄_task and ε̄_pos are fitted on the patched positions of train pairs, A and B rows pooled. A
  row's key is its semantic slot if it has one, else its offset from the final token; unknown keys fall back to
  offset, then to the task mean. Coverage is recorded per cell.
- **Resampled errors.** The donor C is a prompt of another test pair with the same family and same length whose
  span covers the pair's patched positions. `sameAns`: answer = A's. `diffAns`: answer ∉ {A's, B's}, else ≠ A's
  ("relaxed"). Pairs with no donor are flagged `none` and excluded from summaries.
- **Metrics.**
  - NLDR and probability recovery are reported as a ratio of means (the pilot convention) and as a pair mean.
  - Gap = M(full_A) − M(condition), using the measured full_A.
  - Flip rate is computed over pairs whose clean B top-1 is B's answer.
  - The B baseline is the unpatched B run through the same stage-3 path; equality with the stage-1 clean B is
    checked per cell (`hook_checks`).
- **Tiny-model runs** live only under `tests/_tiny_runs`, set every filter flag True (a random model passes no
  behavioural filter), and are refused by both bundle and ingest.
- **Bundles** are `.tar.gz`, not `.tar.zst`: zstd is not in Python 3.12's stdlib, and gzip avoids an extra
  dependency.
- **Memory estimates** use the pilot's 256 KiB/token activation heuristic plus fp32 final-position logits. Stage 2
  runs without the model, so ≤ 262k fp32 and 524k bf16 SAEs are planned as GPU-resident. 1m is planned block-wise.

## 2026-10-31: E3 fresh template families (release 2)

- **Families** (`harness/families.py`): three per task, generated in place of the pilot's surface only.
  - Object tracking: `ot_placed` (hat, shoe, …; boxes H–P; "is placed in box" / "goes to box"), `ot_bin` (spoon,
    sock, …; bins Q–W; "sits in bin" / "is moved into bin" / "is now in bin"), `ot_drawer` (shell, stone, …; drawers
    A–G; "There is a … in drawer" / "Someone puts the … into drawer" / "So the … is in drawer"). No pilot nouns. Labels
    are single letters because Gemma tokenizes " 3" as two tokens. `ot_drawer` reuses the pilot's labels A–G (only
    X–Z are left unused); its nouns, frames and container word are new.
  - Running sum: `rs_plus` ("Begin at / Starting value / Initially … Plus … Answer: / Final value: / We get:"),
    `rs_symbolic` ("[Compute: | Evaluate:] 3 + 2 + 1 + 1 = | equals | gives"), `rs_story` (new names and items,
    "had … Then … found … more. In the end … had").
  - A single fixed phrasing has only a few hundred distinct running-sum prompts, which is too few for prompt-disjoint
    900/450 pools (the first build had 0 test pairs). So, like the pilot's calc template (3 × 3), each family varies
    its opener and closer.
  - Unchanged from the pilot: pair construction, difficulty (gen_kwargs), n_shots (8, drawn from the family's own
    generator, so prefixes differ from the pilot's), answer format, candidate multiplier, filter mode. The seed per
    family is `seed + 1000·(crc32(family) mod 997 + 1)`.
- **Sets**: `<task>_e3.jsonl.gz` = the pilot set's rows, byte-identical (tested), plus the family rows. A run refuses a
  set with fresh rows unless every cell names `fit_families` and `eval_families`.
- **Selection**: all pilot families form one group in the pilot's own rank order, so `[pilot]` selects exactly the
  pilot pairs. Fit sets take `n_train` in total, round-robin over the families (100 per fresh family). Evaluation takes
  `n_test` per family (450 fresh test pairs and 150 pilot pairs).
- **Cross-family mean errors**: slot names match across families. Patched tokens outside a slot (frame words) fall
  back to the offset from the final token, which is approximate across frames of different lengths. Coverage is
  recorded per cell.
- **Open for the humans**: the preregistration does not say whether the RQ1 gate's `feat_A_meanPos` uses mean errors
  fitted on pilot or on fresh families. E3 produces both (`fit_families` pilot → fresh, fresh → fresh). Three families
  per task make a cluster bootstrap coarse; DEV reports pair-level and family-clustered CIs side by side.
- **`fit_families` column**: optional in `per_pair.parquet` (absent in release-1 outputs; readers default it to
  `pilot`). It is part of the cell key in analysis and of the duplicate check.

## 2026-11-01: preflight fixes after the first RUN attempt (run-20261031-1)

- **CUDA device index.** `torch.cuda.set_per_process_memory_fraction` rejects an index-less `cuda` device, so both
  preflight smoke runs (and every E job) crashed in `cap_memory`. `resolve_device` now returns `cuda:<current>`, and
  `cap_memory` resolves an index itself. A regression test stubs the CUDA calls.
- **Check 3 tolerance (loosened, with evidence).** "head == full forward last logits" failed at 0.19 against 1e-2.
  `runtime.head` applies the soft-cap in fp32 on the last position; TransformerLens' full forward applies it in the
  model dtype. On DEV with the tiny Gemma-2 at Gemma-like logit magnitudes, they agree to ~1e-5 in fp32 and differ by
  0.122 in bf16, where one bf16 ulp at |logit| 16–32 is 0.125. New tolerance: 2 ulp of the model dtype at the largest
  |logit|, plus equal argmax. fp32 is still held to 1e-3 on RUN and 1e-4 in the DEV test.
- **Check 5 tolerance (loosened, with evidence).** Span residuals vs the golden export failed at 0.0201 (max relative
  row error) against 0.02. The harness batches the 16 golden pairs together; the pilot cached those rows from a
  different batch, and bf16 kernels round differently by batch shape. A hook-site or position error would be O(10–100%).
  All functional checks passed on RUN (hook identity exact, x_hat 1e-4, per-condition D vs golden 0). New tolerance:
  0.05. Preflight now also reports the pilot's own batch-1 vs batched spread on the same rows, as context.
- **Seen on RUN, not fixed yet.** During check 5, ~1126 MiB allocations hit OOM and the token budget backed off
  (2048 → 512). The size equals a bf16 copy of the 256000 × 2304 unembedding, so something makes a transient copy of
  W_U. The run recovered; it should be found before larger SAEs share the GPU (E4).

## 2026-11-02: preflight Check 4 bf16 tolerance fix (run-20261102)

- **Check 4 tolerance (loosened, with evidence).** "hook identity (full_B == clean_B, resumed clean_B == stage 1)"
  and "full_A reproduces clean A" failed on RUN at max |dD| full_B 0.081, resumed clean_B 0.12, full_A 0.18 against
  1e-3. The 1e-3 tolerance was a holdover from fp32 CPU testing (where atol=1e-4 passes). On GPU in bf16, Gemma-2-2b
  last logits reach magnitudes ~28, where 1 bf16 ulp is 0.125 and 2 ulps is 0.25. Stage 1 executes an unbroken 26-layer
  forward pass, while stage 3 breaks execution at layer 12 and resumes layers 13..25. GPU non-determinism and kernel
  rounding across 13 bf16 transformer layers accumulate 0.5 to 1.5 ulps difference (0.08–0.18). The golden export
  itself (`scripts/export_golden.py:451,455`) and Check 5 use 0.25 (`D_BATCHING.atol`) for the exact same checks.
  New tolerance: `HOOK_IDENTITY_D` atol is loosened from 1e-3 to 0.25 (2 bf16 ulps, matching `D_BATCHING.atol`); fp32
  remains at 1e-3 (`hook_identity_atol`).

## 2026-11-03: RQ1 gate filter (humans' decision)

- **The RQ1 gate uses filter `f_pilot`** (decided by the humans on 2026-11-03, before any E3 result). The preregistration
  names the metric, condition, threshold and fresh families but not the filter. On the pilot templates the RS L18
  quantity met the criterion under `f_A_correct` (11.8 points [7.4, 16.4]) but not under `f_pilot` (8.7 [4.0, 13.3]), so
  the choice had to be fixed before E3. `scripts/make_gate.py` marks `f_pilot` rows on fresh families as the gate and
  reports `f_A_correct` / `f_none` as sensitivity rows. `PREREGISTRATION.md` is not frozen yet (placeholder date); adding
  this line there is the humans' edit.
- **The gate's `feat_A_meanPos` uses mean errors fitted on fresh families** (decided by the humans on 2026-11-03,
  before any E3 result): the E3 cells `fit_families: [fresh], eval_families: [fresh]`. The pilot-fitted cells evaluated
  on fresh families are reported as sensitivity rows.
- **The gate table reports the noise control alongside every row** (decided by the humans on 2026-11-03): the flip rate of
  `feat_B_noise` (x̂_B + η) in the same cell and filter, averaged over the noise seeds. On the OT L12 pilot templates it
  is 88% (99% at L8): destroying B's information alone often gives A's answer, so the OT flip rate is read next to it.
  The gate criterion itself is unchanged.

## 2026-11-03: E4 matched-L0 widths, E5 subthreshold add-back, W_U copy (release 2, branch release2-e4-e5)

- **SAE inventory.** I re-listed layers 12 and 18 from `google/gemma-scope-2b-pt-res` at revision `fd571b47c1c6`;
  `configs/sae_inventory.json` matches exactly. Layer 12 has 16k–1m (5–6 L0 variants each); layer 18 has only 16k and
  65k.
- **L0 matching rule (E4, E5).** The target is the pilot's 16k variant at the cell's layer (canonical, L0 closest to
  100: L12 → 82, L18 → 74). For every other width, `match_l0` takes the variant closest in |log L0|.
  - Primary task (OT L12): also the nearest variants below and above the target (bracketing). At L12 the closest
    variant of every wider width lies below 82 (76, 72, 67, 67, 65, 73), so bracketing adds the next one above (155,
    141, 129, 121, 115, 107).
  - Secondary tasks (RS L18, IOI L12): closest only. RS L18 can only compare 16k with 65k (L0 64).
  - The configs are generated by `scripts/make_e4_config.py` from the inventory. A test fails if the committed YAML
    differs from the script's output.
- **Roles.** Only the 16k cells of OT L12 and RS L18 are confirmatory (PREREGISTRATION). All other widths and IOI are
  exploratory. Width persistence, at matched realized L0 within 25%, is then analysed on DEV.
- **Realized L0** is recorded per cell on the patched test positions (`mean_l0_test`, as before) and on the generic
  set (`generic_l0: true`: 100 docs × 124 tokens from pile10k, about 12k rows, plus the error energy
  Σ‖ε‖²/Σ‖x‖²). The generic stage-2 pass reuses the existing generic path.
- **Placement and memory (run profile, 7.0 GB cap, max_batch_tokens 1024).**
  - 16k–65k: GPU-resident (stage-2 peak ≤ 5.9 GB).
  - 131k–524k: block-wise on the GPU from CPU weights (57344 latents per block for fp32 131k/262k, 115712 for bf16
    524k/1m; peak ≤ 3.4 GB). SAEs ≥ 524k are held in bf16 (pilot rule).
  - Stage 2 never runs with the model loaded: `run_stage2` frees it if present, and a test asserts it.
  - 1m is in a separate `E4_1m.yaml`, to run only if time permits. It needs ~9 GB of host RAM. Any SAE whose
    weights × 1.25 exceed the available host RAM is skipped with a logged reason (`<run>/skipped_saes.json`) and its
    cells are skipped instead of failing the run. I planned 1m block-wise on the GPU rather than CPU-only: without the
    model the GPU is free, and CPU-only encoding of 1m on ~60k rows would take hours.
- **E5 semantics.**
  - `subthresh(α)` = x̂_A + Σ_{α·θ_i ≤ z_i < θ_i, z_i > 0} z_i W_dec[i] + error slot (ε_B, or ε̄_pos fitted on train).
  - α grid {0.5, 0.7, 0.8, 0.9, 0.95}; α = 1 adds nothing (tested equal to `feat_A_errB`).
  - Stage 2 stores only A's inactive near-threshold entries down to α_min = 0.5, plus a decoder table with the W_dec
    rows of the latents that occur. Stage 3 builds every variant on CPU without the SAE.
  - Per-pair key: condition `subthresh`; estimator `all` or `greedy`; source `alpha=<a>|slot=<B|meanPos>`; r = greedy
    k (0 = all latents).
  - E5's wide SAE is 262k at L12 (matched L0 67) and, at L18 (no 262k), the widest available: 65k (L0 64).
- **Greedy objective (E5).** I chose a model-free one, fitted on train pairs only (`FitSet`-guarded): forward
  selection of k ≤ 10 latents minimising Σ_train rows ‖(ε_A − ε_B) − add-back_L‖² at α = 0.5. That moves
  `feat_A_errB` closest to x_A in the residual stream. Each latent is added with its actual z, not a fitted
  coefficient, and selection stops early when no latent reduces the objective. The picked ids, the objective trace
  and the candidate count are saved per cell. The set is evaluated at k ∈ {1, 2, 5, 10} at every α.
  **Open for the humans:** a model-based objective (train flip rate through patched forwards) would be closer to the
  primary metric, but it costs about k × candidates forward passes.
- **W_U copy (fixed).** TransformerLens' `Unembed.forward` calls `F.linear(x, W_U.T.contiguous(), b_U)`, which copies
  the whole 256000 × 2304 unembedding on every call: 1125 MiB in bf16, rounded to the 1126 MiB seen on RUN.
  `harness.model.vocab_major_unembed` stores W_U as the transpose view of a contiguous [d_vocab, d_model] tensor (one
  copy on CPU at load time), so `.contiguous()` is a no-op. Logits are bitwise identical (fp32 and bf16, tiny-model
  test). `Module.to` keeps the strides; `load_gemma` warns if they are lost.
  - Not yet seen on the GPU. The pilot code path (`src/`, used by `export_golden.py` and `hf_agreement.py`) still
    copies; it is frozen.

## 2026-11-03: multi-GPU server runs and the clean run branch

- **Measured, not typed.** `profiles/server.yaml` sets `auto` for the GPU memory cap (total − 0.6 GiB), tokens and
  rows per forward, embedding placement (CPU only under 16 GiB) and CPU threads (physical cores ÷ concurrent
  workers); `harness.memory.resolve_resources` fills them from `harness.device.probe_machine` (nvidia-smi + /proc, no
  CUDA initialisation in the scheduler). The batch size gives 40% of the memory left after the model to one forward
  (pilot BYTES_PER_TOKEN heuristic, 1024-32768 tokens, 64-256 rows). The heuristic underestimates on the laptop, so
  preflight check 9 runs one real forward per GPU at that size, halving on OOM, and the scheduler pins the smallest
  measured budget per experiment (HL_MAX_BATCH_TOKENS). `profiles/run.yaml` (laptop) is unchanged.
- **One budget per experiment.** All units of one run use the pinned budget (each `.done` records the effective
  budget and OOM backoffs). `run_hash` still excludes profile and device: server results are a new run of the same
  experiment, comparable to the laptop's at bf16-batching tolerance.
- **No concurrent cache writers by design, locks as a guard.** Shared stage-1 entries are filled by scheduled
  warm-up jobs (E3 sets; the pilot sets via E8, whose layers cover E4/E5); stage 2 runs in disjoint SAE slices. Cache
  files are written to per-process temp names and published with os.replace; per-entry flock serialises writers
  (lock order = key order; only a warning on network filesystems, where flock is unreliable).
- **Stage 3 by dynamic claims, not fixed shards.** Workers claim 8-condition units with O_EXCL claim files (a dead
  worker's claim is taken over); idle time is at most one unit. Only `--finalize-only` (CPU) writes a run's
  `manifest.json`, `per_pair.parquet` and summaries, from units with `.done` only; partial processes write
  `manifest.<w|s2|stage>.json`. Tested: 3 concurrent workers + finalize give per-pair rows identical to one
  sequential run (tiny model, fp32 CPU).
- **Host RAM.** Workers reserve bytes from a machine-wide budget (80% of MemAvailable at scheduler start) before an
  SAE or model load; a short dip waits instead of being recorded as a skipped SAE. Only an SAE that cannot fit the
  whole budget is skipped.
- **Queues opt in.** Only queues with `parallel: true` use the multi-GPU scheduler; older queues run one job at a
  time in order with an unchanged environment, as before.
- **Run branch.** `scripts/make_run_branch.py` builds `run` from an allow list (harness, src, configs, data, profiles,
  queue, lockfile, run scripts) with git plumbing, adds `RUN_SOURCE.json` (source commit) and the golden fixtures as
  < 95 MB parts, and signs the commit and a `run-*` tag. Tests, results, runs, paper and analysis scripts stay on
  `harness`. Results from the server carry the run-branch commit, which is tagged, so DEV ingest accepts them.

## 2026-11-04: golden R2 accepted; Turing (RTX 20xx) servers

- R2 golden export (`incoming/R2.tar`, sha256 cfec8740...18e4) arrived: 20/20 checks PASS, the 11 pilot conditions of
  entity_track L12 reproduce the pilot to 0.00 points, refit U columns |cos| = 1.000000. Exported on the laptop at
  5ac8ef58d5 with `scripts/export_golden.py` modified locally (the inference-mode OOM fix later committed); the
  fixture content does not depend on that change. It goes into the run branch as `golden_parts/`.
- The server GPUs may be RTX 2080 class (compute capability 7.5). The model stays bf16: it is what the pilot,
  the golden export and run_hash use, and Gemma-2 overflows in fp16; fp32 does not fit 8-11 GB with activations.
  Turing runs bf16 without tensor cores (slower, same numbers to bf16 tolerance; preflight check 5 tests that).
- `setup_server.sh` now stops on a driver older than 580 (CUDA 13 wheels) and runs `harness.device.gpu_compat`
  per GPU: kernels for the GPU's arch in the torch build, bf16 GEMM vs fp32, and fp32/bf16 GEMM TFLOPS
  (`runs/gpu_compat.json`). No fallback torch build: a different CUDA build would differ from the lock.

## 2026-11-04: the "multi-GPU server" is 4 x GTX 1080 Ti (Pascal, sm_61)

- Setup on the server (`incoming/machine.json`, `incoming/gpu_compat.json`): 4 x GTX 1080 Ti 11 GiB, driver
  580.178.04, 126 GiB RAM, 40 logical CPUs of which this process may use 16. torch 2.14.1+cu130 saw no usable GPU
  (`gpus: []`): CUDA 13 has no Pascal kernels (minimum sm_75). Nothing ran on the GPUs; no numbers exist from it.
- Pascal also has no bf16 hardware and no tensor cores; fp32 Gemma-2-2B does not fit 11 GiB with activations.
- Fixes: the probe counts physical cores only inside the CPU affinity set (it reported 20, the process may use 16
  logical CPUs) and measures disk at the nearest existing parent of cache_root (it reported null); setup stops
  before installing on GPUs older than sm_75 and records the torch arch list in gpu_compat.json.
- Running on this server needs a torch build with sm_61 kernels (a CUDA 12.x build), i.e. a change to the locked
  environment: a human decision, not taken here.

## 2026-11-11: CUDA 12.6 build of the same torch for driver-570 servers (4 x RTX 2080 Ti)

- The 2080 Ti server (gnode047) has driver 570.211.01, which runs CUDA <= 12.8, not the CUDA 13 build in
  `requirements/harness.lock`. The user asked to make the harness run there.
- Added `requirements/harness-cu126.lock`: torch 2.14.1+cu126 (same version; download.pytorch.org publishes
  2.14.1 for cu126 and cu130 only, not cu128) and the NVIDIA CUDA 12.6 libraries; every non-CUDA package is
  pinned to exactly the version in harness.lock (compiled with those pins as constraints; a test checks it).
- `setup_server.sh` chooses the lock from the driver (>= 580: harness.lock; 560-579: harness-cu126.lock; older:
  stop) and installs with `uv pip sync` (exact set). `check_env.py` and preflight check 1 compare against the lock
  matching the installed torch build. run_hash is unchanged (it does not include the CUDA build); provenance
  carries it in each manifest's env (`torch_cuda`).
- Numbers from the two builds can differ at the level of kernel choice in bf16, which is what preflight checks
  3-5 (golden comparison, bf16 tolerances) already test on each machine. Not yet verified on the 2080 Ti.

## 2026-11-12: GTX 1080 Ti (Pascal, cc 6.1) through the CUDA 12.6 build

- The user asked to run on the 4 x GTX 1080 Ti server (driver 570.211.01). torch 2.14.1+cu126 (the wheel in
  harness-cu126.lock, sha256 75727245...f86c) is compiled for sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90 (read from
  libtorch_python.so); sm_60 cubins run on sm_61. The build has no Ampere-only guard on bf16 GEMM; only SDPA
  backends and sparse bf16 ops require sm_80, and neither the harness nor TransformerLens uses them.
- `setup_server.sh` now sends any machine with a GPU below cc 7.5 to harness-cu126.lock (whatever the driver,
  >= 560) and refuses only below cc 5.0. `harness.device.kernels_for` replaces the exact-match arch test in
  `gpu_compat` (which would have failed sm_61 against sm_60 code).
- Not verified on hardware: whether cuBLAS runs bf16 GEMM on Pascal, its accuracy and speed. Setup step 5
  (`gpu_compat`) tests exactly that and stops setup if it fails; preflight checks 3-5 then test the model numbers
  against the golden export. Pascal has no tensor cores: expect it to be the slowest machine so far.

## 2026-11-12: heavy data on a scratch disk (server with ~30 GB home and /ssd_scratch)

- The 1080 Ti / 2080 Ti server has ~30 GB in the home directory and an SSD at /ssd_scratch; release 2 needs ~35 GiB
  of SAE downloads plus the model, activation caches, runs and a ~8 GB venv.
- `setup_server.sh` now uses `PRISM_WORK` or `/ssd_scratch/$USER/prism` (PRISM_SCRATCH=<mount> to look elsewhere)
  for: the venv (real dir there, `.venv-harness` is a symlink, added to `.git/info/exclude` so git status stays
  clean), `UV_CACHE_DIR` and `TMPDIR`, and `profiles/local.yaml` (git-ignored; written only if absent or written by
  the script) with absolute `cache_root`, `runs_root`, `hf_home`. No scratch disk: unchanged behaviour with a warning.
- `run_queue.py` and `run_prefetch.py` now call `profile.apply_env()` so `HF_HOME` reaches every child process
  (prefetch used to download into `~/.cache/huggingface` regardless of the profile). Bundles are written next to
  `runs_root` when it is outside the repo (`bundle_results.default_out_dir`). `gpu_compat.json` and the disk probe use
  the profile roots. run_hash is unaffected (it excludes profile roots).
- An existing real `.venv-harness` in the home directory is not deleted: setup stops and asks the user to remove it.

## 2026-11-12: GPU numbering = PCI bus order (mixed 3 x RTX 2080 Ti + 1 x RTX 3080 server, driver 580)

- First setup on the 4-GPU server: `gpu_compat.json` listed the RTX 3080 as GPU 0, `machine.json`/nvidia-smi as GPU 2.
  torch numbers fastest-first by default, nvidia-smi by PCI bus, and the scheduler hands nvidia-smi indices to
  children as CUDA_VISIBLE_DEVICES: on a mixed machine "GPU 2" would have been a different card than the one the
  probe, `gpus.json` budgets and failure list named.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is now set (setdefault) in scripts/_bootstrap.py, harness/run.py and setup step 5;
  `gpu_compat` also records each GPU's uuid. No effect on single-GPU or identical-GPU machines.
- Same run: all four GPUs ok on torch 2.14.1+cu130 (arch list sm_75 ... sm_120). bf16 GEMM error 0.0029 against fp32.
  2080 Ti: fp32 ~10.5 TFLOPS, bf16 ~5.9 (emulated, slower than fp32); 3080: fp32 21.3, bf16 52.7. Stage-3 workers claim
  units dynamically, so the 3080 takes proportionally more; stage-2 slices are static and equal. The model stays bf16
  (pilot / golden / run_hash); running Turing in fp32 would be a numerics change and a human decision.

## 2026-11-13: caches on scratch, finished results in the repo directory; disk threshold per profile

- First server preflight (4 GPUs, all checks PASS incl. the golden comparison) failed only on `RAM and disk`: 19-20 GB
  free at the cache root against a fixed 20 GB, after 49 GB of downloads on the scratch disk.
- Run directories are small (release 1: per_pair.parquet <= 4 MB, parts/ the same data again); the big things are the
  Hugging Face downloads and the activation caches. So only those (and the venv, uv cache) stay on scratch:
  `profiles/local.yaml` written by setup now sets `cache_root` and `hf_home` only; `runs_root`, queue state
  (`runs/_queue/<tag>`), preflight reports and bundles stay in the repo directory.
- `min_free_gb` is a profile key (default 20, unchanged for the laptop; 10 in profiles/server.yaml). Preflight needs
  more than that free at cache_root and 2 GB at runs_root. The real cache need of release 2 was not measured (the 60 GB
  total in the notes was an estimate: 35 GB downloads + ~25 GB caches), so the queue now watches the cache disk:
  below min_free_gb / 4 it starts no new unit ("paused: ...", running units continue) and resumes by itself. Units are
  resumable, so a full disk costs at most the running units, never finished work.
- Numeric tolerances (preflight 3-5) are unchanged: every one passed on the 2080 Ti / 3080 with margin (span residuals
  0.031 vs 0.05; per-condition |dD| <= 0.134 vs 0.25).
- New last queue job `publish` (scripts/publish_results.py): copies the same finished files as the bundle (never
  parts/ or caches) to `results_server/<tag>/` with PUBLISHED.json (sha256 + source commit), files over 90 MB split;
  git-ignored so the run checkout stays clean, committed with `git add -f` on a results branch. The bundle is kept:
  it is what scripts/ingest_results.py validates.

## 2026-11-14: tiered storage (fast SSD, bulk HDD, home)
- `/ssd_scratch` had 14 GiB free after the 35 GB download, too little for the caches. Roles now follow access pattern:
  the Hugging Face download is read once per process at model load (the page cache serves repeats) so it goes to the
  bulk disk (`/scratch`, HDD); the hot activation caches (re-read by every stage-3 worker) use the SSD first and
  `Profile.cache_spill` takes only the entries that no longer fit (`harness/cache.py: Cache.locate`). One entry lives in
  one tier, found by every process by looking for `<tier>/<kind>/<key>`; a complete copy wins; a partial entry on a
  disk now below `cache_spill_min_gb` (4) restarts on the next tier. Nothing is moved or evicted under a reader. Entry
  keys and manifests are unchanged (`rel()` stays `<kind>/<key>`); results do not depend on the tier.
- Preflight check 1 counts free space summed over the cache tiers; the queue pauses only when every tier is below
  min_free_gb / 4 and logs `disk: ...` every 10 minutes. `scripts/storage_report.py` shows disks (SSD/HDD from sysfs,
  optional speed test), what the harness keeps on each, advice, `--watch N`, and `--write-profile`. `setup_server.sh`
  detects `/scratch` (or PRISM_BULK), writes cache_spill and hf_home, and moves an existing download only with
  PRISM_MIGRATE_HF=1. Unmeasured: the speed penalty of a cache entry on the HDD (bounded by the overflow size).

## 2026-11-15: finalize jobs run on CPU
- The queue's `*_fin` jobs have no GPU slot (CUDA_VISIBLE_DEVICES empty) but Runner still resolved the profile's `cuda`
  and raised. `harness.run.pick_device` now returns CPU for `--finalize-only` (as for tiny and dry runs); compute jobs
  still fail loudly without CUDA. Earlier tests were all `--tiny` (forced CPU) and could not see it.

## 2026-11-15: language data for E12/E13 (delegated to the assistant by the humans: "for dataset i want you to make a infomred choice keeping scope and quality of dat in mind")
- Chosen (docs/PLAN_COMPLETION.md section 4): English = Pile-10k (same revision as the pilot generic set); Hindi (Devanagari) and
  French/Tamil/Bengali = FineWeb-2 (ODC-By); native-vs-Latin script on identical sentences = Dakshina (hi, ta, bn) through the
  hub mirror `suhani-sarvam/google-dakshina` (columns `indic`, `roman`; one shuffle order for both columns); Hinglish = PHINC
  (CC BY 4.0) filtered to rows with >= 2 romanized-Hindi function words and scrubbed of @handles/URLs.
- Measured with the Gemma tokenizer (scored tokens = positions >= 4 of 2000 windows of 128, i.e. 248,000 when a source has enough
  text): every set meets the 200k floor; **PHINC after filtering and scrubbing has 1,663 windows = 206,212 scored tokens, only
  3% above the floor** (a looser filter or `cmu_hinglish_dog` would add margin; not needed for the floor).
- Not used: IndicCorp v2 (only a README is cached on DEV; the hub copy is a loader script that no longer exists), FineWeb-2
  `hin_Latn` (mixed natural / Hinglish / machine-transliterated / adult), the unlicensed L3Cube hub mirror.
- Shipped as token ids, each file under its source's licence (data/lang/LICENSES.md). This changes the plan text, which said
  third-party text would be fetched rather than shipped: the server may have no internet, and ids under the right licence are
  the robust route. Windows differ between the two Dakshina scripts (same sentence pool, packed independently), so the two
  files cover overlapping but not identical sentences (tokens per sentence differ by script).
- Tweets and Wikipedia sentences are packed into 128-token windows (sentences joined with a newline); web documents use their
  first 127 tokens, as the pilot did. Genre therefore differs across sets: M1 differences can come from genre as well as language,
  and the report must say so.

## 2026-11-15: E13 language families, people-cities task, and what the tokenizer forced
- Measured with the Gemma tokenizer (not assumed): of 45 Devanagari city names only `दिल्ली` is one token, and none of the Tamil or
  Bengali ones tested; Devanagari object nouns are 1-4 tokens (one length class per list is used so that the two objects an OT pair
  swaps have equal length). Consequences: (1) people-cities in Hindi keeps Latin-script city answers inside a Devanagari sentence
  (stated caveat); (2) OT object lists are one token-length class each; (3) Tamil and Bengali get E12 fidelity only, no task packs.
- `harness/langfamilies.py`: OT, RS and people-cities packs for en (parallel baseline), hi (Devanagari), hi_latn (romanized), hinglish
  (English nouns in a romanized-Hindi frame) and fr; 15 families. All are drafts by the assistant and unvalidated (`VALIDATED` all
  False); `scripts/validate_templates.py` checks the tokenizer-side rules and writes `data/lang/templates_review.md` for a native
  speaker. `harness/families.py` is extended through a merged lookup `_fam`; E3's "fresh" selection is untouched (tested).
- people-cities (handoff 7.3): 3 people each in a different city, query one person; the corrupt prompt replaces the queried person's
  city with a city nobody has (one binding differs); 8 shots; single-token Latin city answers. The "optional update step" variant
  is not built.
- Built with the real tokenizer: 15 families x (150 train + 150 test) pairs, **length-matching success 100%** for every family and
  split (candidates kept / generated), all four generation invariants pass. Files: `data/prompt_sets/{entity_track,add_chain,
  people_cities}_e13.jsonl.gz` (3 MB total).
- E13 = base-model behaviour only (`harness/behaviour.py`, `scripts/run_behaviour.py`): clean/corrupt accuracy, filter keep-rates,
  median logit differences, with Gate A (the handoff's/pilot's VIABLE rule: acc >= 0.6 and clean_only keep >= 0.3) and Gate B (the
  preregistered "within 10 points of English", applied per task in the analysis). The two gates are separate columns. E13 uses the
  test candidates only (rank order); E14 (gap/repair on these languages) would use the train split too and is not built.
- Ingest accepts table experiments (manifest `table`: fidelity.csv / behaviour.csv) instead of per_pair.parquet.

## 2026-11-15: template validation (decision by the humans: Prakhar validates the language templates)
- Sign-offs live in `data/lang/signoff.json`, written only by `scripts/sign_templates.py --langs ... --by "<name>"` after reading
  `data/lang/templates_review.md` (3 sample pairs per family). Each entry stores the pack's content hash (`LF.pack_hash`), so an
  edit to a template voids the sign-off (`sign_templates.py --list` shows VOID). E13 stamps `validated` from this file when it runs,
  so signing does not require rebuilding the prompt sets; editing a template does (`scripts/build_lang_tasks.py`).
- The assistant does not sign. Until the file has valid entries, every E13 row says `validated: false` and E14 stays blocked.

## 2026-11-16: release-2 results: provenance, finalize on DEV, unfrozen preregistration
- Written before any release-2 outcome number was computed. `PREREGISTRATION.md` was still unfrozen ("frozen on <date the humans
  confirm>") when the release-2 results were first read, at the humans' request on 2026-11-16. The gate rule applied is the text in
  that file plus the 2026-11-03 human decisions (f_pilot, fresh-fitted mean errors); nothing in it was changed.
- Results branch: the humans committed the raw `runs/` of gnode077 (commit 2b9c898 on top of 02c606f). Compute for E3/E4/E5/E8 ran at
  31713f8 = tag run-20261114-1 + a 2-line change to scripts/slurm_release2.sbatch only (`git diff --stat 20809e8 31713f8`); harness,
  configs, data and src are identical to the tag. All stage-1/2 manifests and all stage-3 workers finished (`worker_done`, no claims
  left, clean tree, not smoke). The four CPU finalize jobs failed on the server (CUDA asked for on a CPU-only job; fixed in d9cfb0b).
- Finalized on DEV from the committed parts/, in a worktree at 31713f8 with HL_DEVICE=cpu, by calling harness.run.main(--finalize-only)
  with one substitution: the cache walk of run_stage3(compute=False) (needs the server's activation caches) was replaced by checks that
  need none: run_hash computed == run directory, per cell checks.json and __clean part present, unit indices contiguous and equal to the
  workers' logged 'unit k/N', every unit parquet = n_conditions x test pairs with the clean part's pair ids. finalize() itself unchanged.
  Each manifest carries `finalize_note` and `finalize_checks`. Bootstrap CIs use DEV's numpy/pandas (versions in the manifest), which
  can differ from the server lock in the last digits.
- The commit 31713f8 carries no run-* tag; ingest needs --allow-untagged for these runs, justified by the diff above.
- E1/E2 runs in the results branch (0c8b47bd230b, a975fc2ace9c) are preflight smoke runs and are excluded from every table.

## 2026-11-16: human decisions after release 2
- **RQ1: running sum (add_chain) L18 = GO** (the humans, 2026-11-16). Object tracking L12 is not declared GO; its gated experiments
  may still be run as exploratory, redundant rows (labelled so) and rejected in analysis.
- **Templates**: all 15 language packs (en, hi, hi_latn, hinglish, fr x OT/RS/PC) reviewed and approved by Prakhar Singhal; recorded
  in data/lang/signoff.json by the assistant at the reviewer's instruction (each entry bound to the pack's content hash).
- **Two-pass plan** (the humans): build the harness so that every remaining experiment (E9, E10, E11, E14, and redundant variants)
  runs in one compute pass, then one analysis pass that applies the preregistered gates and rejects what does not qualify.
  Release 3 (E12, E13) starts now, ahead of that pass.
- Tests read sign-offs from PRISM_SIGNOFF_FILE when set, so test runs never touch data/lang/signoff.json.

## 2026-11-16: preregistration frozen; E9 thresholds, E10 reading and M3 definition
- The humans froze `PREREGISTRATION.md` (sha256 2d336914...672cf) and `PREREGISTRATION_ADDENDUM.md` (sha256 4ba48a08...4aaa) on
  2026-11-16; `tests/harness/test_prereg_frozen.py` fails on any edit. Release-3 and final-pass results had not been read; release-2
  results had (logged above).
- E9 thresholds: the humans asked the assistant to choose them from a small analysis. Calibrated ONLY on release-2 data and random
  geometry (`analysis/calibrate_thresholds.py` / `.txt`): random-U repair moves flips <= 1 point; flip-gap CI half-width ~3 points at
  450 pairs; Haar principal-cosine p99 in R^2304 = 0.052 / 0.072 / 0.097 / 0.131 for k = 1 / 2 / 4 / 8. Rules (addendum E, constants
  in harness/final.py): effective = flip >= random p95 + 10 pts and CI lower bound > random p95; illusion = full_along_U effective
  and full_along_U - feat_A_along_U >= 10 pts with paired CI > 0; aligned >= 0.5 / unrelated <= Haar p99 / partial; DAS fit valid =
  train interchange accuracy >= 0.5.
- E10: the humans confirmed "move to the other box" = the steered B run's top-1 becomes A's answer.
- M3 ("similar"), chosen by the assistant at the humans' request: non-inferiority, paired over the same evaluation pairs: share
  closed by English-fit U minus native U, 95% lower bound > -10 points; flip rate, pca_pooled r = 1, 2; only where the native share's
  lower bound > 0 and the row is readable. Chosen over a CI-overlap rule because overlap rewards wide CIs.

## 2026-11-17: bonus / redundant runs, one sbatch for both queues
- Added at the humans' request (exploratory, lowest queue priority): E9b and E10b (E9 / E10 on running sum L18 fitted on pilot
  templates, evaluated on the three fresh families: does the subspace, DAS rotation and steering vector generalize to new
  templates?) and E2r (E2's confirmatory cells with seed 1: new random U, noise and donor draws on the same pairs; checks the
  stability of the random nulls and agreement between a server-finalized run and the DEV-finalized release 2). Profiles allow
  E2r / E9b / E10b; a test checks every queued experiment is allowed by every profile (the dry-run caught the missing entries).
- `scripts/slurm_final.sbatch` runs release 3 then the final pass in one job (gnode074, 3 GPUs, 72 h), adds `slurm-*.out` to the
  repo's exclude file itself, never deletes activation caches unless called with --clean-full, and has a --check mode for the
  login node. Operating instructions: docs/FINAL_PASS_RUNBOOK.md.
- Tiny-model end-to-end runs (stages 1-3 + finalize, CPU) passed for E9, E10, E11, E14, and make_final.py ran on their outputs.
  DAS train interchange accuracy is 0 on the random tiny model (it cannot learn the task): this checks plumbing only; on real runs
  the frozen das_fit_valid rule reports a failed fit.

## 2026-11-17: deep code review (5 parallel reviewers) and fixes; one queue for everything
Blockers found and fixed (each has a regression test in tests/harness/test_review_fixes.py or test_cell_skip.py):
- Release-3 preflight crashed on a queue of script jobs only (no experiment config to take the model revision from): it now uses
  the queue's `prefetch: model_revision`. Without the fix all 14 E12/E13 jobs would have been skipped.
- A cell with too few train pairs (e.g. a language the model cannot do) raised and killed every E14 worker at that cell, and blocked
  finalize. Now `CellSkip`: the cell is recorded as skipped in its checks and the run continues (compute and finalize). Minimum
  train pairs 20 (was 2; with 2 the estimators return fewer columns than rank 4 and crashed anyway).
- E4_1m's two 1m SAEs need ~36 GiB of downloads on a ~14 GiB SSD: prefetch would fail and skip everything. E4_1m moved to an
  optional queue (queue/extra-1m-server.yaml).
- make_final.py crashed on real data for E11/E14 (spec rows had no `role`): roles now come from each config cell.
Major / analysis fixes:
- Queue `.done` markers carry the commit and unit count: after a re-tag, commit-dependent units (stage 3, finalize, script jobs,
  bundle, publish) are redone (they resume cheaply); stage 1/2 and preflight markers stay valid. Bundle/publish ship exactly the
  runs a queue produces (`--queue`), at any commit, and run again on every submit.
- DAS: finalize never trains (missing rotation = error); refuses a network-filesystem cache; NaN rotation not cached; cache key
  versioned; workers train different ranks in parallel (non-blocking first pass) instead of serializing on one GPU; cosines with
  DAS only at its own rank.
- Disk: the job fails in seconds when the cache tiers have < 25 GiB free (instead of pausing for hours); the queue gives up after
  60 min paused with nothing running; spill threshold 8 GiB; leftover *.tmp.* files cleaned. RAM budget respects the job's cgroup.
- Analysis decision columns follow the frozen addendum exactly: E9 `effective`/`illusion` and E10 Q2 require >= 20 random seeds
  and are NA (not False) when not evaluable; invalid DAS fits are not evidence; M2/M3 flags only on readable rows; canonical rows
  and BH per evaluation family within task; merges keyed by run_hash and fit families. E13 rows are keyed by split (the train-split
  extension must not overwrite the test-split gates of record).
- Stage 1 skips its locks when every entry is already complete (release-2 caches), so cached jobs never wait on another job.
Single run: queue/complete-server.yaml = release 3 + extensions (E12 at L6/L24 via the canonical L0 rule, which reproduces the
pilot's variants at L12/L18; E13 train split) + final pass + exploratory E9b, E10b, E2r and new E16 (running-sum layer sweep of
along-U / repair with DAS rank 1/2/4 at L12, L14, L16, L20). scripts/slurm_final.sbatch runs it by default.
Not changed (noted): the run-side quick summary.csv pools E14's languages (analysis uses make_final, which does not); each
load_spans reads a whole span cache (~5 GiB host RAM per call for the e13 OT set; fits 96 GB with 3 workers).

## 2026-11-18: final pass ingested (untagged commit accepted), tables and analysis
- The complete-server queue ran on gnode084 (4 GPUs) instead of gnode074: the operator committed a 2-line change of the
  `#SBATCH --nodelist/--gres` lines on the server, so every run carries commit 14e7e1f (= run-20261117-2 + that commit; harness
  code identical). Accepted with `--allow-untagged`; the gap is logged per run in runs/_ingested.jsonl. In future, node and GPU
  count are given on the sbatch command line (they override the #SBATCH lines), never by committing on the server.
- New scripts/ingest_published.py: ingest of a published results directory (results_server/<tag>/ on a results/* branch), with
  the checks of ingest_results.py (sha256 of every file against PUBLISHED.json, schemas, tag, no smoke/tiny/dirty runs, no
  conflicting duplicates). 32 runs merged, 0 errors.
- Tables: tables/final/ (make_e12, make_e13, make_final, n_boot 10 000); tables/ updated to the release-2 versions that were read
  on 2026-11-16/17 (E4_width_persistence, E8_*, RQ1_gate). Analysis of record: analysis/final_pass_results.md (numbers copied from
  the tables, decisions by the frozen addendum rules). Release-2 exploratory digests moved into analysis/release2/.
- Read under the frozen rules: E9 RS L18 error-derived U not effective at r <= 8; DAS-4/8 effective with both the illusion and the
  feature-mediated flags true (the two rules are not mutually exclusive; both are reported, neither is redefined); DAS-1/2 failed
  fits. E10 Q2 not met. M1 supported everywhere. Only pc_fr passes both gates, so M2 is testable only there (not supported) and
  M3 has no usable row. These are the answers of record; any later exploratory run cannot change them.

## 2026-11-18: the last compute run, exploratory E17-E22 (post hoc)
Designed after reading the final pass; all exploratory, labelled post hoc, and unable to change any answer of record (the
preregistration and its addendum are not touched). Each targets one open point of analysis/final_pass_results.md:
- E17 DAS robustness (RS L18): das1/2 failed in E9 with a loss that barely moved. Seeds 1-2, 10 and 30 epochs, lr 3e-2; plain
  das<k> reuse E9's rotations (cache key unchanged, tested). Rotation agreement across seeds: E17_das_stability.csv.
- E18 rank: error-derived U up to r = 64 and DAS up to 16 on RS L18, OT L12, people-cities L12/L16 (DAS on people-cities is new).
  Alignment labels at r = 16/32/64 use HAAR_P99_EXPLORATORY (same Monte Carlo as the frozen k <= 8 values: 0.179, 0.246, 0.339);
  the frozen constants are unchanged and an unknown rank now gets no label instead of the k = 8 value.
- E19 steering dose 0.25-16 (RS L18, OT L12, PC L12); Q2 columns computed mechanically but not a revisit of Q2.
- E20 E9's contrasts with wider SAEs at matched L0 (RS L18 65k/64, OT L12 65k/72 and 262k/67).
- E21 E12 fidelity with 65k SAEs at L0 matched to the 16k pilot variants (run_fidelity.py --l0-rule match16k, E4's closest rule);
  make_e12.py now keys rows by (set, layer, SAE) so wider runs sit next to the 16k rows and compare with English at the same SAE.
  262k fidelity left out: on 11 GB GPUs a float32 262k SAE leaves room for a few windows per batch (hours per set, OOM risk).
- E22 multilingual DAS (rank 4/8) and pca_pooled, fitted natively or on English, on RS L18 (fr, hinglish, hi) and PC L12 (fr, hi,
  hinglish): descriptive, E22_transfer.csv; Gate A/B status is reported alongside, nothing enters M2/M3.
Harness: estimator grammar das<k>[_e<epochs>][_lr<lr>][_s<seed>] (harness/config.py parse_das), steering conditions
steer_x<scale> for any positive scale, r_max per config up to 64. scripts/slurm_final.sbatch now defaults to gnode084, 4 GPUs and
this queue; node/GPU changes go on the sbatch command line (never a server-side commit). Tiny end-to-end runs (CPU) of all five
configs and make_final on their outputs passed; make_final on the real E9/E10 runs gives identical decision columns.

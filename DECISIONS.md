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

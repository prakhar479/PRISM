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

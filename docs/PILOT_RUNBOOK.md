# Breadth pilot: runbook (branch `pilot-breadth`, 2026-11-22)

One Slurm job runs one queue, `queue/pilot-breadth-server.yaml`, on 4 x RTX 2080 Ti for up to 96 h. Every experiment answers one
question of `docs/PILOT_BREADTH_PLAN.md` with a decision rule written down before it runs. All results are **exploratory and post
hoc**: none of them touches a preregistered answer.

| job ids | pilot | question | cells | est. GPU-h |
|---|---|---|---|---|
| `P0_*` | P0 protocol | Does feature capture (RS L18 0.75, PC L12 0.69) survive the DAS objective (ce / ld / kl), 3 independent seeds, held-out validity, the write-back operator (subspace vs code level) and matched random nulls? | RS L18, PC L12 | ~30 |
| `P1_*` | P1 binding blind spot | At matched rank, do SAE features carry the *pairing* (binding swap) or the *pointer* (query swap) less than the *content*? | PC x {content, swap, query} x L8/12/16 x 16k/65k | ~40 |
| `P11_*` | P11 OT without default | Object tracking where both runs move the queried object (no prefix fallback): split, error-only, noise, DAS; fallback box recorded | OT-v2 L8/12/16 | ~10 |
| `P2_*`, `PF_P2`, `T_frozen_*` | P2 (+P9) dictionary sweep | Do architectures (7 SAEBench families x 4k/16k/65k x 2 L0), widths (Gemma Scope 16k-262k at L12, 16k-1m at L19), a frozen-random-decoder control and the temporal SAE differ in captured task state, and do fidelity metrics predict it? | PC L12, RS L12 (~50 dictionaries each), RS L19 | ~45 |
| `P4_*`, `PF_P4`, `T_klft_*` | P4 (+P6) logit fixes vs state | Do a KL+MSE fine-tune, the linear error add-on (Engels' predictable error) and dense low-rank add-backs (top-24 / top-1 PCs) close the task gap as much as the CE gap? | RS L18, PC L12, OT-v2 L12 x 5 variants | ~25 |
| `P5_*` | P5 magnitude | Is the state the SAE misses the smooth magnitude axis (ridge line) rather than the periodic digit code (Fourier planes)? Which carries the far-donor resampled-error effect? | RS L16/18/19 x 16k/65k, +1m at L19 | ~12 |
| `P7_*`, `E23_en` | P7 sensitivity | Do gradient-chosen error directions (first-order optimal `grad_err`, `grad_pooled`) repair better than variance-chosen ones (PCA, answer-separating)? English splice-vs-noise cost per token (E23 design) | RS L18, PC L12, OT-v2 L12 | ~12 |
| `P8_*` | P8 second model | Gemma 3 1B + Gemma Scope 2 (16k 'medium', L7/13/17/22): capability (keep rates), causal split, DAS capture | RS, PC, OT-v2 x 4 layers | ~25 |
| `E23_hi_*`, `E23_fr_web` | P10 (optional) | Multilingual E23 noise controls (never run before); `required: false` | 3 language sets x L12/18 | ~6 |

Total about 200-230 GPU-h (about 55-60 h of wall time on 4 GPUs), estimated from E17-E22's measured cost per condition, +-50 %.
P3 (transcoders / cross-layer transcoders) is **not** in this pilot: see "Not built" below.

## Before (login node)

1. **Checkout and tag.** The harness refuses a dirty tree. Tag the commit you run, so results ingest without `--allow-untagged`:
   ```bash
   cd <repo>; git fetch origin; git checkout pilot-breadth     # or the tag
   git tag run-$(date +%Y%m%d)-pilot && git push origin --tags  # if not tagged yet
   git status --short                                           # must print nothing
   ```
2. **Hugging Face access.** The token (`HF_TOKEN` or `~/.hf_token`) must have accepted **both** the Gemma 2 and the **Gemma 3**
   licences (`google/gemma-2-2b`, `google/gemma-3-1b-pt`); Gemma Scope 2 (`google/gemma-scope-2-1b-pt`) and the SAEBench / temporal
   SAE repos are public.
3. **Checks.** `bash scripts/slurm_pilot.sbatch --check` (tag, clean tree, token, queue, the committed pilot inputs, generated
   configs, golden fixtures, disk).
4. **Disk.** The job needs about **110 GiB** over the cache tiers (`MIN_FREE_GIB=120` by default): ~50 GB of downloads
   (SAEBench 65k dictionaries ~1.2 GB each x 14, Gemma Scope 262k at L12, **1m at L19 (~19 GB)**, Gemma 3 1B) and ~60 GB of
   stage-2 caches (P2 keeps x_hat of ~50 dictionaries). If `/ssd_scratch` is short, spill to the bulk disk:
   `sbatch --export=ALL,PRISM_BULK=/scratch/$USER/prism scripts/slurm_pilot.sbatch`.

## Smoke on the real model (recommended, ~1-2 h)

```bash
sbatch --time=04:00:00 scripts/slurm_pilot.sbatch --smoke
grep -E "SMOKE (OK|FAIL)|smoke (finished|passed)" slurm-prism-pilot-<jobid>.out
```
The smoke job covers every pilot path, with tiny sizes, on the real models across the 4 GPUs:
- every pilot config run with `--smoke`;
- 3-step runs of the four training jobs;
- the P4 fidelity job and the English E23 job.

Caches and runs go to a throwaway `$PRISM_WORK/smoke` (deleted afterwards; logs are kept in `$PRISM_WORK/smoke/logs`), so nothing
leaks into the real run. Its prefetch is real, so the full job starts with every download in place. It catches what the CPU tests
cannot: GPU memory, device placement, real checkpoints and the gated-model access.

## Run

```bash
sbatch scripts/slurm_pilot.sbatch                                         # everything (gnode084, 4 GPUs, 96 h)
sbatch --export=ALL,ONLY=P0,P11 scripts/slurm_pilot.sbatch                # a subset: pilot ids or job ids (+ their dependencies)
sbatch --nodelist=<node> --gres=gpu:<n> scripts/slurm_pilot.sbatch        # another node (never edit and commit the script)
```
If a DAS v2 batch runs out of GPU memory, the batch is split and later batches are halved (`oom_backoffs` in the rotation's
info). Time limit, preemption or a cancel: resubmit the same command. Finished units are skipped, and so are cached DAS rotations,
gradient directions, add-on maps and trained dictionaries. gnode084 holds the final pass's RS L18 / PC L12 caches, which P0 reuses;
any node works, it only recomputes them.

Order (the scheduler fills 4 GPUs by priority):
- prefetch and preflight;
- the four training jobs, alongside stage 1;
- stage 2, then stage 3 per pilot: P0 first, since it trains the RS L18 / PC L12 rotations;
- the fidelity jobs, after stage 2 of P2 / P4;
- finalize on CPU, bundle and publish.

## During

```bash
squeue -u $USER
tail -f slurm-prism-pilot-<jobid>.out
cat runs/_queue/pilot-breadth-server/queue_status.md
```
Expect `DAS v2 k=... epoch` lines early (P0 trains 54 rotations; P1 27; the others reuse or train a few), `grad directions:` lines
for P7, and `add-on ... fitted on 262144 generic rows` for P4.

## After

```bash
bash scripts/slurm_pilot.sbatch --record          # results/pilot-<tag> branch with results_server/ committed (no push)
git push -u origin HEAD                           # then tell DEV the branch name
bash scripts/slurm_pilot.sbatch --clean-full      # only after DEV confirms the results arrived
```
If a `*_fin` job failed, also `git add -f runs/P* runs/_queue` before pushing.

On DEV:
1. `scripts/ingest_published.py` (as for earlier releases).
2. `python scripts/make_pilot.py --runs results_server/pilot-breadth-server/runs --out analysis/pilot`, which writes per-pilot CSVs
   and `analysis/pilot/VERDICTS.md` with each decision rule evaluated.
3. Decide the paper direction from the decision matrix in `docs/PILOT_BREADTH_PLAN.md`.

## What was built for the pilot (code map)

| piece | where | tested in |
|---|---|---|
| DAS v2: `das<k>v2_<ce\|ld\|kl>[_e][_lr][_s]` | `harness/model/das.py` (`fit_das_v2`), `harness/config.py` (`parse_das2`) | `test_pilot.py` |
| Dictionary layer (TopK / gated kinds, registry / trained / add-on sources) | `harness/dicts.py`, `harness/dict_formats.py`, `configs/dict_registry.yaml` (`scripts/build_dict_registry.py`) | `test_pilot.py` |
| Add-ons `lin` (ridge of eps on x) and `pcx<r>` (top-r uncentered PCs of x) | `harness/dicts.py` (`fit_addon`, `AddonDict`); runner `addon_map` (cached) | `test_pilot.py` |
| Code-level write-back `feat_codes_along_U` | `harness/conditions` (`codes_writeback`, `select_latents_along_U`), `harness/model/stages.py` (`stage2_active`) | `test_pilot.py` |
| `resampU_sameAns/diffAns` (U-part of a donor's error) | `harness/conditions` | `test_pilot.py` |
| Estimators `ridge_value`, `fourier_T{2,5,10}` | `harness/directions` (`value_direction`) | `test_pilot.py` |
| Estimators `grad_err`, `grad_pooled` | `harness/model/grads.py`; runner `grad_directions` (cached) | `test_pilot.py` |
| P1 / P11 counterfactual families | `harness/pilotfamilies.py`, `harness/families.py`; sets `scripts/build_pilot_sets.py` | `test_pilot.py` |
| Gemma 3 support (`model:` in configs) | `harness/model/__init__.py` (`MODEL_REPOS`) | local check, HF vs TL logits equal |
| Training (KL+MSE fine-tune; frozen-random-decoder control) | `scripts/train_dictionary.py` -> `<cache_root>/trained_dicts/<name>`, record `runs/PD/<name>` | `test_pilot.py` |
| Dictionary fidelity (CE etc. per dictionary, held-out generic set) | `scripts/run_dict_fidelity.py` -> `runs/PF/<hash>` | tiny run |
| Configs, queue, Slurm, verdicts | `scripts/make_pilot_configs.py` -> `configs/pilot/`, `queue/pilot-breadth-server.yaml`, `scripts/slurm_pilot.sbatch`, `scripts/make_pilot.py` | `test_pilot.py`; tiny smoke of every config |

## Dictionary sources (verified 2026-11-22)

- **SAEBench** (`canrager/saebench_gemma-2-2b_width-2pow{14,16}_date-0107`, `adamkarvonen/..._width-2pow12_date-0108`), resid_post
  L12, `dictionary_learning` checkpoints on raw activations.
  - Two trainers per architecture and width: L0 closest to 40 and 80.
  - The L1-penalised 65k (and 4k gated) sweeps record no L0, so two mid-sweep penalties were taken and their realised L0 is
    measured in P2 (`generic_l0`).
- **Temporal SAE:** `alex-oesterling/temporal-saes` trainer_0 (Bhalla et al.; Matryoshka-BatchTopK format, k = 20). Not the
  attention-based "Priors in Time" model, which cannot be used per token.
- **Gemma Scope 2:** `google/gemma-scope-2-1b-pt`, `resid_post/layer_{7,13,17,22}_width_16k_l0_medium`.
- **Format checks.** Every converter was checked against an independent implementation of the original formula:
  - on real gemma-2-2b L12 activations (all seven SAEBench formats at 4k, Matryoshka / gated at 16k, the temporal SAE);
  - on Gemma 3 1B L13 activations (Gemma Scope 2).

  Reconstructions agree to ~1e-4 (|x| ~ 150), and L0 equals the target (40 / 80 / 20 / ~60).

## Deviations and limits to state

- **KL fine-tune.** It follows arXiv 2503.17272's loss (0.5 (MSE + alpha KL), alpha = MSE/KL detached, AdamW 5e-5 with linear
  decay, clip 1.0), with these differences:
  - about 0.7 M tokens (1 epoch of the 6000-window fit set) instead of 15 M;
  - KL at 32 random positions per window;
  - decoder norms kept by exact rescaling.

  A weak fine-tune biases P4 toward "the fix does not restore state"; the CE gap closed (PF_P4) shows how strong it was.
- **Frozen-random-decoder control.** A TopK encoder (k = the canonical L0) trained for 3 epochs on ~0.7 M tokens.
- **Code-level write-back.** The swapped latents are the smallest top set covering 90 % of
  sum_rows |f_A - f_B| |U^T d_i| on train pairs (at most 512; the selection is logged per cell in `parts/*.codesets.json`).
- **Add-ons.** They are fitted on the first 262,144 rows of `data/generic/pile10k_6000x128_skip100.json`, which shares no document
  with the 100-window evaluation set.
- **Object tracking (P11) and P2's scope.**
  - The P11 families are the E3 templates `ot_placed`, `ot_bin` and `ot_drawer` with the `dest` counterfactual.
  - The reverse sufficiency patch of the plan is dropped: OT-v2 has no prefix fallback, and the fallback box is recorded per pair.
  - P2 uses PC and RS only: caching x_hat for ~50 dictionaries on the 8,100-pair OT set would need ~80 GB.
- **Not built:**
  - **P3** (transcoders / CLTs). Formats are known, and recorded in the handoff for a follow-up.
  - **Gemma 3 4B.** Fp32 does not fit 11 GB; bf16 is emulated on Turing.
  - **The LoRA fix.** No public gemma-2-2b checkpoint.
  - **SAEBench at L19.** None exist, so P2's L19 rows are Gemma Scope widths plus the frozen control.

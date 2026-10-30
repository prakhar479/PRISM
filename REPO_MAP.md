# REPO_MAP: pilot code, outputs and gaps (Phase 0, 2026-10-30)

Written from reading the code at `bc867a8` (branch `proposal-slides`; Phase-0 work is on branch `harness`).
Every number below was read from a committed file; paths are relative to the repo root (`prism_sae_dark_matter/`).
Task names in code differ from the handoff:

| handoff | code (`task`) |
|---|---|
| running sum (RS) | `add_chain` |
| object tracking (OT) | `entity_track` |
| variable copying | `var_binding` |
| names (IOI) | `ioi_control` |
| prompt A / B | `clean` / `corrupt` |
| features-only `x̂_A + ε_B` | `sae_only` |
| error-only `x̂_B + ε_A` | `error_only` |
| noise `x̂_B + η` | `noise_control` |
| `feat_A_zero` `x̂_A` | `sae_splice` |
| `full_A` `x_A` | `clean_full` |
| top principal directions (`pca_pooled`) | `unsupervised_svd` |
| answer-separating directions (`answer_sep`) | `supervised_centroid_svd` |
| repair | exp3 `hybrid_patch` |

**The pilot code is the current `src/`.** `git diff 430aa27 HEAD -- src configs scripts/run_full_suite.py
scripts/preflight.py` is empty, and the paper run (2026-10-26 00:13) started after `430aa27` (2026-10-25 13:12). So
HEAD reproduces the pilot code paths. Proposed: tag `pilot-ref` at `430aa27` and treat `src/` as frozen.

---

## 1. Layout

```
MILP/                         parent dir, NOT a git repo
  .venv/                      DEV virtualenv (python 3.12; no transformer_lens / sae_lens / pyarrow / datasets)
  assignment-1/               unrelated course assignment
  prism_sae_dark_matter/      THE git repo (origin git@github.com:prakhar479/PRISM.git)
    run.sh                    one entry point: setup | test | dryrun | pilot | paper | start | status | resume | decide
    requirements.txt          loose pins (torch>=2.2, transformers>=4.57,<5, transformer_lens>=2.16,<4, sae_lens>=6,<7)
    configs/                  YAML; `base:` inheritance from default_config.yaml (src/utils.load_config)
    src/
      utils.py                device/dtype choice, seeding, YAML/JSON/NPZ/table I/O
      pipeline.py             run_suite(): load model, setup runtime, prepare tasks, exp1 -> exp2 -> exp3 -> plots
      data/synthetic_tasks.py task generators, few-shot prefix, FallbackTokenizer, tokenization/alignment, filter
      models/runtime.py       last-position-only forwards, resume-from-layer, token budget, OOM backoff
      models/sae_loader.py    TL model loader, JumpReLU SAE (npz loader, resident/streamed), mock world + mock SAE
      experiments/common.py   Baseline cache, run_patched, summarize (NLDR + bootstrap), noise, UnitLog, RunContext,
                              generic-text residuals, SAE fidelity
      experiments/exp1_causal_split.py   conditions + controls (pilot Table B.1, B.5, controls text)
      experiments/exp2_geometry.py       probes, spectra, surface-token baseline (Table B.6)
      experiments/exp3_hybrid_repair.py  direction estimators + repair (Tables B.2-B.4)
      analysis/summary.py, plot_figures.py   per-tier SUMMARY.md and fig1..fig7
    scripts/
      _bootstrap.py           sys.path + BLAS thread caps (OMP/MKL/OPENBLAS=4)
      run_full_suite.py       one tier: --config --overrides preflight.json --out --resume --deadline
      run_server_gpu.sh       launcher: env checks -> smoke -> preflight -> prefetch -> tiers -> INDEX.md
      preflight.py            prompt-format x n_shots grid per task -> task_overrides; IOI gate (exit 3)
      prefetch.py             downloads model + SAEs for given configs
      check_forward.py        TL vs HF forward comparison, tokenizer, prompt-space checks (source of the "95%")
      analyze_tasks.py        task viability study (results/task_analysis)
      decide.py               GO/MAYBE/NO-GO verdicts -> DECISIONS.md inside a results dir
      run_smoke_test.py       one layer/SAE end to end (mock or GPU)
      make_handoff_figures.py, make_proposal_figures.py, make_slide_figures.py   read results -> figures/CSVs
      probe_env.py            NEW (Phase 0): environment probe
      export_golden.py        NEW (Phase 0): golden-fixture export, RUN only, pilot code paths
      hf_agreement.py         NEW (Phase 0): HF top-1 agreement on pilot task prompts, RUN only
    tests/test_pipeline_cpu.py   78 tests, all CPU + mock (see §9)
    results/                  in .gitignore but force-added: ALL pilot outputs are committed (see §4)
    docs/RUNBOOK.md, docs/handoff/ (HANDOFF.md, data/*.csv, figures/), proposal/, slides/
```

## 2. Modelling backend and hooking

- **TransformerLens** `HookedTransformer.from_pretrained_no_processing("gemma-2-2b", dtype=bf16 on CUDA, fp32
  otherwise)` (`src/models/sae_loader.py:load_model`). No weight folding or centering, so the residual matches what
  Gemma Scope was trained on. The model is built on CPU and moved per child module. On GPUs < 16 GB the embedding
  (256k × 2304) stays on CPU (`offload_embed`); forwards then start at block 0 with `start_at_layer=0` and
  CPU-gathered embeddings (`runtime._input`).
- **Hook site:** `blocks.{L}.hook_resid_post` (`HOOK_FMT`, `sae_loader.py:31`), i.e. the residual stream after
  block L. This is the Gemma Scope `res` site.
- **Reading:** `cache_baseline` registers read-only hooks at the needed layers and stores (a) the last-position
  residual as float32 and (b) the residual at positions `diff_pos..T-1` (the "span") in **model dtype (bf16)**.
- **Overwriting:** `run_patched` (`common.py`) either
  - (default after a startup check) **resumes**: computes `resid_post[L]` of the unpatched B run once per layer
    (`Baseline.layer_resid`), writes the patch rows into a copy, and runs blocks `L+1..25` with
    `start_at_layer=L+1`; or
  - runs a full forward with a hook that clones the activation and assigns the rows.
  `verify_resume` checks the two agree (`resume_check_max_abs_diff` = 0.0 in all three paper tiers).
- **Logits:** only the last position is unembedded: `final_resid` (`stop_at_layer=n_layers`) → `head()` =
  `ln_final` → `unembed` → `.float()` → softcap `30·tanh(x/30)`. **D uses post-softcap float32 logits.**
- No padding: batches group prompts of equal token length (`batch_by_length`) under a token budget
  (`max_batch_tokens`, 2048 in the paper tiers).

## 3. Dtypes

| object | dtype |
|---|---|
| model weights / forward | bf16 on CUDA (`get_dtype`), fp32 on CPU/MPS |
| cached last-position residual | fp32 cast of a bf16 activation |
| cached span residual (what from_diff patching decomposes) | bf16 (cast to fp32 in `Baseline.acts`) |
| SAE weights | fp32 for widths < 524k, bf16 for ≥ 524k (`bf16_min_width`) |
| SAE math | pre-activations in weight dtype; decode accumulated in fp32; `ε = x − x̂` exact in fp32 |
| patch rows | built in fp32, cast to the activation dtype (bf16) when written |
| logits | fp32 after softcap |

## 4. How each pilot number was produced

**Main pilot run:** `results/paper_20261026_001341/` (committed). It was launched as `./run.sh paper`, which runs
`scripts/run_server_gpu.sh --name paper --tiers paper_residue,paper_widths,paper_audit --quick-preflight
--skip-smoke`. For each tier that executes:

```
python scripts/run_full_suite.py --config configs/<tier>.yaml \
    --overrides results/paper_20261026_001341/preflight/preflight.json \
    --out results/paper_20261026_001341/<tier> --resume --deadline <t>
```

**Machine:** RTX 5060 Laptop GPU (7.53 GiB usable), ~16 GiB RAM, 4 CPU threads, Python 3.14.4, torch
2.14.1+cu130, transformers 5.20.1, sae_lens 6.54.5, numpy 2.5.4. The check_forward report records
transformer_lens as `null` (not detected), so its version is **unrecorded**. Runtime plan:
`max_batch_tokens=2048`, embedding on CPU, resume verified (0.0).

**Pair settings** (`preflight.json` overrides config defaults): seed 0; test seed 10 000; 300 train / 150 test.

| task | shots | filter | cand. multiplier | expected keep | mean T (tokens) | mean span |
|---|---|---|---|---|---|---|
| add_chain | 8 | strict | 3 | 0.625 | 234.7 | 11.5 |
| entity_track | 8 | **clean_only** | 6 | 0.271 | 377.0 | 13.4 |
| var_binding | 2 | strict | 4 | 0.396 | 71 | 14 |
| ioi_control | 2 | strict | 2 | 1.0 | 47 | 5 |

**Proposal appendix tables → sources** (all under `results/paper_20261026_001341/`; the proposal's header comment and
`scripts/make_handoff_figures.py` say the same):

| proposal table | content | source |
|---|---|---|
| B.1 `tab:split` | features/error NLDR, L4–20 | `paper_audit/metrics/exp1_results.csv` (from_diff) |
| controls text | noise at L12: 7/51/26/9; resample 28 vs 26 | `paper_audit` exp1 (noise_control; error_resample_* at `last`) |
| B.2 `tab:repair` | share closed, r = 1/2, generic, random, "for half" | `paper_residue` (L12, 14, 18) + `paper_audit` (L8, 16, 20) exp3 |
| B.3 `tab:spec` | sources at OT L12 / RS L18 | `paper_residue/metrics/exp3_results.csv` |
| B.4 `tab:width` | OT L12 across widths | `paper_widths` exp1 + exp3 |
| B.5 `tab:positions` | clean_full at `last` | `paper_audit` exp1 |
| B.6 `tab:probes` | probes at L12 | `paper_audit/metrics/exp2_probes.csv` |
| model table fidelity | CE recovered 16k | `paper_residue/metrics/sae_fidelity.csv` |
| replication text | 160/80 run | `results/pilot_20261025_230749/` |

**Regression targets, verified from the CSVs** (`paper_residue`, OT L12 16k, from_diff, n = 150):

| condition | NLDR [95% CI] | handoff says |
|---|---|---|
| sae_only (features-only) | 0.741 [0.726, 0.758] | 74 [73, 76] ✓ |
| error_only | 0.620 [0.595, 0.646] | 62 ✓ |
| noise_control (2 seeds) | 0.514 [0.476, 0.551] | 51 ✓ |
| unsupervised_svd r=1 | 0.877 [0.864, 0.891] | 88 ✓ |
| random_subspace r=1 (mean of 3 seeds) | 0.741 | 74 ✓ |
| sae_splice (= `feat_A_zero`) | 1.086 [1.066, 1.106] | (not in handoff) |

RS L18: sae_only 0.874, error_only 0.143, noise 0.208, unsupervised r=1 0.903, supervised r=2 0.948.

**What is in the repo vs only on RUN:**

| exists on DEV (committed) | only on RUN / nowhere |
|---|---|
| all metrics CSV/JSON/parquet, `*_units.jsonl`, SUMMARY/DECISIONS, run_info, preflight, prefetch | activations / residuals (never saved) |
| per-pair `D` for every exp1 condition (`tensors/exp1_pairs_*.npz`: dl_clean, dl_corrupt, states, dl_<cond>) | **pair ids and prompt texts** (never saved; regenerated from seed + model filter) |
| fitted U, r_max = 32, both estimators + random seeds (`tensors/exp3_U_*.npz`) | per-pair D for exp3 (repair) conditions (only summaries) |
| exp2 coordinates/spectra | any logits beyond D (no candidate logits, no p_A/p_B) |
| logs (`*/logs/run.log`, `logs/prism_*.log`) | HF / SAE caches, model weights |

Consequences: (a) E1 cannot be computed from stored outputs, because no candidate-set logits were kept, so it needs
new final-position passes. (b) The pilot pairs can only be rebuilt on a machine that runs the bf16 model filter.
`export_golden.py` does this and checks the result against the stored per-pair `D` and states (§13, RUN_REQUESTS R2).

## 5. Tasks, filters, first differing token

- Generators (`synthetic_tasks.py`): `_gen_add` (depth 3; templates `calc` with "Start with/Begin with/Start
  at … Add k. … Total/Sum/Result =" and `story` "Tom has 2 apples. Tom gets 3 more. … Tom now has"; totals ≤ 9),
  `_gen_entity` (depth 2: 3 objects, 2 moves; B makes the queried object do move k), `_gen_var`
  (depth 1), `_gen_ioi`. Each returns a `CounterfactualPair` with answers and state ids.
- `generate_pairs(task, n, seed, exclude)`: deterministic `np.random.default_rng(seed)`. It rejects duplicates and
  prompts in `exclude`. Pair ids are `f"{task}-{split}-{k:05d}"` by acceptance order, so ids are stable given
  (seed, n·multiplier).
- `build_task_splits`: the few-shot shots are drawn first (seed + 77) and excluded. Train candidates use `seed`, test
  candidates use `seed + 10000`, both `n × candidate_multiplier`, and the splits are prompt-disjoint via `exclude`.
  The same shot prefix is prepended to A and B. `tokenize_pairs` (answer format `auto`: `" 8"` after `"="`, else
  `"8"` after `"= "`) drops pairs with unequal length or non-single-token / identical answers. The filter is then
  applied and the first n kept.
- **Filters** (`valid_mask`, `min_logit_diff = 2`):
  - `strict` = top1(A)=ans_A ∧ top1(B)=ans_B ∧ D_A > 2;
  - `clean_only` = top1(A)=ans_A ∧ (D_A − D_B) > 2;
  - `none`.
  Pairs that fail are **deleted** (not flagged). Keep rates are recorded only as preflight estimates on 48 pairs.
- **First differing token:** `TokenizedPair.diff_pos` = first index where A and B ids differ (last index if none).
  `from_diff` patches every position `diff_pos..T-1`. `last` patches position T-1.
- Semantic slot annotations: **none exist**.

## 6. Direction estimators (exact math, `exp3_hybrid_repair.py`)

Fit data for task t at layer L: the **last-position** residuals of the train split, clean and corrupt rows stacked
(`exp2_geometry._stack`): `X = [x_A^(last); x_B^(last)]`, 2 × 300 rows, labels = each prompt's own answer state.
These are SAE-decomposed: `E = ε(X)`. **The U fitted on last-position errors is applied at every from_diff
position.**

- `unsupervised_svd` ("top principal directions"): `Vt` of the SVD of `E − mean(E)` (float64); `U = Vt[:r_max].T`.
  The sign is whatever LAPACK returns.
- `supervised_centroid_svd` ("answer-separating"):
  1. Compute class centroids `c_k = mean(E[y = k])` and center them with class-count weights.
  2. Take the SVD of the centroid matrix and keep the `k1` directions with `S > 1e-8·S_0` (at most n_classes − 1).
  3. If `k1 < r_max`, pad with the top right-singular vectors of the residual
     `(E − Ē) − (E − Ē) V Vᵀ`.
  4. Orthonormalize with QR, with the sign fixed so that `diag(R) > 0`.
- `random_subspace`: QR of `np.random.default_rng(seed + 101·s + L).standard_normal((d, r_max))` with sign fix;
  s ∈ {0, 1, 2}. **Summaries average NLDR (and the CI bounds) over seeds.**
- `generic_text_svd`: top eigenvectors of the centered covariance (`fit_cov_eig`) of ε on
  `NeelNanda/pile-10k`. That is 100 docs × 128 tokens at positions ≥ 4 in paper_residue (~12.4k rows).
- `transfer_from_<other>`: the other task's **`supervised_centroid_svd`** U (not top-PC).
- `top_xhat_svd`, `span_svd`: exist (not used in the paper tiers).
- Repair (`hybrid_patch`): `x̂_A + ε_B + U Uᵀ(ε_A − ε_B)` = `x̂_A + P ε_A + (I−P) ε_B` ✓. `hybrid_splice`
  = `x̂_A + P ε_A`.

## 7. Noise η

`rescaled_noise(ε_ref, gen)`: i.i.d. N(0,1) per element, rescaled **per token** so `‖η_t‖ = ‖ε_ref,t‖`.
**`ε_ref = ε_A` (the clean error), not ε_B.** It is drawn on CPU over all test span rows at once, in Baseline order,
with `torch.Generator().manual_seed(cfg.seed + 1000·s + L)`, s < `noise_seeds` (2 in paper_residue, 1 in
paper_widths). The noise for a given pair therefore depends on the whole test set and its order. The NLDR and CI
reported for noise are means over seeds.

## 8. Bootstrap CIs

`summarize()`:
- **NLDR is a ratio of means**: `(mean D_p − mean D_B)/(mean D_A − mean D_B)`. The per-pair mean and median are
  also stored (`nldr_pair_mean`).
- **CI:** paired percentile bootstrap. One index matrix `default_rng(0).integers(0, N, (n_boot, N))` is shared by
  all conditions of a Baseline. The ratio of means is recomputed per resample and the bounds are the [2.5, 97.5]
  percentiles. `n_boot = 1000` in the paper tiers (300 in the 160/80 pilot).
- `top1_rate` = fraction with top-1 = **A's** answer. `top1_recovery` = (rate_p − rate_B)/(rate_A − rate_B).

## 9. Existing tests

`tests/test_pipeline_cpu.py`: **78 tests, all pass on DEV CPU in ~4 s** (2026-10-30). All use the mock
world/FallbackTokenizer or synthetic tensors. There are **no** GPU-marked or real-model tests. The real-model checks
live in scripts: `check_forward.py`, `run_smoke_test.py` (GPU path), and `preflight.py` (IOI gate). Coverage
includes generator determinism and disjointness, alignment, the JumpReLU encode rule, npz loader, streamed =
resident SAE, NLDR endpoints, rank-0 repair = sae_only, train-only subspaces, resume = hooked forward, bootstrap CI
contains the point, OOM backoff, and config sanity for the paper tiers.

## 10. ⚠ The "95% top-1 agreement with Hugging Face"

- **Where:** `results/check_forward/report.{md,json}`, produced by `scripts/check_forward.py` on 2026-10-20 at commit
  `7a54256` (not the pilot commit). It ran with the tier1_core config + the 2026-10-19 overnight preflight. The task
  suite was the **legacy** one (`mod_sum`, `dyck_depth`, `graph_reach`, `ioi_control`), on which the model is near
  chance.
- **What:** "ours" (pilot path: TL no_processing, bf16, embedding on CPU, last-position head) vs HF
  `AutoModelForCausalLM` (bf16, eager attention), last-position logits on 103 probe prompts:
  - top-1 agree **0.9515**;
  - max |Δlogit| **4.72**, median of per-prompt max |Δlogit| 0.49;
  - status PASS under a lenient tolerance (median ≤ 3.0, agree ≥ 0.9).
  Same-graph checks (`ours vs tl_plain`, `ours vs ours_gpu`) were **not computed** (SKIP).
- **Same report:** `transformers 5.20.1` FAILs the repo's own `transformers < 5` pin. Sanity prompts: ours and HF
  both 6/7 correct with near-identical top-5 probabilities.
- **Not known:** agreement on the four pilot tasks, whether disagreements are near-ties, and whether bf16 TL-vs-HF
  drift flips filter decisions. → `RUN_REQUESTS.md` R3 (`scripts/hf_agreement.py`).

## 11. Hardcoded machine assumptions

| where | what |
|---|---|
| `src/utils.py:get_device/get_dtype` | auto CUDA → MPS → CPU; bf16 iff CUDA (not configurable) |
| `src/models/runtime.py` | `device.type == "cuda"` branches for VRAM planning; `BYTES_PER_TOKEN = 256 KiB` heuristic; `_PHYSICAL_CORES` |
| `src/models/sae_loader.py` | `"cpu"` construction then move; `bf16_min_width = "524k"`; RAM check `need·1.15 > MemAvailable` (`/proc/meminfo`, Linux only) |
| `scripts/_bootstrap.py`, `tests/conftest.py` | `OMP/MKL/OPENBLAS_NUM_THREADS=4` |
| `scripts/run_server_gpu.sh` | requires `nvidia-smi` and `HF_TOKEN`; sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `src/data/synthetic_tasks.get_tokenizer` | **silently falls back to the FallbackTokenizer if `HF_TOKEN` is unset**, even with a cached `hf auth login` token |
| `scripts/check_forward.py` | `embed.to("cpu")`; bf16 iff CUDA |
| HF revisions | **none pinned**: model, SAEs and `NeelNanda/pile-10k` resolve to `main`. SAE ids go through SAELens's `pretrained_saes_directory` (`gemma-scope-2b-pt-res-canonical`), so the mapping depends on the sae_lens version |
| result paths | `results/<name>_<stamp>` relative to repo; tracebacks in `failures.jsonl` contain the RUN user's absolute path (`/home/raveeshvyas/...`) |
| `.gitignore` | ignores `results/` (outputs were force-added) |

No absolute paths, usernames or hostnames appear in code or configs.

## 12. Gap analysis vs handoff Sections 5–9

| component | status | proposed change |
|---|---|---|
| 5.1 lockfile / `check_env.py` | missing (loose `requirements.txt`; RUN violates its transformers pin) | uv `pyproject.toml` + `uv.lock` pinned to **RUN's actual stack** (pip freeze requested, R1); CPU and cu130 torch via uv index markers; `check_env.py` |
| 5.2 profiles | missing (`runtime:` config block is partial) | `profiles/{dev,run}.yaml`, `local.yaml` ignored, `HL_*` env overrides |
| 5.3 `device.py`, tolerances | partial (`get_device/get_dtype`) | `harness/device.py`, `harness/tolerances.py`; `src/` stays frozen and exempt |
| 5.4 pinned revisions, offline, prefetch | partial (`prefetch.py` downloads, no hashes/revisions) | revisions in configs; `scripts/prefetch.py` rewritten for queue + sha256 (keep old one as `prefetch_pilot.py` or leave in place) |
| 5.5 memory estimator / `plan_runs.py` | partial (runtime heuristics after model load) | pure `harness/memory.py` from config; `plan_runs.py` |
| 6 three-stage cached pipeline | missing (single process: cache → SAE → patch in memory; nothing persisted) | stages 1–3 with content-addressed caches; reuse `runtime` forward functions and `JumpReLUSAE` arithmetic |
| 7.1 typed configs, `harness.run` CLI, `--dry-run/--smoke`, run_hash layout | partial (`run_full_suite.py`, `UnitLog` resume, `failures.jsonl`) | new CLI, manifests, per-pair parquet |
| 7.2 run queue | partial (`run_server_gpu.sh` tiers + deadline + STATUS) | `queue/<tag>.yaml` + `run_queue.py` with sentinels/heartbeat |
| 7.3 tasks | partial: 4 generators exist; **filters delete**; no template families, slots, people–cities, multilingual | reuse generators unchanged; add `template_family_id`, slot spans, filter flags, new families, new task, language data files |
| 7.4 SAE inventory / L0 matching / near-threshold sparse z | partial (canonical via SAELens; L0 fallback `target_l0=100`; block-wise encode exists) | `sae_inventory.py` (can run on DEV: hub reachable), load by (layer, width, L0, revision), `match_l0` |
| 7.5 conditions | partial: `feat_A_errB`, `feat_B_errA`, `feat_B_noise`, `feat_A_zero`, `full_A`, `repair` exist; resampling/mean-error at **last position only**; no mean-pos, cov-noise, repair_mean, along_U, subthresh | pure `harness/conditions/` reproducing pilot ones bit-for-bit; add the rest |
| 7.6 estimators | `pca_pooled`/`answer_sep`/random/generic/transfer exist; no pos/slot-centered, no FitSet leakage guard (train-only by convention + test) | `harness/directions/` wrapping the pilot math; `FitSet(split="train")` guard |
| 7.7 metrics | **missing**: only D, NLDR, top-1 = A's answer | candidate logits + logsumexp per pair/condition; p_A/p_B, flip rate, prob recovery, entropy |
| 7.8 statistics | partial: paired bootstrap 1000 percentile | 10k, cluster bootstrap, random-U distributions, Holm/BH, `analysis_role` |
| 7.9 geometry / DAS / steering / binding | missing | Phase 3 |
| 7.10 fidelity audit | partial: CE recovered with **mean ablation**, tiny sample (2 × 16 × 128 tokens) | add zero-ablation variant, FVU, ‖ε‖/‖x‖, L0, fertility; ≥ 200k tokens |
| 8 tests | 78 mock tests; no tiny-Gemma, golden, gpu/realmodel markers | T0 (tiny TL Gemma-2 config + random JumpReLU npz), T1 (golden), markers |
| 8.1 preflight on RUN | partial (`preflight.py` = prompt-format grid + IOI gate; `check_forward.py`) | new `scripts/preflight.py` per 8.1 (keep the old one as `scripts/preflight_pilot.py`) |
| 4.4 bundles / ingest | missing | `bundle_results.py`, `ingest_results.py` |
| 4.3 golden export | **done (Phase 0)**: `scripts/export_golden.py` | — |

## 13. Inconsistencies (code ↔ outputs ↔ handoff/proposal)

1. **D_B ≈ −D_A does not hold for object tracking.** OT test mean D_A = 2.77, D_B = −0.15 (run_info). OT uses the
   `clean_only` filter, so B need not predict B's answer. From the stored `top1_recovery` and `top1_rate` of
   `sae_only` at OT L12, the clean B run already has top-1 = A's answer on **≈ 37%** of OT test pairs (derived, not
   stored per pair). RS L18 is ≈ symmetric (3.48 / −3.02) and its B run never gives A's answer. This bears on §1.3
   point 1 and on the flip-rate denominator in the pre-registration. I report it here without interpreting it.
2. **NLDR is a ratio of means**, not the average of per-pair ratios that the proposal text and handoff §1.2
   describe. For OT L12 sae_only they are 0.741 vs 0.739.
3. **Noise η is norm-matched to ε_A**, not to "ε" generically or to ε_B (§7). The handoff's `feat_B_noise` should
   say which.
4. **CIs for multi-seed conditions** (noise, random U) are averages of per-seed percentile bounds, not bootstrap
   CIs of the seed-averaged statistic.
5. **`feat_A_zero` already exists** (`sae_splice`): OT L12 NLDR 1.086, above `full_A`.
6. **Direction fitting uses last-position errors only** (A and B rows), applied at all from_diff positions. The
   handoff's "PCA of task ε" needs that qualifier, and `pca_pos_centered`/`slot_centered` will differ by
   construction.
7. **Width table is not L0-matched, and one SAE comes from a different release.** The L12 SAEs are
   16k `average_l0_82`, 32k l0_76, 65k l0_72, **131k l0_129 (from `gemma-scope-2b-pt-res` via the
   `target_l0 = 100` fallback, because SAELens lists no canonical 131k)** and 262k l0_121 (canonical). The proposal
   notes the sparsity confound; this is its source.
8. **Layer 18 has only 16k and 65k SAEs** in `google/gemma-scope-2b-pt-res` (L0 variants 16k: 23/40/74/138/280;
   65k: 21/36/64/116/117/216). The E4 secondary "RS L18 across widths" can only compare 16k vs 65k. L12 (repo
   revision `fd571b47`, listed from DEV on 2026-10-30) has these L0 variants:

   | width | L0 variants |
   |---|---|
   | 16k | 22, 41, 82, 176, 445 |
   | 32k | 12, 22, 40, 76, 155, 360 |
   | 65k | 21, 38, 72, 141, 297 |
   | 131k | 12, 20, 36, 67, 129, 264 |
   | 262k | 11, 21, 36, 67, 121, 243 |
   | 524k | 22, 29, 46, 65, 115, 227 |
   | 1m | 19, 26, 58, 73, 107, 207 |
9. **`configs/default_config.yaml` comments are stale.** They say add_chain 2 shots, var_binding 8, OT keep 0.84. The
   pilot used preflight overrides: add_chain 8 shots strict, var_binding 2 shots strict, OT clean_only keep 0.271.
   The proposal matches the overrides.
10. **SAE fidelity is mean-ablation CE-recovered**: `(CE_meanablate − CE_splice)/(CE_meanablate − CE_clean)` on
    ~2 × 16 × 124 tokens. Handoff §7.10 writes `L_zero`.
11. **The 95% HF agreement** was measured on the legacy tasks at an earlier commit (§10). It is not evidence about
    the pilot tasks.
12. **Paper run partial failure.** `paper_residue/metrics/failures.jsonl` records `exp3 fit|10|16k: LinAlgError:
    SVD did not converge`, so there is no L10 repair row. Table B.2 does not claim L10.
13. **requirements.txt vs RUN.** transformers `<5` is pinned but RUN ran 5.20.1; the transformer_lens version is
    unrecorded. DEV's `.venv` has transformers 5.19.0 and no transformer_lens.
14. `get_tokenizer` keys on the `HF_TOKEN` env var only. A machine logged in with `hf auth login` but without the
    env var silently gets the FallbackTokenizer.

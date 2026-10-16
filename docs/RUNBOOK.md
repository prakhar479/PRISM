# PRISM Runbook

This is the one page to read: how to run the experiments, what each step does, how to read the results, and what to do next.

Everything goes through `./run.sh <command>`.

---

## 1. Quick start

**Local machine** (no GPU needed). This checks the code before you copy it to the GPU box.
```bash
./run.sh setup        # creates .venv and installs requirements
./run.sh test         # unit tests + mock smoke test, about 15 s
./run.sh dryrun       # the whole overnight schedule with a fake model, about 90 s
```

**GPU server** (24 GB):
```bash
git clone <this repo> && cd prism_sae_dark_matter
./run.sh setup
export HF_TOKEN=hf_...          # needs accepted access to google/gemma-2-2b
./run.sh start                  # runs in the background; 6.5 h budget (--hours N to change)
./run.sh status                 # any time: stage, last log lines, GPU usage
```

**After about 10 minutes, check that tasks survived:**
```bash
./run.sh status
cat results/overnight_*/preflight/preflight.json | head -40
```
If an algorithmic task is listed under `drop_tasks`, read section 5 (Decision rules, rule S2) before going to sleep.

**In the morning:**
```bash
./run.sh decide       # verdict on H1/H2/H3 + next actions -> DECISIONS.md
./run.sh results      # paths of INDEX / SUMMARY / figures
```

**If the run crashed or the machine rebooted:** `./run.sh resume`. Finished work is skipped.

---

## 2. What the overnight run does (in order)

| # | Stage | Time | Output | Purpose |
|---|---|---|---|---|
| 0a | Environment checks | 1 min | log | GPU, HF access, library versions |
| 0b | GPU smoke test | 2 min | `smoke/` | One layer and one SAE end to end on the real model |
| 0c | **Preflight** | 5–10 min | `preflight/preflight.json` | Per task: Gemma tokenization, prompt format × few-shot grid, and which setting the model solves |
| 0d | Prefetch | 10–30 min | `prefetch/prefetch.json` | Downloads the model and every SAE up front |
| 1 | `tier1_core` | ~1 h | `tier1_core/` | **H1–H3 first answer**: exp1 at 6 layers × {16k, 65k}; exp2 and exp3 at L12/16/20 |
| 2a | `tier2_layers` | ~30 min | `tier2_layers/` | exp1 at all 26 layers (16k) |
| 2b | `tier2_widths` | ~1–1.5 h | `tier2_widths/` | Layer 12 at widths 16k → 1m |
| 3 | `tier3_expansions` | ~1.5–2 h | `tier3_expansions/` | n = 1000/500, second seed, generic-text error subspace |

Each tier writes `metrics/`, `figures/` and `SUMMARY.md` before the next one starts. A tier that is still running when the deadline passes stops cleanly, and the tiers after it are skipped.

### Results layout
```
results/overnight_<stamp>/
  STATUS            one line per stage with its exit code
  INDEX.md          links to every tier summary
  DECISIONS.md      ./run.sh decide output
  preflight/ prefetch/ smoke/
  tier1_core/
    SUMMARY.md      all key numbers as tables
    figures/        fig1..fig7 (.png + .pdf)
    metrics/        exp{1,2,3}_results.{json,csv}, *_units.jsonl (resume log), failures.jsonl
    tensors/        PCA/LDA coordinates, spectra, fitted subspaces U_r (.npz)
    run_info.json   config, per-task pair counts, timings
logs/prism_<stamp>.log   full console log
```

---

## 3. The experiments in one paragraph each

**Notation**
- x is the residual stream at a layer.
- The SAE gives a reconstruction x̂ and the error ε = x − x̂.
- Every task has a *clean* prompt and a *corrupt* prompt. They differ in one token, so the correct answer changes.
- **NLDR** measures how much of the clean answer a patch restores in the corrupt run: (ΔL_patched − ΔL_corrupt) / (ΔL_clean − ΔL_corrupt). A value of 0 means nothing was restored and 1 means the clean answer was fully restored.

**Exp1, causal split (H1).** This patches parts of the clean activation into the corrupt run.

| Intervention | What is patched in | What a high NLDR means |
|---|---|---|
| `sae_only` | x̂_clean + ε_corrupt | The state is in the SAE features |
| `error_only` | x̂_corrupt + ε_clean | The state is in the error ("dark matter") |
| `clean_full` | x_clean | Sanity check. It should be ≈1 with `from_diff` |
| `noise_control`, `sae_splice` | (see table note) | Erase the state instead of setting it, so ≈0.5 is normal |
| `error_resample_same_state` | ε from *another* prompt with the same answer | High: ε encodes the state itself, not the prompt |
| `error_resample_diff_state` | ε from a prompt with a different answer | Should be low |
| `mean_error` | the average ε | Should be low |

`noise_control` patches x̂_corrupt + norm-matched noise; `sae_splice` patches x̂_clean.

There are two patch modes:
- `last` patches only the final token.
- `from_diff` patches every token from the first one that differs.

**Exp2, geometry (H2).**
- Linear probes predict the state from x, x̂, ε and the SAE latents.
- They are trained on the train split and scored on the test split. The R² for mod_sum is circular.
- `selectivity` is accuracy minus the accuracy with shuffled labels.
- SVD spectra show how many directions of ε the state needs.

**Exp3, hybrid repair (H3).**
- The patch is x̂_clean + U_r U_rᵀ ε_clean + (I − U_r U_rᵀ) ε_corrupt. It adds back only a rank-r slice of the clean error.
- U_r is fitted on the train split only.

| Method for U_r | Role |
|---|---|
| `supervised_centroid_svd` | Uses the state labels |
| `unsupervised_svd` | No labels |
| `span_svd` | Fitted on all tokens from the first differing one |
| `generic_text_svd` | Fitted on ordinary web text (task-agnostic) |
| `transfer_from_<task>` | Uses another task's subspace |
| `random_subspace` | Null baseline |
| `top_xhat_svd` | Takes directions from the reconstruction instead of ε (wrong-space baseline) |

`reference` is full `error_only`.

---

## 4. Reading the results: figure by figure

| Figure | Question | What "dark matter is real" looks like |
|---|---|---|
| fig1_causal_split | Where does each part carry the state, by layer? | Algorithmic tasks: orange (`error_only`) ≫ blue (`sae_only`). IOI: the reverse |
| fig6_positions | Is the state hidden at earlier tokens? | `from_diff` curves lie above `last` at early layers. `clean_full` (from_diff) ≈ 1 everywhere |
| fig2_error_manifolds | Does ε have shape? | The ε panel shows ordered states (a ring for mod_sum); the x̂ panel is a blob |
| fig3_spectrum_and_probes | Low-rank and decodable? | ε reaches high cumulative variance in few directions; the ε probe bars beat x̂ |
| fig4_hybrid_pareto | How small a subspace repairs it? | Curves reach the dotted line by r ≈ 4–16, far above the grey random line |
| fig7_subspace_methods | Which subspace works? | Supervised and unsupervised SVD ≫ random. Interesting either way: generic ≈ task-specific? Does transfer work? |
| fig5_width_scaling | Do bigger SAEs absorb the state? | The `error_only` NLDR stays flat as width grows, so it is a blind spot, not just missing capacity |

Every NLDR in `SUMMARY.md` is written as `value [lo, hi]`, a 95% bootstrap CI. Two numbers differ only if their CIs don't overlap.

---

## 5. Decision rules (the task action decider)

`./run.sh decide` applies these rules automatically. This table is the same logic, so you can apply it by hand.

### Setup checks (do these first)

| ID | Observation | Meaning | Action |
|---|---|---|---|
| S1 | `STATUS` shows a non-zero exit, or `failures.jsonl` lists OOM or other errors | Part of a tier did not run | `./run.sh resume`. Missing-SAE errors for some widths are normal and can be ignored |
| S2 | A task is in `drop_tasks` | gemma-2-2b cannot solve it | Make it easier in `src/data/synthetic_tasks.py`: mod_sum `n_steps` 4→2, graph `n_edges` 4→3, dyck `n_ops` 8→6. Then rerun |
| S3 | Retention < 0.3 or `filter_mode: clean_only` | Few clean pairs, so wide CIs | Results are still valid but noisy. Tier 3 (larger n) helps |
| S4 | `clean_full` with `from_diff` < 0.9 | **Patching is broken** | Stop and interpret nothing. Check the hook name and dtype in `run_patched` |

### Hypotheses

| ID | Observation | Verdict | Next |
|---|---|---|---|
| H1a | `error_only` > `sae_only` with separated CIs, and `error_only` > 0.5 on algorithmic tasks | **H1 supported** | Go to H2/H3 |
| H1b | Same direction, but the CIs overlap | Weak | Use tier 3 numbers (n = 500), or rerun with more `n_test` |
| H1c | `sae_only` ≥ `error_only` | **H1 rejected** for that task | Negative result; report it |
| H1d | IOI control also has `error_only` > `sae_only` | The SAE is lossy for everything here | H1 is not specific to algorithmic state. Weaken the claim |
| H1e | `resample_same_state` ≈ `error_only` ≫ `resample_diff_state` | ε encodes the *state* | Strong evidence for a state code in ε |
| H1f | `resample_same_state` is also low | ε carries prompt-specific detail | Don't claim a "state manifold" |
| H2a | ε probe beats x̂ by > 0.1 and ε selectivity > 0.2 | **H2 supported** | Report the rank90 / participation ratio from `exp2_spectra.csv` |
| H2b | x̂ probe ≥ ε probe | The state is decodable from the features but not causally used | Interesting dissociation. Compare with exp1 |
| H3a | The smallest r reaching 90% of `reference` is ≤ 16, and random at that r < half of it | **H3 supported** | Note that r·2304 params ≪ widening the SAE (the `overhead` field) |
| H3b | Random ≈ SVD at the same r | Repair is not specific to these directions | H3 not supported |
| H3c | `generic_text_svd` ≈ supervised | Universal dark directions | Strongest claim. Verify across layers |
| H3d | `transfer_from_X` works | Shared state code across tasks | A follow-up paper direction |
| W1 | `error_only` stays flat 16k → 1m | A true blind spot | Headline result |
| W2 | `error_only` drops with width | Capacity limit | Frame the result as "needs width ≥ X" |

---

## 6. Changing things

- **Run only some tiers:** `./run.sh start --tiers tier1_core,tier2_widths`. Use `--hours 3` for a shorter budget.
- **Edit a tier:** `configs/tier*.yaml`. Each one inherits `configs/default_config.yaml` through `base:`.
- **One-off run:**
  ```bash
  .venv/bin/python scripts/run_full_suite.py --config configs/tier1_core.yaml --layers 12 --widths 16k \
      --overrides results/overnight_X/preflight/preflight.json --out results/my_run --set n_test=500
  ```
- **Regenerate figures and summary only:** `run_full_suite.py ... --experiments plots --out <existing dir>`.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `HF access ... failed` | Accept the gemma-2-2b licence on huggingface.co with the same account as the token |
| CUDA OOM on the 1m width | Remove `"1m"` from `configs/tier2_widths.yaml`, or lower `batch_size` |
| `datasets` import warning | `pip install datasets`. Otherwise only `generic_text_svd` is skipped |
| Run takes far longer than estimated | `./run.sh status`. Lower `n_train`/`n_test` or the number of layers in the tier configs, then `./run.sh resume` |
| Plot error in the log | Metrics are still saved. Fix the plot, then use `--experiments plots` |

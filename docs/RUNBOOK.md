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

**GPU machine** (8 GB or more; memory settings adapt automatically, see section 6a). Start with the pilot:
```bash
git clone <this repo> && cd prism_sae_dark_matter
./run.sh setup
export HF_TOKEN=hf_...          # needs accepted access to google/gemma-2-2b
./run.sh pilot --foreground     # ~20-40 min: is the project viable? ends with a GO / MAYBE / NO-GO verdict
./run.sh decide                 # re-print the verdict and the per-hypothesis checks (DECISIONS.md)
./run.sh start                  # after a GO: the full overnight run in the background (6.5 h, --hours N to change)
./run.sh status                 # any time: stage, last log lines, GPU usage
```

**The pilot** (`configs/pilot.yaml`, results in `results/pilot_<time>/`) runs the same four tasks with their
validated settings, but only layers 6/12/18, the 16k SAE (already cached), 160/80 train/test pairs and the
essential controls (clean-patch sanity, same/different-state error resampling, random-subspace null, prompt-token
probe). Preflight is the quick version (48 pairs, 2 and 8 shots) and still aborts on a failed IOI gate.
`DECISIONS.md` opens with the verdict:

| Verdict | Rule | Next |
|---|---|---|
| ⛔ BLOCKED | IOI gate, SAE CE fidelity < 0.7, clean-patch sanity < 0.9, or no task kept enough pairs | fix that first (section 7, `scripts/check_forward.py`) |
| 🟢 GO | for ≥ 1 state task `error_only` > `sae_only` with separate CIs and NLDR > 0.5, and IOI stays in x̂ | `./run.sh start` |
| 🟡 MAYBE | `error_only` > `sae_only` but CIs overlap, or IOI also lives in the error | `./run.sh start --tiers tier1_core --hours 2`: 400/200 pairs at 6 layers x {16k, 65k} for tighter CIs |
| 🔴 NO-GO | `sae_only` ≥ `error_only` for every task | negative H1; check the other layers in `pilot/SUMMARY.md` first |

The per-task table under the verdict also shows H2 (does ε decode the state better than x̂ *and* than the prompt
tokens alone) and H3 (smallest rank whose error subspace restores 90%, against a random subspace).
`./run.sh resume` on a pilot folder continues the pilot only.

**The paper runs** (`./run.sh paper --foreground`, ~3-4 h on 8 GB, an estimate; results in `results/paper_<time>/`).
The pilot rejected the original claim (the SAE reconstruction carries 75-100% of the causal state), so the paper is
*"Where do SAEs lose task state? A causal audit"*. Three tiers, most decisive first:

| Tier | Question | Read it in DECISIONS.md |
|---|---|---|
| `paper_residue` | Where does the SAE miss part of the state, and is that piece low-rank and task-specific? (L10-18; error subspace vs random, ordinary-text and other-task subspaces) | H3 |
| `paper_widths` | Does the missed piece survive in wider SAEs (16k to 262k at L12)? Stays >= 0.10: a blind spot. Shrinks: capacity | Width scaling |
| `paper_audit` | Full-size H1/H2 and the three pitfalls: last-token vs from_diff patching, probes vs prompt tokens, readable-but-unused state | H1, H2 |

All three cap the token budget at 2048 and leave out the 1m SAE (needs ~9.4 GB free RAM). A CUDA out-of-memory
error now halves the token budget and retries (up to 3 times) instead of failing the step.

**After about 10 minutes, check that tasks survived:**
```bash
./run.sh status
cat results/overnight_*/preflight/preflight.json | head -40
```
If `STATUS` says **IOI GATE FAILED**, the model or pipeline is broken: stop and run `scripts/check_forward.py`
(rule S0). If a task is listed under `drop_tasks`, read rule S2 before going to sleep.

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
| 0c | **Preflight** | 5–10 min | `preflight/preflight.json` | Per task: Gemma tokenization, few-shot grid, and the setting that keeps the most pairs. **Aborts the run if the IOI control fails** (exit 3). Reused on `--resume` only if its fingerprint matches the current task code and config |
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

**Tasks** (validated with `scripts/analyze_tasks.py`; settings in `configs/default_config.yaml`)

| Task | Example (clean) | State | Setting | gemma-2-2b top-1 / kept |
|---|---|---|---|---|
| `add_chain` | `Start with 2. Add 3. Add 1. Add 2. Total =` → 8 | running sum | depth 3, 2 shots | 0.99 / 0.99 |
| `entity_track` | `The key is in box B. The cup is in box E. The key moves to box C. … The key is in box` → C | object location | depth 2, 8 shots | 0.98 / 0.84 |
| `var_binding` | `t = 7. q = 4. x = q. q = 2. x =` → 4 | bound value | depth 1, 8 shots | 0.95 / 0.95 |
| `ioi_control` | `When Mary and John went to the store, John gave a drink to` → Mary | indirect object | 2 shots | 1.00 / 1.00 |

The corrupt prompt changes one step (an increment, which object moved, the source value, the subject), so the answer
changes. `add_chain` also has a story wording (`Tom has 2 apples. Tom gets 3 more. … Tom now has`) to fill the splits;
`gen_kwargs: {templates: [calc]}` turns it off. The legacy `mod_sum` / `dyck_depth` / `graph_reach` are beyond
gemma-2-2b at every difficulty (results/task_analysis/report.md).

Probe caveat: a linear probe on the prompt's tokens alone (exp2 source `surface`, no model at all) decodes
`var_binding` and `ioi_control` perfectly and the `add_chain` total with R² = 1, so for those tasks a probe cannot
show that the model computed the state. `entity_track` is the clean case (prompt tokens: 0.36 vs 0.89 for the
layer-18 residual on gemma-2-2b). Base probe (exp2) claims on `entity_track`, and only where ε beats `surface`;
exp1/exp3 are causal and unaffected.

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
| fig2_error_manifolds | Does ε have shape? | The ε panel shows ordered states (an ordered arc for add_chain); the x̂ panel is a blob |
| fig3_spectrum_and_probes | Low-rank and decodable? | ε reaches high cumulative variance in few directions; the ε probe bars beat x̂ |
| fig4_hybrid_pareto | How small a subspace repairs it? | Curves reach the dotted line by r ≈ 4–16, far above the grey random line |
| fig7_subspace_methods | Which subspace works? | Supervised and unsupervised SVD ≫ random. Interesting either way: generic ≈ task-specific? Does transfer work? |
| `metrics/sae_fidelity.csv` | Are the SAEs themselves good? | CE recovered > 0.8 everywhere. Then a large ε effect is *despite* a good SAE |
| fig5_width_scaling | Do bigger SAEs absorb the state? | The `error_only` NLDR stays flat as width grows, so it is a blind spot, not just missing capacity |

Every NLDR in `SUMMARY.md` is written as `value [lo, hi]`, a 95% bootstrap CI. Two numbers differ only if their CIs don't overlap.

---

## 5. Decision rules (the task action decider)

`./run.sh decide` applies these rules automatically. This table is the same logic, so you can apply it by hand.

### Setup checks (do these first)

| ID | Observation | Meaning | Action |
|---|---|---|---|
| S1 | `STATUS` shows a non-zero exit, or `failures.jsonl` lists OOM or other errors | Part of a tier did not run | `./run.sh resume`. Missing-SAE errors for some widths are normal and can be ignored |
| S0 | `STATUS`: **IOI GATE FAILED** (preflight exit 3) | gemma-2-2b solves IOI, so the model or pipeline is broken | Run `scripts/check_forward.py` (compares our forward pass with Hugging Face's) and fix that first |
| S2 | A task is in `drop_tasks` | gemma-2-2b fails it at this difficulty under every shot count | `scripts/analyze_tasks.py --tasks <task>` finds a difficulty it can do; set `task_overrides.<task>.gen_kwargs` and rerun |
| S3 | Retention < 0.3 | Few clean pairs, so wide CIs | Results are still valid but noisy. Tier 3 (larger n) helps |
| S4 | `clean_full` with `from_diff` < 0.9 | **Patching is broken** | Stop and interpret nothing. Check the hook name and dtype in `run_patched` |
| S5 | `sae_fidelity`: CE recovered < 0.7 for an SAE (Gemma Scope normally gives > 0.8) | That SAE was loaded or encoded wrongly | Ignore its rows. Check `resolve_sae_id` / `RealSAE.encode` |

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
- **Change a task's difficulty or shots:** `task_overrides.<task>` in `configs/default_config.yaml` (`gen_kwargs:
  {depth: N}`, `n_shots`). Check it first with `scripts/analyze_tasks.py --tasks <task>` (~5 min): a setting is
  usable when it is *viable* there (top-1 ≥ 0.6, ≥ 30% of pairs kept, beats the shortcut baselines).
- **Something looks wrong with the model itself:** `scripts/check_forward.py` (~5 min) checks the forward pass against
  Hugging Face, the tokenizer, prompt-space sizes and few-shot prefixes.

## 6a. GPU memory and speed (8 GB and up)

Nothing needs to be set by hand. At startup the run logs a `runtime plan` line, which is also saved in `run_info.json` under `runtime`.

| Mechanism | What it does | Effect |
|---|---|---|
| Last-token logits | The model runs to its final residual, and the vocabulary projection is applied to the last token only | Removes the ~2 GB `[batch, tokens, 256k]` logits tensor: this was the 8 GB OOM |
| `offload_embed` (auto on GPUs < 16 GB) | The 256k × 2304 embedding table stays in CPU RAM | Saves 1.2 GB VRAM, bit-identical results |
| `max_batch_tokens` (auto) | Tokens per forward batch, computed from the VRAM left after loading the model | About 2–3k on 8 GB, 16k on 24 GB |
| `sae_placement` (auto) | An SAE that does not fit next to the activations stays in CPU RAM; weight chunks are streamed to the GPU | Any width, up to 1m, runs on 8 GB. Same arithmetic, so the numbers do not depend on GPU size |
| `resume_from_layer` | A patched run restarts at layer+1 from the cached residual instead of recomputing layers 0..layer | Roughly halves patching time. Checked against full forwards at startup and switched off if they differ (`resume_check_max_abs_diff` in `run_info.json`) |
| `prefetch_saes` | The next SAE is read from disk on a background thread | Hides SAE load time |
| `cpu_workers` | exp2 probe fitting runs on worker threads while the GPU continues | CPU and GPU busy at the same time |

SAEs need CPU RAM: 1m is about 9.4 GB (bf16). If there is not enough free RAM, that width is skipped and listed in `failures.jsonl` (rule S1). To force a setting, use, e.g., `--set runtime='{max_batch_tokens: 1024}'`.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `HF access ... failed` | Accept the gemma-2-2b licence on huggingface.co with the same account as the token |
| CUDA OOM | Check the `runtime plan` line in the log. Rerun with `--set runtime='{max_batch_tokens: 1024, sae_placement: stream}'`, then `./run.sh resume` |
| `MemoryError: SAE ... needs N GiB RAM` | Not enough system RAM for that width (usually 1m): close other programs, or remove it from `configs/tier2_widths.yaml` |
| `datasets` import warning | `pip install datasets`. Otherwise only `generic_text_svd` is skipped |
| Run takes far longer than estimated | `./run.sh status`. Lower `n_train`/`n_test` or the number of layers in the tier configs, then `./run.sh resume` |
| Plot error in the log | Metrics are still saved. Fix the plot, then use `--experiments plots` |

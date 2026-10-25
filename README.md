# PRISM — Probing Residual Information & State Manifolds

**Start here:** `./run.sh help` and [docs/RUNBOOK.md](docs/RUNBOOK.md) (steps, how to read results, decision rules).

Tests whether Gemma Scope SAEs (`gemma-scope-2b-pt-res-canonical`, gemma-2-2b) are blind to multi-step algorithmic
state, which then lives in the reconstruction error `ε = x − x̂`:

* **Exp 1** — causal split: patch the corrupt run at `blocks.{l}.hook_resid_post[-1]` with `sae_only`, `error_only`,
  `sae_splice`, `noise_control`; metric = NLDR.
* **Exp 2** — geometry of ε: SVD spectra, ridge/circular state probes, PCA/LDA coordinates (train-fit, test-eval).
* **Exp 3** — hybrid repair `x̂ + U_r U_rᵀ ε`, `r ∈ {0..64}`, `U_r` fit on train errors only.

Tasks: `add_chain` (running sum), `entity_track` (object locations after moves), `var_binding` (value copied through
variables) and `ioi_control` (control). All four were validated on gemma-2-2b with `scripts/analyze_tasks.py`
(results/task_analysis/report.md). The first suite (`mod_sum`, `dyck_depth`, `graph_reach`) is kept for the mock world
and tests only: gemma-2-2b is at chance on it at every difficulty and shot count, which is why the 2026-10-19 run
kept 0 pairs.

## Local CPU (no GPU, no weights, no HF token)
```bash
pip install -r requirements.txt        # transformer_lens / sae_lens are NOT needed for --mock
python -m pytest tests/test_pipeline_cpu.py -q
python scripts/run_smoke_test.py --mock            # ~20 s; figures in results/smoke_test_mock/figures/
python scripts/run_smoke_test.py --mock --mock-d-model 2304
python scripts/run_full_suite.py --mock --n-train 400 --n-test 200      # full-size mock run
```
The mock world puts the algorithmic state in a reserved low-rank "dark" plane that every mock SAE atom is
orthogonal to (state → ε only; one 2-d plane per algorithmic task), while the IOI signal is a dictionary atom
(state → x̂). The state is written progressively across layers, so patching layer *l* recovers ≈ sigmoid((l−8)/2) of
the logit difference. Mock numbers verify the plumbing only — they say nothing about Gemma.

## Remote GPU server
```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...                  # gemma-2-2b is gated
bash scripts/run_server_gpu.sh          # checks nvidia-smi/HF_TOKEN, GPU smoke test, then full suite under nohup
bash scripts/run_server_gpu.sh --foreground --skip-smoke --config configs/default_config.yaml
```
Outputs: `results/full_<timestamp>/{metrics,tensors,figures,logs}`; logs in `logs/prism_<timestamp>.log`.
Production details: bf16 on CUDA (fp32 on CPU/MPS), one SAE resident at a time (freed + `empty_cache()` between
loads), all forwards in `inference_mode`, SAE math in fp32. The model is loaded with
`HookedTransformer.from_pretrained_no_processing` so the residual stream matches what Gemma Scope was trained on.
Candidate pairs are filtered by the base model (`filter_valid_pairs`; `strict`: both targets predicted and clean
ΔL > 2, `clean_only`: clean target predicted and ΔL_clean − ΔL_corrupt > 2). Tasks with fewer than `min_pairs`
survivors are skipped with a warning. Per-task settings (`task_overrides`: `n_shots`, `filter_mode`, `gen_kwargs` =
difficulty) live in the config; preflight refines them. The few-shot prefix is drawn before the splits, so it is
never starved by a small prompt space.

## Notes on definitions
* `noise_control` = `x̂_corrupt + η` replaces ε_corrupt with norm-matched noise, so it *erases* the corrupt state
  (NLDR ≈ 0.5 where the state is fully written) rather than leaving it untouched. Compare it against `error_only`.
* `sae_splice` likewise erases rather than sets the state.
* Pair counts / graph size: `graph_reach` uses 6 nodes and 4 edges (the 4-node template yields too few unique prompts
  for disjoint 400/200 splits).
* Train/test splits are prompt-disjoint (clean and corrupt prompts); all SVD/PCA/LDA/probe fits use train only.

## Pilot run (any 8 GB+ GPU, ~20–40 min)
```bash
HF_TOKEN=hf_... ./run.sh pilot --foreground    # 3 layers x 16k, 160/80 pairs -> results/pilot_<time>/DECISIONS.md
```
Ends with a GO / MAYBE / NO-GO / BLOCKED verdict on whether the full run is worth it (docs/RUNBOOK.md, section 1).

## Overnight run (one 24 GB GPU, ~4–5 h of a 6.5 h budget)
```bash
HF_TOKEN=hf_... bash scripts/run_server_gpu.sh            # nohup; prints log path + results dir
cat results/overnight_*/STATUS                             # progress
bash scripts/run_server_gpu.sh --resume results/overnight_XXXX   # after a crash: finished units are skipped
bash scripts/run_server_gpu.sh --mock --foreground         # local CPU dry run of the whole schedule (~90 s)
```
Stages, in priority order. Each tier writes `metrics/`, `figures/` and `SUMMARY.md` before the next tier starts.

| stage | what | why |
|---|---|---|
| smoke + preflight | for each task: Gemma tokenization, `answer_format` × `n_shots` grid, filter retention → `preflight/preflight.json` | stops the strict filter from silently removing the algorithmic tasks |
| prefetch | download the model and every SAE used by the tiers | network or gated-access errors show up in minutes |
| `tier1_core` | exp1 at 6 layers × {16k,65k} × {last, from_diff} plus controls; exp2 and exp3 at L12/16/20 with null and transfer subspaces | H1–H3 first pass |
| `tier2_layers` | exp1 at all 26 layers, 16k | dense layer curve |
| `tier2_widths` | L12 at 16k→524k (≥524k in bf16) | does the state stay in ε as width grows? (fig5) |
| `tier3_expansions` | n = 1000/500, seed 1, generic-text ε subspace | task-specific vs generic dark matter |

New in v2:
- `from_diff` patching (every position from the first differing token).
- Paired-bootstrap 95% CIs (`nldr_lo`/`nldr_hi`).
- exp1 controls: `clean_full`, `error_resample_same_state` / `diff_state`, `mean_error`.
- exp2: shuffled-label selectivity and an SAE-latent probe.
- exp3 baselines and variants: `random_subspace`, `top_xhat_svd`, `span_svd`, `generic_text_svd`, `transfer_from_<task>`.
- Resumable work units (`metrics/*_units.jsonl`); per-unit failure isolation (`metrics/failures.jsonl`); a `--deadline`.

# What Sparse Autoencoders Lose When a Model Tracks State

Research artifact for an anonymous submission: the code, data, raw outputs, analyses and figures behind every number in the
paper.

**Model and SAEs:**
- gemma-2-2b (Hugging Face revision `c5ebcd40d208330abc697524c919956e692655cf`);
- Gemma Scope residual-stream JumpReLU SAEs (`google/gemma-scope-2b-pt-res`, revision
  `fd571b47c1c64851e9b1989792367b9babb4af63`).

## The question in one paragraph
An SAE splits the residual stream into a sparse reconstruction from features, x̂, and an error, ε = x − x̂. We ask, causally,
how much task state lives in ε, on three tasks:
- running sum (`add_chain`);
- object tracking (`entity_track`);
- person–city binding (`people_cities`), in English and four other languages.

The method patches a clean prompt's x̂ and/or ε into a corrupt prompt, and measures answer flips and logit differences.
Directions estimated from the error, and directions from distributed alignment search (DAS), localise what is lost.
Findings, figures and claim-by-claim evidence: [`paper/PAPER_FINDINGS.md`](paper/PAPER_FINDINGS.md).

## What is preregistered
- [`PREREGISTRATION.md`](PREREGISTRATION.md) and [`PREREGISTRATION_ADDENDUM.md`](PREREGISTRATION_ADDENDUM.md) were frozen on
  2026-11-16, before the corresponding results existed. `tests/harness/test_prereg_frozen.py` pins their hashes.
- The decision constants are in `harness/final.py`.
- Everything not named there is exploratory and labelled as such. The analyses of record are:
  - `analysis/final_pass_results.md` (preregistered answers);
  - `analysis/team_analysis_2026-11-19.md` (exploratory synthesis, independently audited in `analysis/team/audit.md`).
- `docs/DECISIONS.md` is the dated lab notebook: every design decision, deviation and correction, in order.

## Repository layout
| path | contents |
|---|---|
| `harness/` | the experiment harness: conditions, directions (error PCA, answer separation, DAS), metrics, caching, multi-GPU scheduler, analysis primitives |
| `src/` | model runtime and task generators (from the pilot; frozen) |
| `configs/experiments/E*.yaml` | one config per experiment (E0–E22); `configs/sae_inventory.json` lists the SAE variants |
| `queue/*.yaml` | the job queues that were run on the GPU servers |
| `data/` | prompt sets (`prompt_sets/`, train/test pairs with answers) and multilingual text and task packs (`lang/`, with sources, licences and content hashes in `MANIFEST.json` and `docs/multilingual_corpora.md`) |
| `golden_parts/` | golden fixtures from the pilot, used by the preflight regression check (reassembled by `scripts/setup_server.sh`) |
| `results/runs/E*/<run_hash>/` | raw outputs of every complete run: `manifest.json` (config, package versions, hardware, timings), `per_pair.parquet` (one row per test pair × condition, with logits of all candidate answers), `directions/` (fitted subspaces, DAS rotations), `summary.*`, logs; E12 runs hold per-window fidelity arrays |
| `results/runs/_queue/` | queue state and preflight reports of each server run |
| `results/paper_20261026_001341/` | pilot metrics and the preflight reference the harness checks against |
| `tables/` | tables generated from `results/runs` (`tables/final/` for E9–E22; `tables/SCHEMA.md` documents the columns) |
| `analysis/` | written analyses; `analysis/team/` and `analysis/deep/` hold the exploratory reports and their scripts |
| `paper/` | the findings for the paper and figures F1–F7 (PDF, PNG and the exact plotted data as CSV) |
| `scripts/` | entry points: running (`run_queue.py`, `slurm_final.sbatch`, `setup_server.sh`), shipping (`bundle_results.py`, `ingest_published.py`), tables (`make_*.py`) |
| `tests/` | unit and end-to-end tests (`pytest tests/harness`; a tiny random model is used, no GPU needed) |

## Experiments
| id | what | status |
|---|---|---|
| E0 | pilot regression through the harness | sanity check |
| E1, E2, E2r | causal split of task state into features and error; error-slot baselines; a seed replicate | E2 confirmatory cells |
| E3 | fresh template families (the RQ1 gate) | **preregistered** |
| E4 | SAE width at matched L0 (16k–524k) | preregistered rule |
| E5 | sub-threshold feature add-back | exploratory |
| E6, E8 | layer sweep; 36-cell atlas with repair specificity | preregistered rule (E8) |
| E7 | error-direction estimator ablation | exploratory |
| E9, E9b | illusion checks and DAS (running sum L18 confirmatory) | **preregistered** (addendum) |
| E10, E10b | steering (Q2) | **preregistered** (addendum) |
| E11 | people-cities binding, layers 8–16 | exploratory |
| E12, E13, E14 | multilingual SAE fidelity (M1), task behaviour gates, multilingual gaps (M2/M3) | **preregistered** (addendum) |
| E16–E22 | running-sum layer sweep; DAS robustness; rank up to 64; steering dose; width; 65k multilingual fidelity (stored as E12 runs); multilingual DAS transfer | exploratory, post hoc |

## Reproducing
See [`docs/REPRODUCING.md`](docs/REPRODUCING.md). In short:

```bash
# environment: see docs/REPRODUCING.md (exact lock file); CPU is enough for everything below
python -m pytest -q tests                                              # ~5 min, tiny random model, no GPU
python scripts/make_final.py --runs results/runs --out /tmp/tables     # regenerate the E9–E22 tables; compare with tables/final
python analysis/team/scripts/multilingual_load.py && python analysis/team/scripts/lead_capture.py
for f in analysis/team/scripts/fig_F*.py; do python "$f"; done         # figures to analysis/team/figures (= paper/figures)
```

Re-running the experiments themselves needs GPUs with at least 11 GB each. The queues used are in `queue/`, and the procedure
is in `docs/FINAL_EXPLORE_RUNBOOK.md`.

## Anonymization
Identities, user names, host names and home paths were redacted for review (`docs/REDACTION.md`). Git commit ids that appear in
run manifests refer to the development history, which will be published with the camera-ready version.

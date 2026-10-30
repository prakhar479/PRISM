# Implementation plan (Phase 1 → release 1), for review

Status: Phase 0 done (`REPO_MAP.md`, `docs/dev_env.md`, `RUN_REQUESTS.md`, `scripts/{probe_env,export_golden,
hf_agreement}.py`). Nothing below has been built yet.

## Architecture

- **`src/` stays frozen** as the pilot reference (tag `pilot-ref` = `430aa27`). Its tests keep passing. It is exempt
  from the "no `cuda` literals" rule; only `harness/` must obey it.
- **New package `harness/`** reuses pilot functions by import wherever the math must match bit-for-bit, rather than
  copying them:
  - `JumpReLUSAE` / `load_gemma_scope_npz` arithmetic;
  - task generators, `apply_prefix` and `tokenize_pairs`;
  - `runtime.head` / `last_logits_from`.
  
  Layout follows handoff §7. Model-dependent code lives in `harness/model/`; pure CPU code lives in
  `sae/ conditions/ directions/ metrics/ stats/ geom/`.
- **Backend: keep TransformerLens.** Our tiny test model is a random `HookedTransformerConfig` with Gemma-2
  features: RMS pre+post norms, both soft-caps, rotary, (1 + w) norm scaling, few layers, small d_model. A tiny
  random HF `Gemma2ForCausalLM` is used only to test the HF-comparison code path. This needs transformer_lens
  installed on DEV at the version RUN uses (R1).
- **Prompt sets become files.** A Stage 0 job (RUN) writes candidate pairs with token ids, filter scores and
  filter flags. Filters become flags; nothing is deleted. The pilot pairs come from R2's `pairs_*.jsonl.gz` and are
  committed under `data/pilot_pairs/` (a few MB), so DEV tests run on real pilot prompts and ids.
- **Stages 1–3** (handoff §6), each with content-addressed caches keyed by (model revision, prompt-set hash, layer,
  site, dtype [, SAE id]):
  1. Stage 1 caches residuals at `from_diff` spans (+ the last position), clean final-position outputs, and the
     unpatched B residual for resume.
  2. Stage 2 is the pure SAE pass.
  3. Stage 3 runs patched final-position forwards with resume-from-layer (the pilot's verified trick).
- **Outputs per (pair, condition):** post-softcap logits on the task's candidate answer set, plus full-vocab
  logsumexp, plus top-1 id. That is enough for every §7.7 metric. Per-pair parquet, with summaries derived on DEV.

## Commit sequence (each commit: tests green, small)

1. Environment:
   - `pyproject.toml` + `uv.lock` pinned to RUN's freeze (R1), with CPU/cu130 torch via uv sources;
   - `scripts/check_env.py`;
   - `.gitignore` for `incoming/ runs/ cache/ bundles/ golden/ hf_agreement/ profiles/local.yaml`;
   - `pytest.ini` markers `gpu`, `realmodel`, `golden`.
2. `harness/profile.py` + `profiles/{dev,run}.yaml`, `harness/device.py`, `harness/tolerances.py`, `DECISIONS.md`
   entries. Tests 17.
3. `harness/sae/`:
   - load by (layer, width, L0, revision);
   - one-shot / block-wise encode;
   - sparse near-threshold z;
   - `scripts/sae_inventory.py` (runs on DEV now) + `match_l0`.
   
   Tests 1, 13 and 20 (memory part).
4. `harness/tasks/`: pilot generators re-exported unchanged; `template_family_id`; slot spans; filter flags; the
   FallbackTokenizer stand-in. Tests 3, 8, 12.
5. `harness/conditions/` (all §7.5 rows) and `harness/directions/` (pilot `pca_pooled`/`answer_sep` reproduced,
   `pca_pos_centered`, `pca_slot_centered_pairdiff`, random, FitSet guard). Tests 4–7, 9, 14.
6. `harness/metrics/` + `harness/stats/`: D, NLDR (ratio of means **and** pair mean), p_A/p_B, flip rate,
   probability recovery, entropy, gaps; 10k paired + cluster bootstrap; Holm/BH. Tests 10, 11.
7. `harness/model/`: TL loader through `device.py`, cache stage, patch stage, OOM backoff. Tiny-model tests 2, 15
   (DAS stub), 16.
8. `harness/run.py` CLI, `--dry-run`/`--smoke`, manifest/provenance, run hash; `harness/memory.py` +
   `scripts/plan_runs.py`; `queue/` + `scripts/run_queue.py`. Tests 16, 17.
9. `scripts/preflight.py` (new, per §8.1; the pilot one is renamed `preflight_pilot.py` with `run.sh` updated),
   `scripts/prefetch.py` with revisions and hashes, `bundle_results.py` / `ingest_results.py`. Test 18.
10. T1 golden tests (after R2 returns): SAE encode/decode, conditions, U reproduction, metrics against
    `incoming/R2`. Test 19.
11. Configs E0, E1, E2 (+ E6, E7 so release 1 can queue them); dry-runs pass; `releases/run-YYYYMMDD-1.md`;
    `queue/release1.yaml` (prefetch → preflight → E0 → E1 → E2 → E7 → E6).

Rough size: steps 1–9 ≈ 2–3 days of work on DEV. Step 10 depends on R2.

## Proposed defaults (say so if any is wrong)

| topic | proposal | why |
|---|---|---|
| E0 NLDR | compare the **ratio of means** to the pilot (what it reported) and also store the pair mean | pilot definition (REPO_MAP §8) |
| E0 tolerance | ±1 NLDR point per cell (handoff); pair identity must match exactly via states + D within 0.05 | — |
| `pca_pooled` | exactly the pilot: last-position train errors, A+B rows | reproduce first, then add variants |
| noise | `feat_B_noise` keeps the pilot's ε_A-matched norm. New `feat_A_noise` matches ε_B (the error it replaces). Both recorded in DECISIONS.md | keep pilot reproducible |
| fidelity (E12) | report both mean-ablation (pilot) and zero-ablation CE-recovered | handoff asks to match the pilot first |
| random U | ≥ 20 seeds, distribution stored per seed (no CI averaging) | REPO_MAP §13.4 |
| bootstrap | 10k paired percentile, seed in manifest | handoff §7.8 |

## Decisions that belong to the humans

1. ~~Gemma access on DEV~~: resolved 2026-10-30.
2. **Reference stack**: decided 2026-10-30. DEV and RUN use the same environment; the lockfile is pinned to
   RUN's freeze from R1, and `check_env.py` verifies both machines against it.
3. **Tag and run R1–R3** on RUN once this Phase 0 is approved (`run-20261030-0`).
4. **Pushing.** Phase-0 work is committed on local branch `harness` and not pushed. RUN gets code via git, so it has
   to be pushed before RUN can check it out. OK to push `harness` and the tag to `origin`?
5. **PREREGISTRATION.md** is created verbatim from handoff §10 with the date placeholder. Edit it once, then tell me
   it is frozen. REPO_MAP §13.1 is relevant to its flip-rate definition.

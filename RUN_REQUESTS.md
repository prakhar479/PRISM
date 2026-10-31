# RUN_REQUESTS: what DEV needs from the RUN machine

Each request lists: id, status, purpose, exact commands (scripts in this repo), expected duration, expected outputs
and where to put them on DEV (`incoming/<id>/`, git-ignored). Status is one of `open / sent / returned / resolved`.

**Batching:** R1–R3 are meant to run in one sitting (~1–1.5 h total, mostly unattended) from one commit. The humans
tag that commit once they approve Phase 0 (tag: `run-20261031-1`; R1-R4 run from this one commit. `run-20261030-1` is superseded: its R2 OOMs on 8 GB). Nothing here modifies code, configs or
pilot outputs on RUN. Outputs go to new directories (`golden/`, `hf_agreement/`, `probe/`).

## Common setup (RUN)

```bash
cd <repo>                                   # the PRISM checkout that produced results/paper_20261026_001341
git status --short                          # must be clean (or note what is not)
git fetch && git checkout <TAG>             # tag given by the humans
PY=$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || { [[ -x ../.venv/bin/python ]] && echo ../.venv/bin/python; } || echo python )
export HF_TOKEN=hf_...                      # REQUIRED: without it the pilot code silently uses a toy tokenizer
                                            # (both scripts below refuse to run without it)
```

The same Python environment the pilot used must be active (no package changes before R1 has recorded it).

---

## R1: RUN environment probe and package freeze, `open`

- **Purpose:** pin the lockfile to the stack that produced the pilot numbers. The pilot environment's
  transformer_lens version was never recorded, and transformers 5.20.1 violates `requirements.txt`. Also confirm
  that no pilot outputs exist on RUN beyond what is committed.
- **Commands:**
  ```bash
  mkdir -p probe/R1
  $PY scripts/probe_env.py --out probe/R1/run_env.md --json probe/R1/run_env.json
  $PY -m pip freeze > probe/R1/pip_freeze.txt 2>&1 || uv pip freeze > probe/R1/pip_freeze.txt
  nvidia-smi > probe/R1/nvidia_smi.txt
  $PY -c "import transformer_lens, sae_lens, transformers, torch; print(transformer_lens.__file__); print(sae_lens.__version__, transformers.__version__, torch.__version__)" > probe/R1/imports.txt 2>&1
  { git rev-parse HEAD; git status --short; ls -la results logs; du -sh results/* logs; } > probe/R1/repo_state.txt 2>&1
  $PY -c "from huggingface_hub import scan_cache_dir as s; [print(r.repo_id, r.size_on_disk_str, sorted(x.commit_hash for x in r.revisions)) for r in s().repos]" > probe/R1/hf_cache.txt 2>&1
  tar cf probe_R1.tar probe/R1 && sha256sum probe_R1.tar > probe_R1.tar.sha256
  ```
- **Duration:** ~1 min (the probe downloads only gemma-2-2b config and tokenizer files, already cached).
- **Return:** `probe_R1.tar` + `.sha256` → DEV `incoming/R1/`.

## R2: golden-fixture export + pilot reproduction through the pilot code, `open` (first attempt on `run-20261030-1` OOMed 2026-10-31; fixed in `run-20261031-1`)

- **Purpose:**
  1. Rebuild the exact pilot pairs and verify them against the committed per-pair outputs.
  2. Re-run the pilot conditions for OT L12 and RS L18 (16k) through the unchanged pilot code on today's RUN stack.
     This gives an old-code E0 baseline and shows whether today's stack still reproduces the committed numbers.
  3. Export the golden fixtures for DEV T1 tests (handoff §4.3).
  4. Dump all scored candidate pairs of the four pilot tasks (prompts, token ids, filter scores, kept flags). This
     makes pilot prompt texts available on DEV for the first time.
- **Command:**
  ```bash
  $PY scripts/export_golden.py --out golden/R2
  ```
  Defaults: config `configs/paper_residue.yaml` + `results/paper_20261026_001341/preflight/preflight.json`; cells
  `entity_track:12,add_chain:18`; 16 golden pairs (seed 20261030); dump tasks: all four.
- **Duration (estimate from the paper run's timings):**
  - pair rebuilding: OT ~9 min, RS ~3 min, var_binding ~1.5 min, IOI ~0.5 min;
  - pilot-condition re-runs: ~5 min;
  - golden forwards: ~2 min.
  
  Total ≈ 25–35 min. GPU memory ≈ the pilot's (16k SAEs only).
- **Outputs:** `golden/R2/{report.md, manifest.json, pilot_reproduction.csv, golden_entity_track_L12.npz,
  golden_add_chain_L18.npz, pairs_<task>.jsonl.gz, export.log}`, plus `golden/R2.tar` and `golden/R2.tar.sha256`.
  Expected size < 200 MB, mostly residuals and two full-vocab logit vectors per pair.
- **Exit code:** 0 if all checks pass, 1 if any check FAILs. **Return the outputs either way.** A FAIL is
  information for DEV (e.g. pairs not reproducible on today's stack), not something to fix on RUN.
- **Return:** `golden/R2.tar` + `.sha256` → DEV `incoming/R2/`.

## R3: HF top-1 agreement on the pilot task prompts, `open` (needs R2's output on RUN)

- **Purpose:** investigate the "95% top-1 agreement with HF" figure (REPO_MAP §10). It was measured on the legacy
  tasks at an older commit. This request measures, on the pilot tasks' kept test pairs plus rejected candidates:
  - agreement rate;
  - logit margins of the disagreements and whether they are near-ties;
  - |ΔD|;
  - whether the pilot filter would keep different pairs under HF logits;
  - a float32 HF reference on 24 prompts, to see which bf16 path is closer to fp32.
- **Command:**
  ```bash
  $PY scripts/hf_agreement.py --pairs-dir golden/R2 --out hf_agreement/R3
  tar cf hf_agreement_R3.tar hf_agreement/R3 && sha256sum hf_agreement_R3.tar > hf_agreement_R3.tar.sha256
  ```
  The fp32 reference needs ≥ 14 GiB free RAM and is skipped with a warning otherwise. `--fp32-subset 0` turns it
  off.
- **Duration:**
  - ours: ~3–5 min;
  - HF bf16, batch 1, ≤ 600 pairs × 2 prompts × 4 tasks: ~15–25 min;
  - fp32 CPU subset: ~5–10 min.
  
  Total ≈ 30–40 min. The models are loaded one at a time.
- **Outputs:** `hf_agreement/R3/{report.md, per_prompt.csv, manifest.json, hf_agreement.log}`.
- **Return:** `hf_agreement_R3.tar` + `.sha256` → DEV `incoming/R3/`.

## Report back with the tars

Tag/commit used, start/end times, exit codes of R2 and R3, and anything unusual (OOM, retries, warnings in the
logs).

---

## R4: release 1 queue, `open` (can run in the same sitting as R1-R3)

- **Purpose:** E0, E1, E2, E7, E6 (handoff §11, RT1).
- **Commands:** `releases/release1.md` (separate `.venv-harness`, then `scripts/run_queue.py --queue
  queue/release1.yaml`). Preflight uses R2's `golden/R2` on RUN, so no DEV round trip is needed in between.
- **Duration:** ≈ 7 h unattended (rough; preflight measures throughput).
- **Return:** `bundles/release1_<stamp>.tar.gz` + `.sha256` → DEV `incoming/RT1/`.

## Not needed from RUN (resolved on DEV)

| handoff item | status | how |
|---|---|---|
| pilot output files behind Tables B.1–B.6 | `resolved` | committed in `results/paper_20261026_001341/` (REPO_MAP §4). R1 checks that RUN has nothing extra |
| SAE inventory | `resolved` (Phase 1 script) | `google/gemma-scope-2b-pt-res` is reachable from DEV (not gated) |
| 16k SAE `params.npz` for T1 tests | `resolved` | DEV can download it directly (0.28 GiB each, L12 and L18) |
| gemma-2-2b tokenizer on DEV (T2 tests) | `resolved` 2026-10-30 | Gemma licence accepted for `legend479`; config + tokenizer load on DEV (revision `c5ebcd40`) |

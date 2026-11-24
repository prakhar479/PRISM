# Pilot, second set (branch `pilot-g3`): dictionary families on Gemma 3, and refusal (plan + runbook, 2026-11-24)

Status: **built and tested; not run yet.** It runs next to the breadth pilot (branch `pilot-breadth`, P0–P11) on a second
account with its own 4 × RTX 2080 Ti for 96 h. Everything here is exploratory. Its job is to widen the choice of paper direction
on Dec 14 (decision matrix in docs/PILOT_BREADTH_PLAN.md), not to supply paper numbers.

| id | question | feeds paper |
|---|---|---|
| **P12** | On one model and one site, does the *kind* of dictionary change how much task state it keeps? Covers residual SAEs, transcoders, CLTs, MLP/attention SAEs and crosscoders. And is the missing state in the replacement model's error nodes? | A (taxonomy), B (benchmark); the P3 the breadth pilot never built |
| **P13** | In an instruction-tuned model, is the refusal decision carried by SAE features or by the SAE error? | new safety section (the dropped P10 idea), any paper |

## P12. Dictionary families on Gemma 3 1B (Gemma Scope 2)

**Why Gemma 3 1B.** Gemma Scope 2 ships every family for this model at pinned revisions:
- residual SAEs: 4 widths × 3 L0s, plus a second seed;
- transcoders, all layers or single layer, with and without an affine skip;
- MLP-out and attention-out SAEs at every layer;
- CLTs (262k / 524k);
- weakly causal crosscoders on layers 7/13/17/22.

So families can be compared without mixing models or training pipelines.

**One site for all families (`harness/composite.py`).** The breadth pilot patches the residual after layer L, and so does P12.
The residual is an exact sum of the embedding and every sublayer's output up to L. A sublayer dictionary therefore defines a
decomposition of that same residual from the same run:
- **MLP replacement** (attention kept exact, the circuit-tracing convention): eps = Σ over l in S of (mlp_l − mlp_hat_l).
  - S = layers 0..L for `transcoder_all`, `mlp_out_all` and CLTs.
  - S = {L} for the single-layer `transcoder/` files.
- **Plus attention SAEs:** eps += Σ over l ≤ L of (ln1_post(z_l W_O + b_O) − ln1_post(ẑ_l W_O + b_O)).
- **Crosscoder:** block i reads the residual at L_i and writes only to layers ≥ L_i. So x_hat at L13 uses layers 7 and 13
  only, and eps = x_L − x_hat_L.

x_hat := x − eps. Every existing condition then applies unchanged: feature-only, error-only, capture along a DAS subspace,
random nulls.

**Cells.** Running sum and people-cities (P8's Gemma 3 prompt sets) at L13, with a subset at L17: 86 cells.
- Residual SAEs:
  - L13: 16k/65k/262k/1m × small/medium/big, plus 262k-medium seed 2;
  - L17: 4 widths at medium.
- Composites at L13:
  - transcoders, all layers: 16k small/big, each plain and affine; 262k small;
  - single-layer transcoders: 16k/65k/262k medium, plus 16k affine;
  - MLP-out SAEs: 16k small/big;
  - CLTs: 262k medium, 262k medium affine, 524k medium;
  - with attention SAEs: transcoder 16k + attention and CLT 262k + attention;
  - crosscoders: 65k/262k/524k/1m medium, plus 65k/262k big.
- Composites at L17: transcoder 16k all, CLT 262k, crosscoders 65k/262k.

**Conditions.** `feat_A_errB`, `feat_A_meanPos`, `feat_B_errA`, plus `feat_A_along_U` and `full_along_U` for one DAS v2 rank-8
subspace per task and layer. That subspace scores every dictionary, with 5 random-subspace seeds.

**Go if** (scripts/make_pilot.py `p12`):
- (a) **Family matters:** at L13, capture differs by ≥ 0.10 between dictionaries of different families, with separated
  bootstrap CIs. The 262k two-seed difference is reported as the noise floor.
- (b) **Error nodes hold task state:** for a replacement model (transcoders, CLT or MLP SAEs, with or without attention), the
  feature-only gap and the error-only NLDR are both ≥ 0.10.

**Verified on DEV (CPU).**
- TransformerLens hooks equal the HF modules the dictionaries were trained on:
  - `ln2.hook_normalized·w` = pre_feedforward_layernorm;
  - `hook_mlp_out` = post_feedforward_layernorm;
  - `attn.hook_z` = o_proj input;
  - `ln1_post(z W_O + b_O)` = attention output.
- Each file format matches its reading:
  - transcoder FVU 0.21 (affine 0.17 with `+ x @ A`);
  - CLT layers 0–2 FVU 0.01–0.08;
  - crosscoder FVU 0.09–0.37 as "block i reads layer i". The decoders to earlier layers are exactly 0.
- In the harness, on real running-sum spans at L13, median relative error of x_hat:

  | Dictionary | Median relative error |
  |---|---|
  | Residual SAE 16k | 0.07 |
  | Crosscoder 65k | 0.12 |
  | Single transcoder | 0.04 |
  | MLP-SAE replacement | 0.07 |
  | Transcoder + attention replacement (28 sublayers) | 0.15 |

- On the tiny model, `tests/harness/test_composite.py` checks every part kind's eps against a direct reference, and checks the
  exact residual decomposition.

**Not in P12.**
- CE / FVU on generic text per composite (paper B's fidelity axis). Gemma Scope 2 configs record the target L0.
- Object tracking: the set is large, and P8 will tell whether Gemma 3 does it.

## P13. Refusal state in Gemma 3 1B-IT

**Data (`scripts/build_refusal_set.py`).**
- Gemma 3 chat prompts `<start_of_turn>user\n{request}<end_of_turn>\n<start_of_turn>model\n`, with 8 request templates.
- Harmful (A) and harmless (B) acts are paired at equal token length.
- Train and test use disjoint act lists.
- The patched span runs from the first differing token through the turn tokens.
- Each prompt's answer is the IT model's own first token, typically "I" for a refusal and "Okay" / "Here" for compliance.
  Pairs whose two answers coincide are dropped.

**Cells.**
- Gemma Scope 2 1B-IT residual SAEs at L7/13/17/22, 16k and 262k medium.
- At L13: transcoder 16k all, CLT 262k, crosscoder 65k.

**Directions.** `diffmean` (new; the difference-of-means refusal direction, Arditi et al. 2024), DAS v2 at ranks 1/4/8 and PCA
of the error, with 10 random seeds. Conditions: the causal split, capture along U, and repair.

**Go if.** There must be ≥ 30% usable pairs. Then either:
- **Refusal partly dark:** the feature-only gap or the error-only NLDR is ≥ 0.10 at some layer, or diffmean capture is < 0.9
  (CI) with full NLDR above random p95.
- **Negative result:** diffmean capture is ≥ 0.9 (CI): features carry refusal.

Either outcome is reportable.

**Set as committed (2026-11-25).**
- Built from the IT model's own first tokens: 1,350 pairs (900 train / 450 test).
- Harmful requests: 100% answer "I".
- Harmless requests: "Okay" 90%, "##" 9%, a few others.
- Of the candidates scanned, 63% of train and 88% of test had differing answers and were kept.

The Slurm check refuses a set built with `--answers fixed` (a placeholder). To rebuild (needs the google/gemma-3-1b-it licence on
the token's account):
```bash
python scripts/build_refusal_set.py                  # CPU is enough (1B model, ~4k prompts); prints the first-token counts
python scripts/make_pilot_configs.py --check         # configs reference the set by name: unchanged
git add data/prompt_sets/refusal_p13_g3it.* && git commit -S -m "P13 refusal set (model answers)"
```
P12 can also run alone: `sbatch --export=ALL,ONLY=P12 scripts/slurm_pilot_g3.sbatch`.

## Run (second account; same cluster layout as the breadth pilot)

1. **On DEV:** push the branch and a signed tag.
   ```bash
   git tag -s run-<date>-g3 -m "pilot g3"
   git push origin pilot-g3 run-<date>-g3
   ```
2. **On the cluster, new account:**
   - SSH key on GitHub with write access (for `--record`).
   - `~/.hf_token` from an account that accepted **both** google/gemma-3-1b-pt and google/gemma-3-1b-it.
     The gemma-2-2b licence is needed for the preflight golden check.
   ```bash
   git clone git@github.com:prakhar479/PRISM.git prism-g3 && cd prism-g3 && git checkout run-<date>-g3
   git archive origin/release golden_parts | tar x && echo 'golden_parts/' >> .git/info/exclude
   git status --short                                   # empty
   bash scripts/slurm_pilot_g3.sbatch --check
   sbatch --time=04:00:00 scripts/slurm_pilot_g3.sbatch --smoke      # P13 + P12 split over 4 GPUs, throwaway caches
   grep -E "SMOKE (OK|FAIL)|smoke (finished|passed)" slurm-prism-g3-<jobid>.out
   sbatch scripts/slurm_pilot_g3.sbatch                             # resubmit the same command after a time limit
   ```
3. **After:** run `bash scripts/slurm_pilot_g3.sbatch --record`, then `git push -u origin HEAD`. That creates the
   `results/pilot-g3-<tag>` branch. On DEV, ingest it and run `python scripts/make_pilot.py --runs <...>/runs`, which adds P12
   and P13 to VERDICTS.md.

**Disk.**
- ~200 GB of Gemma Scope 2 downloads. Composites read every layer's file; CLTs are 1.2–2.4 GB per layer. Keep HF_HOME on the
  bulk disk (setup_server.sh does this).
- ~30 GB of stage-1 sublayer spans: ~50 (layer, site) entries per task.
- MIN_FREE_GIB is 300.

**Cost estimate.** ~35–60 GPU-h. Stage 2 streams one weight file at a time on the GPU, each read once per task. DAS v2 is fitted
once per task and layer (as in P8).

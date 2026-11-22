# Breadth pilot: which follow-up to PRISM can carry an ICML 2027 paper? (plan, 2026-11-22)

Status: **plan only, nothing built or run.** Everything this pilot produces is exploratory. Its job is to choose a direction,
not to supply paper numbers. Paper numbers come from the depth runs that follow, with their own frozen decision rules.

Context: ICML 2027 abstracts are due Jan 14 2027 and full papers Jan 22 2027 (AoE). The 2026-11-21 literature review is
summarised in the session notes. Short version: "SAEs lose to DAS" and "SAE causal scores depend on how you measure them" are
both taken. Three things are open:
- *accounting*: where a known variable's causal effect goes (features, error, dense part);
- *which kinds* of variables dictionaries lose;
- whether newer dictionaries or fixes change that.

Compute: 4 x RTX 2080 Ti for stretches of 96 h or more, so 384 GPU-h per window. For scale, the last exploratory run (E17-E22)
was about 62 GPU-h, or 16-18 h on 4 GPUs.

## Principles
1. **Each pilot has one question and one decision rule, written before it runs** (the "go if" lines below).
2. **Smallest design that can decide.** n_train 300 and n_test 150 pairs per cell, as in E18/E20, with 1000 bootstraps. Seeds
   only where seed variance is the question.
3. **The controls that make a "go" believable are included from the start.** Without them a "go" is worthless. Reviewers will
   now cite:
   - rank- and norm-matched random subspaces ("Calibrating Causal-Localization Scores", ICLR 2027 submission);
   - a stated write-back operator;
   - held-out DAS validity;
   - a DAS objective that cannot exceed the full-state effect ("Whose Effect Is It", 2026);
   - a frozen-random-decoder baseline (Korznikov et al., NeurIPS 2026).
4. **Budget: about 230 GPU-h of window 1.** The remaining ~40% covers reruns and failures. Window 2 is kept for depth runs on
   whichever pilots pass.

## Schedule (estimates)

| dates | what |
|---|---|
| Nov 24 - Dec 4 | Engineering E-1 to E-5 (below). This overlaps the course report, which is due Dec 1-7. |
| ~Dec 5 - 9 | **Window 1**: P0 first, everything else once P0's DAS v2 rotations exist. |
| Dec 9 - 13 | Analysis. Each pilot gets a one-page verdict: go, no-go, or unclear. |
| **~Dec 14** | **Pick the paper** (decision matrix at the end). Freeze its depth-run rules. |
| Dec 15 - 19 | **Window 2**: depth runs, a second model, seeds. |
| Dec 22 - 26 | **Window 3** (if needed): replications, extra dictionaries. |
| Jan 5 - 22 | Write. Abstract Jan 14, paper Jan 22. |

## Engineering prerequisites (DEV, no GPU needed; tests on the tiny model)

| id | work | where | needed by | est. dev-days |
|---|---|---|---|---|
| E-1 | **DAS v2.** Mini-batch training. Train/validation split of the train pairs, with validity judged on held-out pairs. Initialisation seed separate from data seed, so seeds do not share a start. Objectives: `ce_A` (current), `logit_diff`, `kl_full`: KL to the output distribution of the `full_A` patch, which cannot exceed the full-state effect. | `harness/model/das.py` | P0, P1, P2, P5, P11 | 1.5 |
| E-2 | **Dictionary adapter.** Wrap any SAELens residual SAE (JumpReLU, TopK, BatchTopK, Matryoshka, ReLU, Gated, P-anneal) behind the existing `decompose()` / `encode_sparse()` interface. Add a frozen-random-decoder control (optional: train one 16k SAE with a frozen random decoder at L12). | `harness/sae/` | P2, P4, P9 | 1.5 |
| E-3 | **New task families.** People-cities with three counterfactual types (content / binding-swap / query; see P1). Object tracking without a default answer, per Tang et al. (ICML 2026): both A and B contain a move of the queried object after `first_diff`. Run validator and keep-rate checks for both. | `harness/families.py`, `scripts/validate_templates.py` | P1, P11 | 1.5 |
| E-4 | **New directions and conditions:** magnitude and Fourier directions (P5); `eps_lin` / `eps_nl` split conditions (P6); empirical-Fisher and task-gradient error directions (P7); SAE KL fine-tune script (P4). | `harness/directions/`, `harness/conditions/`, `scripts/` | P4-P7 | 2 |
| E-5 | **Second model.** Gemma 3 1B is the default; 4B is optional. Check whether TransformerLens 2.16-3.x supports Gemma 3; otherwise port `resid_at` / `last_logits_from` to HF forward hooks. Add a Gemma Scope 2 loader (SAELens registry). | `harness/model/`, `harness/sae/` | P8 | 2-4 (riskiest) |
| E-6 | *(optional)* **Transcoder / CLT site-matched splice.** Replace the MLP outputs of layers <= L with transcoder reconstructions (errors from the corrupt run) and compare with the residual SAE at L. Sources: Gemma Scope 2b transcoders; circuit-tracer CLTs (426k, 2.5M). | new `harness/replace.py` | P3 | 2-3 |

Hardware notes:
- Turing has no native bf16: it runs emulated and slowly, as the harness already records.
- gemma-2-2b in fp32 with the embedding offloaded fits 11 GB, as PRISM already does.
- Gemma 3 1B in fp32 is about 4 GB and fits easily.
- Gemma 3 4B in fp32 (~17 GB) does not fit. It needs bf16-emulated (~8.6 GB, slow) or a split across two GPUs.
- Llama-3.2-1B is the fallback second model. It has an EleutherAI SAE, a skip-transcoder and a 524k CLT.

---

## P0. Protocol hardening (blocks everything; the "required regardless" items)

**Question.** Do PRISM's core numbers survive the new measurement critiques? The numbers in question are the running-sum (RS)
L18 feature capture of 0.75 and the people-cities (PC) L12 capture of 0.69.

**Design.**
- Cells: RS L18 and PC L12, Gemma Scope 16k.
- DAS v2 at ranks 4, 8 and 16:
  - 3 objectives (`ce_A`, `logit_diff`, `kl_full`);
  - 3 independent-init seeds;
  - held-out validity.
- Evaluate `feat_A_along_U` and `full_along_U` under two write-back operators:
  - residual-subspace replace (current);
  - code-level swap: patch only the SAE codes of latents whose decoder rows have the largest projection on U.
- Nulls: 20 rank-matched random U, plus a norm-matched random null (|P_U(x_A - x_B)| matched).
- Also run the implemented-but-never-run noise conditions: `feat_A_noise`, `feat_A_covnoise`.

**Cost.** About 30 GPU-h, mostly 2 cells x 3 ranks x 3 objectives x 3 seeds = 54 DAS fits.

**Go if.**
- Capture moves by <= 0.10 across objectives, seeds and write-back operators.
- Held-out validity is >= 0.5 at the rank used.

**Otherwise.**
- Adopt `kl_full` plus held-out validity everywhere.
- Report the sensitivity itself as a finding. On its own it is not a paper: the space is crowded.

**Feeds.** Every pilot uses P0's rotations and protocol.

## P1. Binding blind spot: do dictionaries lose relational state more than content? (idea 2)

**Question.** At matched rank, is the *pairing* (which person goes with which city) captured by SAE features less than the
*content* (which city)?

**Why.**
- The only "SAEs miss bindings" result is in a toy 2-layer model (Hönig et al., ICML 2026 MI workshop).
- No one has measured how much of a known binding mechanism Gemma Scope captures.
- PRISM's numbers so far:
  - PC binding is rank 16, and error directions capture none of it.
  - PC capture is 0.66-0.69, against 0.75 for running sum. That is not yet a big gap, so this is a real test.

**Design.** Three counterfactual families on the same people-cities prompts (E-3). Clean, corrupt and answer are equal in
length and single-token.
- **content:** corrupt changes A's city token ("Paris" to "Tokyo"). The variable is the attribute's identity.
- **binding-swap:** corrupt keeps every token but swaps which city goes with which person ("Alice ... Rome. Bob ... Paris").
  The variable is pure pairing, with the same set of entities and attributes.
- **query:** corrupt changes which person is asked about. The variable is the pointer to the entity being retrieved.

Plan:
- Layers 8, 12 and 16. Gemma Scope 16k and 65k at matched L0.
- DAS v2 (`kl_full`) at ranks 4, 8 and 16 per family.
- Measure `feat_A_along_U` / `full_along_U` capture, error-U repair to r = 64, and the code-level write-back.
- Optional: project onto Gur-Arieh et al.'s positional, lexical and reflexive route subspaces (ICLR 2026; their code supports
  Gemma) to label which route each family exercises.

**Cost.** About 40 GPU-h.

**Go if.** Capture(content) - capture(binding-swap) >= 0.15, with non-overlapping CIs, in >= 2 of 3 layers, and the gap does
not shrink at 65k.

**No-go.** If relational and content capture are equal within 0.05, the taxonomy has no relational leg. That is still a useful
negative result, but it is not a headline.

## P2. Dictionary-family sweep at a fixed site: does the dictionary matter, and do fidelity metrics predict causal faithfulness? (idea 1)

**Question.**
- At matched L0 and width, do SAE architectures differ in how much task state they keep?
- Is that predicted by the usual fidelity metrics (FVU, CE-recovered, SAEBench scores)?

**Why.**
- MIB tested only SAEs against DAS and neurons.
- The Gemma Scope 2 authors list "how much they miss and how much this matters" as open.
- gemma-2-2b L12 resid has the richest public set:
  - Gemma Scope JumpReLU at 16k to 1M;
  - SAEBench's 7 architectures (ReLU, TopK, BatchTopK, JumpReLU, Gated, P-anneal, Matryoshka-BatchTopK) at 4k, 16k and 65k, with
    about 6 L0s each;
  - chanind Matryoshka-BatchTopK 32k;
  - a Temporal SAE [verify availability].

**Design.**
- Cells:
  - PC L12 (content family from P1);
  - OT-v2 L12 (from P11), or the pilot OT if P11 slips;
  - RS at L19, where SAEBench has TopK and ReLU, plus Gemma Scope 16k, 65k and 1M.
- Dictionaries: about 40 SAEs per L12 cell, chosen as 7 architectures x 3 widths x 2 L0s (closest to 40 and 100).
- DAS U is fixed per cell, taken from P0/P1. It is dictionary-independent, so every dictionary is scored against the same
  causal subspace.
- Lean conditions:
  - `feat_A_errB`, `feat_A_meanPos`;
  - `feat_A_along_U` at DAS-4/8;
  - 5 random U at rank 8;
  - code-level write-back.
- Fidelity per dictionary: generic-text splice CE (E12 machinery, English only, about 200k tokens).

**Cost.** About 45 GPU-h. Downloads about 15 GB.

**Go (benchmark/taxonomy paper viable) if either holds:**
- (a) at matched L0 (+/-20%) and width, capture spreads by >= 0.10 across architectures, with non-overlapping CIs;
- (b) across >= 30 dictionaries, |Spearman(CE-recovered, capture)| <= 0.3. Fidelity would then not predict causal faithfulness.

**If neither holds:** "it's the sparsity budget, not the architecture". That becomes a section of whichever paper wins, not a
paper of its own.

## P3. Transcoders and CLTs vs residual SAEs on the same variables (optional; needs E-6)

**Question.** Does a replacement model built from transcoders or CLTs keep more or less task state than a residual SAE at the
matching depth? And is the missing state in their error nodes?

**Design.**
- Cells: RS L18, PC L12.
- Replacements:
  - Gemma Scope 2b transcoders, all MLPs <= L replaced;
  - circuit-tracer CLT 426k.
- Conditions: features-only vs error-only at the MLP-out sites. Also capture of the fixed residual DAS U, measured at the
  residual stream after layer L.

**Cost.** About 25 GPU-h.

**Go if.** Task-state retention differs from the residual SAE by >= 0.10, or CLT error nodes carry >= 50% of the missing
state.

**Note.** Lange et al. (2026) report that CLTs learn "shortcut" circuits. If the CLT keeps task state *better* than the
residual SAE, check whether it does so through those shortcuts.

## P4. Does fixing logits fix state? (idea 4; cheapest sharp result)

**Question.** KL fine-tuning, LoRA and dense add-ons are each reported to close 20-55% of the CE gap from SAE splicing. Do they
close the *task-state* gap by a similar fraction?

**Design.**
- Cells: RS L18, PC L12, OT-v2 L12.
- For each Gemma Scope 16k SAE, compare:
  - (a) the original;
  - (b) a KL+MSE fine-tune (Karvonen 2025 recipe; about 20-50M generic tokens, about 3-4 GPU-h each);
  - (c) a linear add-on x_hat + W x, fitted on generic text. This is Engels et al.'s linearly predictable error, shared with
    P6.
  - (d) a rank-24 dense complement: top-24 PCs of the residual added back exactly, a cheap stand-in for Deng et al.'s scaffold.
- For each, measure:
  - CE gap closed on generic text;
  - NLDR and flip gap closed on task pairs;
  - capture of the fixed DAS U.
- Optional (e): Chen et al.'s LoRA, if a gemma-2-2b checkpoint is public [verify]. Otherwise skip it in the pilot.

**Cost.** About 20 GPU-h.

**Go if.** The decoupling ratio (task-gap fraction closed / CE-gap fraction closed) is <= 0.5 or >= 2 in >= 2 cells. That is a
clean "Negative Results" or decoupling finding.

**If the ratio is about 1:** these fixes do fix state. Report that positively; it is a useful section for paper B or C.

## P5. Does magnitude go dark? The dense magnitude axis vs the periodic digit code in the running sum (idea 3)

**Question.** Is the part of the running-sum state that the SAE misses the smooth *magnitude* axis, while the periodic, per-digit
structure lives in features?

**Why. PRISM evidence:**
- The error carries graded magnitude: far donors give +0.16 to +0.21 NLDR, ±1 donors about 0.
- Steering is a one-sided push along the number line.
- 7 SAE atoms recur in DAS.

**Literature:**
- numbers sit on a helix (Kantamneni & Tegmark);
- per-digit circles (Levy & Geva);
- periods 2, 5 and 10 (Feucht et al., 2026);
- SAEs fragment number manifolds ("dilution": Bhalla et al., 2026; Gurnee et al., 2025).

**Design.** RS at L16, L18 and L19, last position plus the patched span.
- Fit on train pairs:
  - a ridge magnitude direction (value on x);
  - Fourier planes for periods 2, 5 and 10.
- Split DAS-4/8 into the magnitude line, the Fourier planes and the remainder.
- For each component U, measure:
  - features' share of the pair difference: |P_U(x_hat_A - x_hat_B)| vs |P_U(eps_A - eps_B)|;
  - causal effect: `full_along_U`, `feat_A_along_U`;
  - which component reproduces the far-donor resampled-error effect (patch only P_U eps_C).
- Width trend: Gemma Scope L19 at 16k, 65k and 1M.
- Check whether any always-on (dense) latent aligns with the magnitude direction.

**Cost.** About 12 GPU-h. Mostly cached activations plus a few patch sweeps.

**Go if.**
- Magnitude capture <= periodic capture - 0.20.
- Patching P_mag eps alone reproduces >= 50% of the far-donor effect.

**Theory hook if it goes.** An L0 / L1 / JumpReLU penalty prices an always-on direction at the full sparsity cost on every
token. So a dense continuous axis is cheaper to leave in the error. Test how this scales with width.

## P6. Anatomy of the task-relevant error: linearly predictable, nonlinear, or the dense scaffold? (links ideas 1, 3, 5)

**Question.** The SAE error splits into a part predictable linearly from x (Engels et al.) and a nonlinear remainder. When the
error carries task state, which part carries it? And is that part inside the dense low-rank scaffold (top PCs and massive
activations; Deng et al., Delisle & Tan)?

**Design.**
- Fit W by ridge on about 2-5M generic tokens per layer (L12, L18), so that eps_lin = W x and eps_nl = eps - eps_lin.
- Cells: RS L18, PC L12, OT-v2 L12.
- New conditions:
  - x_hat_A + eps_lin_A + eps_nl_B;
  - x_hat_A + eps_lin_B + eps_nl_A;
  - the same two with eps projected onto the top-24 PCs (and PC1 alone) vs its complement.

**Cost.** About 12 GPU-h.

**Go if.** The share in eps_lin is >= 0.6, so the loss is fixable by a linear add-on (a dictionary misallocation), or <= 0.3,
so it is genuine nonlinear "dark matter". Either way it is informative. A mixed 0.3-0.6 is a weak result.

## P7. Sensitivity, not variance: does a Fisher-weighted error explain cost and repair better than FVU and error PCA? (idea 5)

**Question.** Per token, splice cost ≈ 1/2 eps^T F eps, where F is the Fisher information of the next-token distribution taken
with respect to the residual at L. Does the *alignment factor* eps^T F eps / |eps|^2 explain the following?
- cost varying across layers, tasks and (optionally) scripts at equal FVU;
- why error-PCA directions miss the causal variable until r >= 32.

**Design.**
- Estimate an empirical Fisher at L12 and L18 from about 20k sampled-token gradients on generic text. Estimate a task-gradient
  matrix from ∇_x (logit_A - logit_B) on train pairs.
- (a) Predict per-window splice KL using |eps|^2 alone vs |eps|^2 x alignment (E12 windows; English, plus the native/romanized
  sets if multilingual stays).
- (b) New error directions:
  - top generalized eigenvectors of (Σ_eps, F);
  - the same for the task-gradient matrix.

  Use them as U in `repair` / `feat_A_along_U` at r = 1-16, against `pca_pooled`, `answer_sep` and DAS.
- Do a prior-art check *before* building: "Fisher-weighted SAE", Jacobian SAEs, e2e SAE.

**Cost.** About 10 GPU-h.

**Go if.**
- At r <= 8, Fisher/task-gradient directions close >= 2x the task gap of variance PCA, and beat random p95.
- The alignment factor explains >= 50% of the variance in splice cost that FVU leaves unexplained.

## P8. Second-model gate: Gemma 3 1B (and 4B if feasible) with Gemma Scope 2 (needs E-5)

**Question.**
- Can a second model do the tasks?
- Does the basic split (features carry most of the state; a gap remains in some cells) replicate?

This decides whether a multi-model paper is feasible. Any ICML version needs at least 2 models.

**Design.**
- Capability gate (`scripts/analyze_tasks.py` logic) for RS, PC (3 families), OT-v2 and IOI.
- For tasks that pass:
  - E8-style split at the Gemma Scope 2 layers (25/50/65/85% depth);
  - DAS v2 capture in the best cell.
- Gemma Scope 2 also ships transcoders, crosscoders and CLTs for 1B. That makes Gemma 3 1B the natural site for the full P2/P3
  family comparison in window 2.

**Cost.** About 25 GPU-h.

**Go if.**
- >= 2 tasks keep >= 30% usable pairs.
- The direction of the feature/error split replicates: gap > 0 with CI excluding 0 in >= 1 cell.

**Fallback.** Llama-3.2-1B.

## P9. Positional and temporal state (fold into P2 if E-2 handles the Temporal SAE)

**Question.**
- Is task state carried at earlier tokens better or worse captured than at the final token?
- Does a Temporal SAE capture accumulated state (the running sum) better than per-token SAEs?

**Design.**
- Per-span split: `from_diff` span vs last token vs the operand tokens only.
- Add the Temporal SAE (if available at gemma-2-2b L12) as one more P2 dictionary.

**Cost.** About 5 GPU-h on top of P2.

**Go if.** Temporal SAE capture is >= 0.10 above per-token SAEs at matched L0.

## P10. Multilingual (optional; drop if it gets too hard)

**Cheapest informative step: run E23.** It is implemented in tag run-20261118-2 and has never run. It answers the open red-team
question: is the native-script excess cost the error's *direction*, or are those contexts fragile to any perturbation?

**Cost.** About 11 GPU-h.

**Go (keep a multilingual section) if.** The native/English excess persists against norm-matched noise (x_hat-splice cost /
x+eta cost ratio still > 1.3 at L12) **and** P7's alignment factor explains it.

**Otherwise.** Drop multilingual from the ICML paper.

The safety-tool decomposition (dense vs SAE vs error probes for refusal in low-resource languages) needs IT models with SAEs.
That means Gemma 3 1B/4B-IT with Gemma Scope 2, so it waits for P8. It is not a window-1 item.

## P11. Object tracking without a default answer (loose end; makes OT usable in any paper)

**Design.**
- Build OT-v2 (E-3).
- Run the gate, the E8 split at L8-16, DAS v2 at L12, and the reverse sufficiency patch x_A + P_U(x_B - x_A): does it *write*
  B's box into A's run?

**Cost.** About 10 GPU-h.

**Go if.**
- >= 30% usable pairs.
- The reverse patch writes B in >= 3x random p95.

**Otherwise.** OT stays out of every claim.

---

## Window-1 queue (about 230 GPU-h; GPUs are a guide, the queue scheduler balances)

| GPU | order | GPU-h |
|---|---|---|
| 0 | P0 (DAS v2 rotations; everything else waits for RS/PC rotations), then P1 DAS fits | 30 + 40 |
| 1 | P11 then P2 (L12 sweep starts as soon as P0/P1 U exist; RS L19 part after P0) | 10 + 45 |
| 2 | P6 ridge fits, P4 KL fine-tunes, P4 evals, P7 | 12 + 20 + 10 |
| 3 | P8 (if E-5 is ready, else P10/E23 first), P5, P9, P3 if E-6 is ready | 25 + 12 + 5 + 25 |

## Decision matrix (Dec 14): pilot outcomes → paper

| paper | thesis | needs "go" in | supporting |
|---|---|---|---|
| **A. What dictionaries lose: a causal taxonomy** | content kept, magnitude partly, relations lost; architecture/width/fixes help only the first two | P1 + (P5 or P4) + P8 | P2, P6, P0 |
| **B. Fidelity is not faithfulness: a causal benchmark of dictionaries** | standard fidelity metrics do not predict task-state retention across ~40 dictionaries x 2 models | P2 + P8 | P3, P4, P9, P0 |
| **C. Magnitude goes dark** | sparsity priors push dense continuous axes into the error; scaling prediction + test | P5 + P6 | P4, P7 |
| **D. Sensitivity, not variance** | error cost = size x alignment with output sensitivity; Fisher-weighted repair beats PCA; explains script/width/layer effects | P7 + P6 | P4, P10 |
| **E. Logit fixes don't fix state** (Negative Results type) | CE-gap fixes leave task state unrepaired | P4 + P8 | P2 |

**If several go:** A is the umbrella. B, C and D slot in as its sections.

**If nothing goes:** the fallback is B with whatever P2 shows, plus the P0 protocol as the methods contribution, targeted at
NeurIPS 2027 rather than ICML.

## Key references per pilot (see the 2026-11-21 review for the full list)
- **P0:**
  - "Calibrating Causal-Localization Scores" (ICLR 2027 submission 3KUTq95sXr, withdrawn);
  - "Sparse-Dictionary Causal Claims Are Not Robust to How They Are Measured" (ICLR 2027 submission bbDKLX9z7A);
  - "Whose Effect Is It" (2026);
  - Korznikov et al., "Sanity Checks for SAEs" (arXiv 2602.14111).
- **P1:**
  - Hönig et al., "Additive Relational Bindings…" (ICML 2026 MI workshop);
  - Gur-Arieh et al., "Mixing Mechanisms" (ICLR 2026);
  - Feng & Steinhardt (ICLR 2024);
  - Dai et al. (EMNLP 2024);
  - Prakash et al., lookbacks (ICLR 2026).
- **P2/P3:**
  - MIB (ICML 2025);
  - SAEBench (ICML 2025);
  - Gemma Scope 2 (Dec 2025);
  - Paulo et al., transcoders (arXiv 2501.18823);
  - Anthropic circuit tracing (2025);
  - Lange et al., CLT shortcuts (2026).
- **P4:**
  - Chen, Engels & Tegmark, LoRA for SAEs (ICML 2025);
  - KL+MSE fine-tune (arXiv 2503.17272);
  - Deng et al., "Decompose Sparsely…" (arXiv 2606.14040).
- **P5:**
  - Kantamneni & Tegmark (arXiv 2502.00873);
  - Levy & Geva (NAACL 2025);
  - Feucht et al. (arXiv 2605.01148);
  - Bhalla et al. (arXiv 2604.28119);
  - Gurnee et al., manifolds (arXiv 2601.04480);
  - Michaud et al. (arXiv 2509.02565).
- **P6:**
  - Engels et al., dark matter (TMLR 2025);
  - Delisle & Tan (arXiv 2611.13548);
  - Deng et al.
- **P7:**
  - Gurnee, pathological errors (2024);
  - Braun et al., e2e SAE (NeurIPS 2024);
  - prior-art search pending.
- **P11:** Tang et al., "Do LMs track entities across state changes?" (ICML 2026).

## Implementation notes (2026-11-22, branch `pilot-breadth`)

Built and runnable as one Slurm job: `scripts/slurm_pilot.sbatch` -> `queue/pilot-breadth-server.yaml`. Runbook and code map:
`docs/PILOT_RUNBOOK.md`. Changes from the plan above, each for a measured or verified reason:

- **P6** is folded into P4. The `lin` add-on is Engels' linearly predictable error, and `pcx24` / `pcx1` are the dense scaffold and
  the massive-activation direction. The P6 share is the task gap the `lin` variant closes.
- **P9** is folded into P2 as one more dictionary: the Bhalla et al. temporal SAE, gemma-2-2b L12. The "Priors in Time" model is
  attention-based and cannot be used per token.
- **P10** runs only the never-run E23 noise controls (English as P7(a), three multilingual sets optional).
- **P11** uses three 7-label E3 templates with the `dest` counterfactual: one prompt set needs one candidate count. The reverse
  sufficiency patch is dropped, because OT-v2 has no prefix fallback; the fallback box is recorded per pair instead.
- **P2** evaluates PC L12 and RS L12, not OT. Caching x_hat for ~50 dictionaries on the 8,100-pair OT set would need ~80 GB.
  Write-back operators are compared in P0 only.
- **Not built:** P3 (transcoders / CLTs), Gemma 3 4B and the LoRA fix (no public checkpoint).
- **Verdicts:** `scripts/make_pilot.py` evaluates the "go if" rules above from the finished runs.

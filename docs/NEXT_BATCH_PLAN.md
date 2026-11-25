# Next batch: where we stand, what could make a paper, and the two 36-hour branches (plan, 2026-11-26)

Status: **plan + build.** Findings behind every claim here are in docs/FINDINGS_CATALOGUE.md (ids S*, R*, O*). Everything below is
exploratory until a depth run freezes its decision rules.

## 1. Where we stand

**We have a trustworthy instrument and one real mechanism, and we have closed the two stories we hoped to tell.**

- *Instrument.* Causal accounting of a known variable: DAS gives the subspace U that carries it, and write-back conditions say how much
  of that effect the dictionary's features carry (*capture*) and how much sits in the error. It replicates across GPUs, models
  (Gemma 2 2B, Gemma 3 1B) and tasks, and it has a measured noise floor (S10: ~0.1 from DAS objective and seed).
- *What it says.* Features carry most of a task variable (S1: capture 0.55-0.96); the missing part is small in decisions, large in logits
  (S2), shrinks only slowly with width (S3), and is **not** found by error variance (R6) but **is** found by output sensitivity (S4).
  SAE error is less harmful than random noise of equal norm (S5, 0.56-0.71), so the error is anti-aligned with sensitive directions,
  and the part that is not anti-aligned is where the task state is.
- *Closed.* Binding (R1), magnitude (R2), multilingual (R3, S6) and running sum on Gemma 3 (R7). Papers A and C of the breadth plan are dead.
- *Unfinished and decisive.* P2 and P4 never completed (O4, O5); P7's strongest rows used a task-specific gradient (O1).

**The honest reading:** the strongest sentence we can currently defend is "the unexplained part of an SAE is not noise and not
high-variance; it is a low-dimensional, sensitivity-aligned subspace that carries part of a task variable", with the weakness that
the sensitivity was estimated with the task in hand.

## 2. Three papers this could become (and what each needs)

| paper | claim | needs | status |
|---|---|---|---|
| **D+ "Dark matter has a direction"** | the task-relevant SAE error lies in a *task-agnostic* sensitivity subspace (Fisher of the LM loss), so a few linearly predicted directions repair it, and variance-chosen directions do not | O1 (generic sensitivity beats variance, same budget), S4, S5, a CE cost, 2 models | **branch repair decides it** |
| **B+ "A causal atlas of dictionaries"** | where (depth), at what width / sparsity / architecture, and for which variables sparse dictionaries lose task state; which cheap metrics predict it | O3 (depth), O4 (metric predictors), S7, S10 protocol, 2 models | **branch atlas decides it** |
| **E "Error nodes of replacement models"** | transcoder / CLT / crosscoder error nodes carry task state; family effects shrink once error size is matched | S7, S8, P14, IOI, CE/FVU per composite | partly in hand (P12), P14 pending |

D+ is the best bet for impact: a method people can apply to *their* SAE, grounded in a mechanism we already measured, and it makes a
prediction that can fail. B+ is the safe, valuable data paper (ICML's evaluation track fits). They share a protocol and a harness, so one
branch each makes both decidable after a single window. E rides on the second account's P14 and a later window.

## 3. Hypotheses and the experiments that test them

| hypothesis | test | prediction if true | prediction if false |
|---|---|---|---|
| **H1** a generic-text Fisher subspace contains the task-relevant error | repair the task with a k-dim, *linearly predicted* error correction; subspace chosen by (a) variance (`var`), (b) Fisher alone (`fis`), (c) Fisher-weighted error (`sens`), (d) random (`rnd`); each also as an *oracle* projection of the true error (`o*`) | `sens` / `fis` close a large share of the gap at k <= 32; `var`, `rnd` do not (like S4, without task labels) | `sens` ~ `var` ~ `rnd`: the task-relevant error is task-specific; then the repair needs task data |
| **H2** the missing part is a decoder problem, not an information problem | refit the decoder on the frozen codes (`clin`, 16k) and train an MLP on x (`mlp`): do they repair the task? | `clin` repairs: information is in the codes | `clin` fails but `mlp` repairs: the codes drop it |
| **H3** capture dips where the variable is computed (mid-depth) | depth atlas, all layers, 3 tasks, 3 DAS seeds, two models | minimum near mid-depth in both models, aligned across tasks | monotone or flat; layer effects are noise |
| **H4** a calibrated fidelity metric predicts capture | CE / FVU / L0 / relative error / noise-normalised cost (splice / noise, S5) for every atlas dictionary vs capture | one metric explains most of it | no standard metric does |
| **H5** the effect is the same on a second model | Gemma 3 1B replicate of H1 and the atlas | same sign and size | model-specific |
| **H6** the sensitivity-chosen repair stacks with KL fine-tuning | `klft` dictionaries with and without the `sens` add-on | complementary | fine-tuning already removes it |

## 4. The two branches (each ~36 h on 4 x RTX 2080 Ti; run on separate accounts)

Both branch off `batch2-base`, which holds the shared harness upgrades below. IDs start with `P` so the bundle / ingest globs keep working.

### Branch `batch2-atlas` (queue `queue/pilot-x1-server.yaml`): H3, H4, H5
| id | content | cells |
|---|---|---|
| P20 | Gemma 3 1B depth atlas: people-cities, object tracking, IOI, all 26 layers, Gemma Scope 2 16k small / big | 156 |
| P21 | Gemma 2 2B depth atlas: running sum, people-cities, object tracking, 12 layers, 16k and 65k at matched L0 | 72 |
| P22 | Gemma 3 1B-IT refusal depth atlas: 26 layers, 16k small / big; difference of means vs DAS | 52 |
| PF_* | CE / FVU / L0 / noise-normalised cost on generic text for every dictionary above (the H4 predictors) | ~130 dictionaries |
Protocol: DAS v2 kl rank 8 with 3 independent seeds (rank 1 / 4 / 8 for refusal), held-out validity, 5 random-subspace nulls per cell.
Verdicts: `scripts/make_atlas.py` (capture vs depth with seed spread, per-task minimum, metric correlations).

### Branch `batch2-repair` (queue `queue/pilot-x2-server.yaml`): H1, H2, H5, H6
| id | content | cells |
|---|---|---|
| S_* | generic-text Fisher of the LM loss per layer (Gemma 2: 6/12/18/24; Gemma 3: 7/13/17/22) | 8 jobs |
| P23 | Gemma 2 repair curves at the canonical cells (RS L18, PC L12, OT L12, 16k): `var` / `fis` / `sens` / `rnd`, predicted and oracle, k = 2, 8, 32, 128 | ~100 |
| P24 | the same at 65k, k = 8, 32 | ~30 |
| P25 | layer sweep of `sens` vs `var` vs `rnd` at k = 32 (4 layers x 3 tasks) | ~48 |
| P26 | Gemma 3 1B replicate (people-cities, object tracking, IOI at L13 / L17) | ~54 |
| P27 | decoder-vs-information (`clin`, `mlp`) and KL fine-tune with / without the add-on | ~14 |
| PF_* | CE / FVU / L0 of every add-on dictionary (the CE cost of each repair) | all |
Verdicts: `scripts/make_repair.py` (gap and capture closure vs k, with seed spread, oracle vs predicted, CE cost).

### Shared harness upgrades (`batch2-base`)
1. Add-on subspaces `var<k>` / `fis<k>` / `sens<k>` / `rnd<k>` (linear predictor restricted to the subspace) and their oracle forms
   `ovar` / `ofis` / `osens` / `ornd`; `clin` (decoder refit on frozen codes); `mlp` (nonlinear predictor from x).
2. A generic-text Fisher stage (`scripts/fit_sensitivity.py`): sampled-token score vectors through the rest of the network, cached per
   (model, corpus, layer).
3. A Gemma 3 generic corpus; IOI prompt sets for Gemma 3; Gemma Scope 2 registry entries for every layer (`resid_post_all`, base and IT).
4. `make_atlas.py`, `make_repair.py`; the verifier already covers any queue.

## 5. Decision rules (written before the run)

- **H1 go** if, pooled over the three Gemma 2 canonical cells at k <= 32, `sens` or `fis` closes >= 2x the gap that `var` closes and
  exceeds the `rnd` p95, at <= 10 % of the CE delta (PF) the KL fine-tune adds. Otherwise D+ is dropped as a method and kept as a negative.
- **H3 go** if the capture minimum falls in the middle third of depth for >= 2 of 3 tasks in both models, with seed spread < the dip.
- **H4 go** if one cheap metric has |Spearman| >= 0.7 with capture across the atlas and beats FVU; **no-go** if FVU already does.
- Any difference below the protocol noise (0.1, S10) is reported as a tie.

## 6. Schedule (ICML 2027: abstract Jan 14, paper Jan 22)

| dates | what |
|---|---|
| Nov 26 - Dec 3 | build, tiny-model tests, smoke on the cluster (both branches); P2, P4, P14 finish on the existing accounts |
| Dec 4 - 9 | **batch 2**: atlas and repair in parallel, ~36 h each |
| Dec 10 - 13 | analysis; catalogue update; **pick the paper** (D+, B+, or both) |
| Dec 14 - 22 | depth window: frozen rules, more seeds, second family if D+ holds |
| Jan 5 - 22 | write |

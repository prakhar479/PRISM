# PRISM research team brief (2026-11-19) — read fully before starting

You are one member of a research team analysing the complete results of PRISM, an interpretability study. Work ONLY with data
already on this machine (no GPU, no new model runs, no downloads except what is already in place). You may write anything under
`$SP/team/` (your report + scripts). Do NOT modify anything under `$SP/hw` (a git worktree with tracked files) or `$SP/ana_runs`.

Paths: `SP=/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad`
Python: `/home/prakhar/dev/legend479/MILP/.venv/bin/python` (pandas, numpy, scipy, torch CPU, pyarrow; statsmodels/sklearn: check
before relying on them). Run scripts with `cd $SP/hw` so `import harness...` works (or `sys.path.insert(0, SP+'/hw')`).
Machine: 15 cores / 23 GB shared by ~8 analysts at once: set `OMP_NUM_THREADS=2`, keep bootstraps <= 2000, read only the parquet
columns you need.

## The project in brief
Model gemma-2-2b (26 layers, residual width 2304), Gemma Scope JumpReLU residual SAEs (16k canonical; wider widths in E4/E20/E21).
x = x̂ + ε (x̂ = SAE reconstruction = "features", ε = reconstruction error = "dark matter"). Question: is task state (a running sum,
which object is in which box, which city a person lives in) captured by features or does part of it live in ε?
Clean prompt A vs corrupt prompt B differing in one step. Patching at the positions from the first changed token to the end.
Tasks: add_chain = running sum (RS, layer 18 confirmatory), entity_track = object tracking (OT, layer 12 confirmatory),
people_cities (PC, binding), ioi_control (IOI names), var_binding (VB).
Metrics (per pair, test split): flip rate = share of eligible pairs whose top-1 becomes A's answer (eligible: clean B top-1 = B's
answer); NLDR = (D_patched − D_B)/(D_A − D_B) with D = logit(ans_A) − logit(ans_B); prob_recovery. "gap" = M(full_A) − M(cond).
Filters: f_none, f_A_correct, f_pilot (subsets of test pairs; f_pilot is the fit filter).
Conditions (harness/conditions/__init__.py): full_A, full_B, clean_B; feat_A_errB = x̂_A + ε_B; feat_B_errA = x̂_B + ε_A;
feat_A_meanPos = x̂_A + mean train error per position; feat_A_zero; noise controls (feat_B_noise etc.); resample (error from another
context with same/different answer); directions U (r columns) from estimators pca_pooled / answer_sep (fitted on train errors) or
DAS (distributed alignment search: trained rotation, `das<k>[_e<epochs>][_lr<lr>][_s<seed>]`, defaults 3 epochs, lr 1e-2, seed 0);
repair = x̂_A + ε_B + P_U(ε_A − ε_B); full_along_U = x_B + P_U(x_A − x_B); feat_A_along_U = x̂_B + P_U(x̂_A − x̂_B) + ε_B;
steer_x<s> = x_B + s·|m|·P_U m/|P_U m| (m = train mean of x_A − x_B per row key; matched-norm by construction); random U = 20 Haar
draws per rank (null). Haar p99 of top principal cosine in R^2304: k=1 .052, 2 .072, 4 .097, 8 .131, 16 .179, 32 .246, 64 .339.

## Data
- Runs (merged, all complete): `$SP/ana_runs/E*/<hash>/` with `manifest.json` (config, git, machine), `per_pair.parquet`
  (columns: pair_id, split, family, task, layer, sae, condition, estimator, source, r, seed, ans_A_id, ans_B_id, donor_status,
  fit_families, f_none/f_A_correct/f_both_correct/f_pilot, logit_A, logit_B, lse, top1_id, cand_logits = logits of all candidate
  answers), `directions/*.npz|json` (U matrices; json has DAS meta: train_interchange_acc, loss per epoch), `summary.*`.
  E12 runs (fidelity on ordinary text per language) have `fidelity.csv` and `per_window_<set>_L<layer>.npz` (per window: nll_clean,
  nll_splice, nll_mean, nll_zero, pred, fvu_num, fvu_den, eps_ratio_sum, l0_sum, pos; splice cost = (nll_splice−nll_clean)/pred).
- Prompts: `$SP/hw/data/prompt_sets/*.jsonl.gz` (pair_id → clean/corrupt prompt, answers, meta). Language sets: `$SP/hw/data/lang/`.
- Standard tables already computed (harness/final.py + scripts/make_final.py; read their code for definitions):
  * previous: `$SP/hw/tables/final/` (E9 illusion/DAS, E10 steering, E11 people-cities spec, E12–E14 multilingual, E16 RS layer
    sweep, E2r, E9b, E10b) and `$SP/hw/tables/*.csv` (E3 gate, E4 width, E5 subthreshold, E8 36-cell atlas, RQ1 gate)
  * NEW (this run): `$SP/r5/tables/` → E12_fidelity_lang.csv (now incl. 65k rows = E21), E17_illusion/_conditions/_das_stability,
    E18_illusion/_conditions, E19_steering/_conditions, E20_illusion/_conditions, E22_illusion/_conditions/_transfer
    (if a file is missing, make_final may still be running: wait/retry; log at `$SP/r5/make_final.log`)
- 16k Gemma Scope decoders for L12 and L18 (public) are in `$SP/deep/hf/` (HF cache layout; load params.npz, W_dec rows = atoms).
- Prior analyses (read for context, then go beyond them): `$SP/hw/analysis/final_pass_results.md` (answers under the frozen
  rules), `$SP/hw/analysis/deep_analysis_2026-11-18.md` (post hoc thesis: error SIZE does not predict error COST; script effect;
  gain model; OT prefix-fallback confound; DAS shared-initialisation artefact), `$SP/hw/analysis/deep/*.md` (+ scripts).
- Frozen rules: `$SP/hw/PREREGISTRATION.md`, `$SP/hw/PREREGISTRATION_ADDENDUM.md` (E9 rules: effective = flip ≥ random p95 + 10
  pts and lower CI > p95; illusion = full_along_U effective and full − feat ≥ 10 pts with paired CI > 0; feature-mediated =
  feat_A_along_U effective; DAS fit valid = train interchange acc ≥ 0.5; aligned ≥ 0.5 cosine; Q2 = steer flip ≥ 50 % and random
  p95 ≤ 5 %).

## The new experiments (all exploratory, post hoc; designed from the final-pass results)
E17 DAS robustness, RS L18: das1/2/4/8 plus seeds s1/s2, 10 and 30 epochs, lr 3e-2 (question: were E9's failed rank-1/2 DAS fits
  an optimisation failure? are rotations seed-stable?). E9's DAS rotations were retrained on this node with identical settings:
  a cross-machine reproducibility check.
E18 rank: error-U (pca_pooled, answer_sep) r up to 64, DAS rank 1–16, on RS L18, OT L12, PC L12, PC L16 (PC DAS is new).
E19 steering dose s = 0.25…16, RS L18, OT L12, PC L12, estimators pca/answer_sep/das4/das8/das8_e10.
E20 E9 contrasts with wider SAEs at matched L0: RS L18 65k (L0 64), OT L12 65k (72) and 262k (67).
E21 (rows in E12 table with sae L12_65k_l072 / L18_65k_l064): multilingual fidelity with a 4x wider SAE.
E22 multilingual DAS/PCA: fitted natively vs on English, RS L18 (fr, hinglish, hi) and PC L12 (fr, hi, hinglish).
NOT run: E23 (norm-matched noise controls); the server ran the older tag. Say so where it matters; don't speculate it in.
Provenance: this run used node gnode058 (not gnode084), commit 240e5f6 = tag run-20261118-1 + a server-side #SBATCH node edit.

## Ground rules
- Never fabricate or round away numbers; every number you report must come from a file you read or code you ran. Give n, CI or
  test, and the exact file/script. Report nulls and failures as findings.
- Separate FACT (what the data show), INTERPRETATION (what it may mean), and SPECULATION. Label everything post hoc/exploratory.
- Where several tests are run, say how many and correct (BH) or state uncorrected.
- Deliverable: `$SP/team/<your_role>.md` (findings ranked by strength, each: claim, numbers, test, n, caveats, alternative
  explanations) and scripts in `$SP/team/scripts/<your_role>_*.py`. Final message: a concise summary (≤ 40 lines) with the
  strongest findings, numbers and the report path.

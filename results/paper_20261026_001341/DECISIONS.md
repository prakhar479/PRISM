# Decisions for `paper_20261026_001341`

## Verdict: 🔵 **REFINE**

- Why: the strong H1 is rejected (the SAE reconstruction carries most of the causal state), but the SAE misses a small low-rank, task-relevant part: add_chain (L18: gap 0.13, rank 2 closes 58%); entity_track (L12: gap 0.26, rank 1 closes 52%). Caveat: the IOI control shows a residue too (L18: gap 0.24, rank 32), so this is partly a general SAE loss, not specific to state tracking
- Next: reframe as 'how much causal state do SAEs miss, where, and does it shrink with width': run width scaling on those layers (./run.sh start --tiers tier2_widths) with from_diff patching

| task | H1 | layer | error_only | sae_only | H2 probe (clean = beats x_hat and prompt tokens) | H3 (error subspace vs the SAE gap) |
|---|---|---|---|---|---|---|
| `add_chain` | rejected | L20 | 0.13 | 0.86 | decodable, not used here | L18: rank 2 closes 58% of a 0.13 gap |
| `entity_track` | rejected | L12 | 0.62 | 0.74 | decodable, not used here | L12: rank 1 closes 52% of a 0.26 gap |
| `var_binding` | rejected | L20 | 0.03 | 0.96 | decodable, not used here | - |

## 0. Did the run work?
- ℹ️ **INFO** `preflight: ok`
- ℹ️ **INFO** `paper_residue: exit 0 after 58 min`
- ℹ️ **INFO** `paper_widths: exit 0 after 124 min`
- ℹ️ **INFO** `paper_audit: exit 0 after 199 min`
- ✅ **OK** `add_chain`: n_shots=8 format=auto filter=strict retention=0.62
- ⚠️ **WARN** `entity_track`: n_shots=8 format=auto filter=clean_only retention=0.27
  - → *few pairs pass the filter: results for this task are noisier; see scripts/analyze_tasks.py for an easier difficulty (gen_kwargs) or more shots*
- ✅ **OK** `var_binding`: n_shots=2 format=auto filter=strict retention=0.40
- ✅ **OK** `ioi_control`: n_shots=2 format=auto filter=strict retention=1.00
- ✅ **OK** `paper_residue`: all 4 SAEs recover >70% of CE on generic text (min 0.98)
- ⚠️ **WARN** 1 failed work units in `paper_residue`
  - → *see paper_residue/metrics/failures.jsonl; missing SAE widths are expected, OOM/other errors -> rerun with --resume*
- ⚠️ **WARN** 1 failed work units in `paper_widths`
  - → *see paper_widths/metrics/failures.jsonl; missing SAE widths are expected, OOM/other errors -> rerun with --resume*

## H1. Does the SAE error carry the task state? (exp1)
- ✅ **OK** sanity: full clean patch (from_diff) recovers NLDR 1.00 (should be ~1)
(judged on `from_diff` patching, at layers where the full clean patch restores >= 0.8)
- ❌ **FAIL** H1 rejected: `add_chain` most error-favourable layer L20: error_only 0.13 vs sae_only 0.86 (sae_only clearly higher); noise_control 0.17 (clean error adds -0.04 over erasing the corrupt error)
  - → *the SAE reconstruction carries this state; report as a negative result*
- ❌ **FAIL** H1 rejected: `entity_track` most error-favourable layer L12: error_only 0.62 vs sae_only 0.74 (sae_only clearly higher); noise_control 0.51 (clean error adds +0.11 over erasing the corrupt error)
  - → *the SAE reconstruction carries this state; report as a negative result*
- ❌ **FAIL** H1 rejected: `var_binding` most error-favourable layer L20: error_only 0.03 vs sae_only 0.96 (sae_only clearly higher); noise_control -0.03 (clean error adds +0.07 over erasing the corrupt error)
  - → *the SAE reconstruction carries this state; report as a negative result*
- ✅ **OK** control `ioi_control` most error-favourable layer L4: error_only 0.00 vs sae_only 1.00 (sae_only clearly higher); noise_control 0.01 (clean error adds -0.00 over erasing the corrupt error)

## H2. Is the state a low-rank manifold in the error? (exp2)
- ⚠️ **WARN** `add_chain` L12: probe acc eps 0.53 vs x_hat 0.42, prompt tokens only 0.45 (R^2 eps 0.98 vs tokens 1.00), eps selectivity 0.33
  - → *the probes read the final token, but patching the final token at L12 restores only 0.03: the state is linearly present in eps there but not yet causally used (the model reads it from earlier positions). Interesting, but not evidence for H1*
- ⚠️ **WARN** `entity_track` L12: probe acc eps 0.84 vs x_hat 0.48, prompt tokens only 0.33 (R^2 eps 0.71 vs tokens 0.08), eps selectivity 0.65
  - → *the probes read the final token, but patching the final token at L12 restores only 0.47: the state is linearly present in eps there but not yet causally used (the model reads it from earlier positions). Interesting, but not evidence for H1*
- ⚠️ **WARN** `var_binding` L12: probe acc eps 0.95 vs x_hat 0.39, prompt tokens only 1.00 (R^2 eps 0.93 vs tokens 1.00), eps selectivity 0.84
  - → *the probes read the final token, but patching the final token at L12 restores only 0.00: the state is linearly present in eps there but not yet causally used (the model reads it from earlier positions). Interesting, but not evidence for H1*

## H3. Does a small error subspace repair the state? (exp3)
(score: share of the gap 1 - sae_only closed by a rank-r error subspace; random subspace as the null)
- ✅ **OK** `add_chain` L18: the SAE misses 0.13 of the effect; a rank-2 error subspace (supervised_centroid_svd) closes 58% of that gap (random rank-2: 1%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*
- ✅ **OK** `entity_track` L12: the SAE misses 0.26 of the effect; a rank-1 error subspace (unsupervised_svd) closes 52% of that gap (random rank-1: 0%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*
- ✅ **OK** `ioi_control` L18: the SAE misses 0.24 of the effect; a rank-32 error subspace (supervised_centroid_svd) closes 56% of that gap (random rank-32: 2%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*

## Width scaling (paper_widths)
- ℹ️ **INFO** `add_chain`: SAE gap (1 - sae_only, from_diff) by width: 16k 0.06, 32k 0.06, 65k 0.07, 131k 0.05, 262k 0.05
  - → *wider SAEs absorb it: a capacity limit, not a blind spot (the refined claim fails here)*
- ℹ️ **INFO** `entity_track`: SAE gap (1 - sae_only, from_diff) by width: 16k 0.26, 32k 0.20, 65k 0.24, 131k 0.23, 262k 0.15
  - → *wider SAEs absorb it: a capacity limit, not a blind spot (the refined claim fails here)*
- ℹ️ **INFO** `ioi_control`: SAE gap (1 - sae_only, from_diff) by width: 16k 0.04, 32k 0.03, 65k 0.02, 131k 0.02, 262k 0.00
  - → *wider SAEs absorb it: a capacity limit, not a blind spot (the refined claim fails here)*

# Decisions for `pilot_20261025_230749`

## Verdict: 🔵 **REFINE**

- Why: the strong H1 is rejected (the SAE reconstruction carries most of the causal state), but the SAE misses a small low-rank, task-relevant part: add_chain (L18: gap 0.12, rank 1 closes 53%); entity_track (L12: gap 0.25, rank 1 closes 51%). Caveat: the IOI control shows a residue too (L18: gap 0.24, rank 32), so this is partly a general SAE loss, not specific to state tracking
- Next: reframe as 'how much causal state do SAEs miss, where, and does it shrink with width': run width scaling on those layers (./run.sh start --tiers tier2_widths) with from_diff patching

| task | H1 | layer | error_only | sae_only | H2 probe (clean = beats x_hat and prompt tokens) | H3 (error subspace vs the SAE gap) |
|---|---|---|---|---|---|---|
| `add_chain` | rejected | L18 | 0.12 | 0.88 | no | L18: rank 1 closes 53% of a 0.12 gap |
| `entity_track` | rejected | L12 | 0.61 | 0.75 | decodable, not used here | L12: rank 1 closes 51% of a 0.25 gap |
| `var_binding` | rejected | L18 | 0.05 | 0.95 | decodable, not used here | L18: rank 8 closes 51% of a 0.05 gap |

## 0. Did the run work?
- ℹ️ **INFO** `preflight: ok`
- ℹ️ **INFO** `pilot: exit 0 after 24 min`
- ✅ **OK** `add_chain`: n_shots=8 format=auto filter=strict retention=0.62
- ⚠️ **WARN** `entity_track`: n_shots=8 format=auto filter=clean_only retention=0.27
  - → *few pairs pass the filter: results for this task are noisier; see scripts/analyze_tasks.py for an easier difficulty (gen_kwargs) or more shots*
- ✅ **OK** `var_binding`: n_shots=2 format=auto filter=strict retention=0.40
- ✅ **OK** `ioi_control`: n_shots=2 format=auto filter=strict retention=1.00
- ✅ **OK** `pilot`: all 3 SAEs recover >70% of CE on generic text (min 0.98)

## H1. Does the SAE error carry the task state? (exp1)
- ✅ **OK** sanity: full clean patch (from_diff) recovers NLDR 1.00 (should be ~1)
(judged on `from_diff` patching, at layers where the full clean patch restores >= 0.8)
- ❌ **FAIL** H1 rejected: `add_chain` most error-favourable layer L18: error_only 0.12 vs sae_only 0.88 (sae_only clearly higher); noise_control 0.21 (clean error adds -0.09 over erasing the corrupt error)
  - → *the SAE reconstruction carries this state; report as a negative result*
- ❌ **FAIL** H1 rejected: `entity_track` most error-favourable layer L12: error_only 0.61 vs sae_only 0.75 (sae_only clearly higher); noise_control 0.50 (clean error adds +0.11 over erasing the corrupt error)
  - → *the SAE reconstruction carries this state; report as a negative result*
- ❌ **FAIL** H1 rejected: `var_binding` most error-favourable layer L18: error_only 0.05 vs sae_only 0.95 (sae_only clearly higher); noise_control 0.00 (clean error adds +0.05 over erasing the corrupt error)
  - → *the SAE reconstruction carries this state; report as a negative result*
- ✅ **OK** control `ioi_control` most error-favourable layer L6: error_only 0.02 vs sae_only 0.98 (sae_only clearly higher); noise_control 0.01 (clean error adds +0.02 over erasing the corrupt error)

## H2. Is the state a low-rank manifold in the error? (exp2)
- ⚠️ **WARN** `add_chain` L12: probe acc eps 0.44 vs x_hat 0.35, prompt tokens only 0.46 (R^2 eps 0.97 vs tokens 1.00), eps selectivity 0.25
  - → *state is not more decodable from the error than from the reconstruction; H2 not supported here*
- ⚠️ **WARN** `entity_track` L12: probe acc eps 0.70 vs x_hat 0.36, prompt tokens only 0.30 (R^2 eps 0.60 vs tokens 0.01), eps selectivity 0.57
  - → *the probes read the final token, but patching the final token at L12 restores only 0.45: the state is linearly present in eps there but not yet causally used (the model reads it from earlier positions). Interesting, but not evidence for H1*
- ⚠️ **WARN** `var_binding` L12: probe acc eps 0.75 vs x_hat 0.18, prompt tokens only 1.00 (R^2 eps 0.86 vs tokens 1.00), eps selectivity 0.64
  - → *the probes read the final token, but patching the final token at L12 restores only 0.00: the state is linearly present in eps there but not yet causally used (the model reads it from earlier positions). Interesting, but not evidence for H1*

## H3. Does a small error subspace repair the state? (exp3)
(score: share of the gap 1 - sae_only closed by a rank-r error subspace; random subspace as the null)
- ✅ **OK** `add_chain` L18: the SAE misses 0.12 of the effect; a rank-1 error subspace (supervised_centroid_svd) closes 53% of that gap (random rank-1: 1%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*
- ✅ **OK** `entity_track` L12: the SAE misses 0.25 of the effect; a rank-1 error subspace (unsupervised_svd) closes 51% of that gap (random rank-1: 0%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*
- ✅ **OK** `var_binding` L18: the SAE misses 0.05 of the effect; a rank-8 error subspace (supervised_centroid_svd) closes 51% of that gap (random rank-8: 1%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*
- ✅ **OK** `ioi_control` L18: the SAE misses 0.24 of the effect; a rank-32 error subspace (supervised_centroid_svd) closes 55% of that gap (random rank-32: 2%)
  - → *a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether it is task-specific (generic_text_svd / transfer)*

## Width scaling (tier2_widths)
- ℹ️ **INFO** no width-sweep results (tier not run or deadline)

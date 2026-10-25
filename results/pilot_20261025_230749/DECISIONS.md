# Decisions for `pilot_20261025_230749`

## Verdict: 🟡 **MAYBE**

- Why: error_only beats sae_only for add_chain, entity_track, var_binding but the CIs overlap
- Next: tighten the CIs before committing a night: ./run.sh start --tiers tier1_core --hours 2 (400/200 pairs, 6 layers x {16k, 65k})

| task | H1 | best layer | error_only | sae_only | H2 probe (clean = beats x_hat and prompt tokens) | H3 (smallest rank with 90% repair) |
|---|---|---|---|---|---|---|
| `add_chain` | weak | L6 | 0.00 | -0.00 | no | - |
| `entity_track` | weak | L12 | 0.26 | 0.22 | clean | - |
| `var_binding` | weak | L12 | 0.00 | 0.00 | confounded | - |

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
- ⚠️ **WARN** H1 weak: `add_chain` best layer L6: error_only 0.00 vs sae_only -0.00 (CIs overlap)
  - → *more data: rerun tier3 (n=1000/500) for tighter CIs*
- ⚠️ **WARN** `add_chain` L6: error from another prompt with the SAME state 0.01 vs DIFFERENT state 0.00
  - → *the error carries prompt identity rather than a clean state code; be careful claiming a 'state manifold'*
- ⚠️ **WARN** H1 weak: `entity_track` best layer L12: error_only 0.26 vs sae_only 0.22 (CIs overlap)
  - → *more data: rerun tier3 (n=1000/500) for tighter CIs*
- ⚠️ **WARN** `entity_track` L12: error from another prompt with the SAME state 0.25 vs DIFFERENT state 0.24
  - → *the error carries prompt identity rather than a clean state code; be careful claiming a 'state manifold'*
- ⚠️ **WARN** H1 weak: `var_binding` best layer L12: error_only 0.00 vs sae_only 0.00 (CIs overlap)
  - → *more data: rerun tier3 (n=1000/500) for tighter CIs*
- ⚠️ **WARN** `var_binding` L12: error from another prompt with the SAME state 0.02 vs DIFFERENT state 0.02
  - → *the error carries prompt identity rather than a clean state code; be careful claiming a 'state manifold'*
- ✅ **OK** control `ioi_control` best layer L18: error_only 0.10 vs sae_only 0.56 (CIs overlap)

## H2. Is the state a low-rank manifold in the error? (exp2)
- ⚠️ **WARN** `add_chain` L12: probe acc eps 0.44 vs x_hat 0.35, prompt tokens only 0.46 (R^2 eps 0.97 vs tokens 1.00), eps selectivity 0.25
  - → *state is not more decodable from the error than from the reconstruction; H2 not supported here*
- ✅ **OK** `entity_track` L12: probe acc eps 0.70 vs x_hat 0.36, prompt tokens only 0.30 (R^2 eps 0.60 vs tokens 0.01), eps selectivity 0.57
- ⚠️ **WARN** `var_binding` L12: probe acc eps 0.75 vs x_hat 0.18, prompt tokens only 1.00 (R^2 eps 0.86 vs tokens 1.00), eps selectivity 0.64
  - → *the prompt's tokens alone decode the state as well: the probe cannot show the model computed it (expected for add_chain / var_binding); base the H2 claim on exp1/exp3 or on entity_track*

## H3. Does a small error subspace repair the state? (exp3)
- ⚠️ **WARN** `add_chain` L12: no rank <= 64 reaches 90% of full error, or random does as well
  - → *the state is spread out (high-rank) at this layer; look at other layers in pilot/SUMMARY.md*
- ⚠️ **WARN** `entity_track` L12: no rank <= 64 reaches 90% of full error, or random does as well
  - → *the state is spread out (high-rank) at this layer; look at other layers in pilot/SUMMARY.md*
- ⚠️ **WARN** `var_binding` L12: no rank <= 64 reaches 90% of full error, or random does as well
  - → *the state is spread out (high-rank) at this layer; look at other layers in pilot/SUMMARY.md*

## Width scaling (tier2_widths)
- ℹ️ **INFO** no width-sweep results (tier not run or deadline)

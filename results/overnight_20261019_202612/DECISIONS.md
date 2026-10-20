# Decisions for `overnight_20261019_202612`

## 0. Did the run work?
- ℹ️ **INFO** `preflight: reused`
- ℹ️ **INFO** `tier1_core: exit 0 after 20 min`
- ℹ️ **INFO** `tier2_layers: exit 0 after 32 min`
- ℹ️ **INFO** `tier2_widths: exit 0 after 44 min`
- ℹ️ **INFO** `tier3_expansions: exit 0 after 71 min`
- ⚠️ **WARN** `mod_sum`: n_shots=2 format=space_prompt filter=clean_only retention=0.10
  - → *few pairs pass the filter: results for this task are noisier; consider easier prompts (fewer steps)*
- ⚠️ **WARN** `dyck_depth`: n_shots=2 format=space_prompt filter=clean_only retention=0.21
  - → *few pairs pass the filter: results for this task are noisier; consider easier prompts (fewer steps)*
- ⚠️ **WARN** `graph_reach`: n_shots=4 format=space_prompt filter=clean_only retention=0.16
  - → *few pairs pass the filter: results for this task are noisier; consider easier prompts (fewer steps)*
- ❌ **FAIL** `ioi_control` dropped: gemma-2-2b cannot solve it under any prompt setting
  - → *make the task easier (mod_sum: n_steps 4->2; graph_reach: 3 edges) in src/data/synthetic_tasks.py, rerun with --resume*

## H1. Does the SAE error carry the task state? (exp1)
- ❌ **FAIL** no tier1 exp1 results
  - → *tier1 did not finish: check STATUS / failures, then --resume*

## H2. Is the state a low-rank manifold in the error? (exp2)
- ❌ **FAIL** no exp2 results
  - → *check tier1 failures*

## H3. Does a small error subspace repair the state? (exp3)
- ❌ **FAIL** no exp3 results
  - → *check tier1 failures*

## Width scaling (tier2_widths)
- ℹ️ **INFO** no width-sweep results (tier not run or deadline)

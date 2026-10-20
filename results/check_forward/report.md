# Forward / data check (2026-10-20 22:46)

**6 FAIL, 2 WARN, 23 PASS**

| status | check | detail |
|---|---|---|
| WARN | code matches a commit (no local edits outside results/) | modified: ['?? .env'] (full diff in git_diff.patch) |
| FAIL | transformers < 5 (TransformerLens 2.x is not tested with 5.x) | 5.20.1 |
| PASS | tokenizer prepends BOS (Gemma is garbage without it) | encode('When Mary') = [2, 4115, 7473], bos_token_id = 2 |
| PASS | mod_sum: answer is one token and prompts are token-aligned | answer token '6' |
| PASS | dyck_depth: answer is one token and prompts are token-aligned | answer token '3' |
| PASS | graph_reach: answer is one token and prompts are token-aligned | answer token '3' |
| PASS | ioi_control: answer is one token and prompts are token-aligned | answer token ' Bob' |
| PASS | mod_sum: prompt space fills train+test candidates | train 5600/5600, test 2800/2800 (n_train=400, n_test=200, x14) |
| FAIL | mod_sum: few-shot prefix actually gets its 2 shots | got 0/2 (shots are drawn after train/test have used up the prompt space) |
| FAIL | dyck_depth: prompt space fills train+test candidates | train 1140/2800, test 0/1400 (n_train=400, n_test=200, x7) |
| FAIL | dyck_depth: few-shot prefix actually gets its 2 shots | got 0/2 (shots are drawn after train/test have used up the prompt space) |
| PASS | graph_reach: prompt space fills train+test candidates | train 3600/3600, test 1800/1800 (n_train=400, n_test=200, x9) |
| PASS | graph_reach: few-shot prefix actually gets its 4 shots | got 4/4 (shots are drawn after train/test have used up the prompt space) |
| FAIL | ioi_control: prompt space fills train+test candidates | train 1200/1200, test 144/600 (n_train=400, n_test=200, x3) |
| PASS | ioi_control: few-shot prefix actually gets its 0 shots | got 0/0 (shots are drawn after train/test have used up the prompt space) |
| PASS | HF config final_logit_softcapping == 30 | 30.0 |
| PASS | TL output_logits_soft_cap == 30 | 30.0 |
| PASS | W_E carries Gemma's sqrt(d_model) embedding scale | median W_E row norm / W_U column norm = 48.0 (sqrt(d_model) = 48.0) |
| SKIP | logits ours vs tl_plain | not computed |
| SKIP | logits ours vs ours_gpu | not computed |
| SKIP | logits tl_plain vs hf | not computed |
| PASS | logits ours vs hf | max 4.717, median 0.493, top-1 agree 0.95 |
| PASS | all logits within the softcap (+-30) | largest abs logit 29.94 |
| PASS | resume from layer 2 == full forward | max dlogit 0.0000 |
| PASS | resume from layer 13 == full forward | max dlogit 0.0000 |
| PASS | resume from layer 23 == full forward | max dlogit 0.0000 |
| PASS | sanity prompts: ours top-1 correct on >= 5/7 | 6/7 correct |
| PASS | sanity prompts: HF top-1 correct on >= 5/7 | 6/7 correct |
| PASS | batched scoring == one-at-a-time scoring | max dl diff = 0.230 over 48 pairs (bf16) |
| PASS | task logit gaps within +-60 (softcap bound) | largest gap 9.3 at ioi_control/space_target/4shot |
| FAIL | IOI zero-shot clean top-1 >= 0.7 (gemma-2-2b normally ~0.9+) | clean top-1 0.69 |
| WARN | mod_sum: best setting beats chance by 1.5x | best clean top-1 0.17 at mod_sum/space_prompt/2shot (chance 0.14) |
| PASS | dyck_depth: best setting beats chance by 1.5x | best clean top-1 0.39 at dyck_depth/space_prompt/4shot (chance 0.17) |
| PASS | graph_reach: best setting beats chance by 1.5x | best clean top-1 0.31 at graph_reach/space_prompt/4shot (chance 0.17) |

## Logit agreement (last position, all probes)

| pair | max abs diff | median | top-1 agree |
|---|---|---|---|
| ours vs hf | 4.717 | 0.493 | 0.95 |

## Sanity prompts (top-5 next tokens)

| prompt | target | path | top-5 |
|---|---|---|---|
| 'The capital of France is' | ' Paris' | ours | [("' a'", 0.207), ("' the'", 0.1), ("' one'", 0.079), ("' also'", 0.079), ("' home'", 0.053)] |
| 'The capital of France is' | ' Paris' | hf | [("' a'", 0.201), ("' the'", 0.095), ("' one'", 0.074), ("' also'", 0.074), ("' home'", 0.058)] |
| 'When Mary and John went to the store, John gave a drink to' | ' Mary' | ours | [("' Mary'", 0.273), ("' the'", 0.202), ("' each'", 0.114), ("' a'", 0.093), ("' his'", 0.067)] |
| 'When Mary and John went to the store, John gave a drink to' | ' Mary' | hf | [("' Mary'", 0.266), ("' the'", 0.207), ("' each'", 0.126), ("' a'", 0.086), ("' his'", 0.067)] |
| 'When Tom and Anna went to the park, Anna gave a book to' | ' Tom' | ours | [("' Tom'", 0.628), ("' her'", 0.104), ("' the'", 0.076), ("' a'", 0.062), ("' Tommy'", 0.009)] |
| 'When Tom and Anna went to the park, Anna gave a book to' | ' Tom' | hf | [("' Tom'", 0.627), ("' her'", 0.109), ("' the'", 0.075), ("' a'", 0.058), ("' each'", 0.008)] |
| 'Monday, Tuesday, Wednesday,' | ' Thursday' | ours | [("' Thursday'", 0.723), ("' and'", 0.129), ("' '", 0.022), ("' Friday'", 0.021), ("'\\n'", 0.01)] |
| 'Monday, Tuesday, Wednesday,' | ' Thursday' | hf | [("' Thursday'", 0.705), ("' and'", 0.139), ("' Friday'", 0.021), ("' '", 0.021), ("'\\n'", 0.011)] |
| 'The opposite of hot is' | ' cold' | ours | [("' cold'", 0.572), ("' not'", 0.059), ("' cool'", 0.039), ("' what'", 0.023), ("' '", 0.022)] |
| 'The opposite of hot is' | ' cold' | hf | [("' cold'", 0.557), ("' not'", 0.059), ("' cool'", 0.04), ("' '", 0.024), ("' what'", 0.024)] |
| '1, 2, 3, 4,' | ' ' | ours | [("' '", 0.93), ("' and'", 0.01), ("' ...'", 0.008), ("'\\n'", 0.006), ("' …'", 0.005)] |
| '1, 2, 3, 4,' | ' ' | hf | [("' '", 0.934), ("' and'", 0.009), ("' ...'", 0.008), ("'\\n'", 0.005), ("' …'", 0.004)] |
| '1, 2, 3, 4, ' | '5' | ours | [("'5'", 0.938), ("'6'", 0.013), ("'1'", 0.012), ("'4'", 0.007), ("'2'", 0.005)] |
| '1, 2, 3, 4, ' | '5' | hf | [("'5'", 0.937), ("'6'", 0.015), ("'1'", 0.012), ("'4'", 0.007), ("'2'", 0.005)] |

## Task behaviour (ours)

| setting | n | chance | clean top-1 | corrupt top-1 | dl clean | dl corrupt | clean_only kept | top predictions |
|---|---|---|---|---|---|---|---|---|
| mod_sum/space_target/0shot | 0 (not single-token) | | | | | | | |
| mod_sum/space_target/2shot | 0 (not single-token) | | | | | | | |
| mod_sum/space_target/4shot | 0 (not single-token) | | | | | | | |
| mod_sum/space_prompt/0shot | 64 | 0.143 | 0.16 | 0.06 | 0.06 | 0.07 | 0.00 | [('1', 54), ('2', 10)] |
| mod_sum/space_prompt/2shot | 64 | 0.143 | 0.17 | 0.23 | -0.03 | -0.09 | 0.00 | [('2', 16), ('3', 12), ('1', 11), ('4', 10), ('5', 10)] |
| mod_sum/space_prompt/4shot | 64 | 0.143 | 0.12 | 0.17 | 0.01 | -0.00 | 0.00 | [('5', 22), ('4', 12), ('3', 9), ('2', 9), ('1', 7)] |
| dyck_depth/space_target/0shot | 0 (not single-token) | | | | | | | |
| dyck_depth/space_target/2shot | 0 (not single-token) | | | | | | | |
| dyck_depth/space_target/4shot | 0 (not single-token) | | | | | | | |
| dyck_depth/space_prompt/0shot | 64 | 0.167 | 0.38 | 0.39 | -0.10 | -0.12 | 0.00 | [('3', 14), ('5', 13), ('4', 12), ('2', 10), ('1', 9)] |
| dyck_depth/space_prompt/2shot | 64 | 0.167 | 0.36 | 0.33 | 0.08 | 0.01 | 0.00 | [('5', 16), ('4', 13), ('3', 11), ('6', 11), ('2', 10)] |
| dyck_depth/space_prompt/4shot | 64 | 0.167 | 0.39 | 0.39 | 0.08 | -0.05 | 0.00 | [('3', 20), ('4', 19), ('1', 9), ('6', 8), ('2', 8)] |
| graph_reach/space_target/0shot | 0 (not single-token) | | | | | | | |
| graph_reach/space_target/2shot | 0 (not single-token) | | | | | | | |
| graph_reach/space_target/4shot | 0 (not single-token) | | | | | | | |
| graph_reach/space_prompt/0shot | 64 | 0.167 | 0.17 | 0.17 | 0.07 | 0.04 | 0.00 | [('3', 22), ('1', 20), ('2', 17), ('4', 3), ('5', 2)] |
| graph_reach/space_prompt/2shot | 64 | 0.167 | 0.14 | 0.20 | -0.20 | -0.26 | 0.00 | [('2', 57), ('3', 7)] |
| graph_reach/space_prompt/4shot | 64 | 0.167 | 0.31 | 0.22 | -0.05 | -0.17 | 0.00 | [('4', 57), ('5', 7)] |
| ioi_control/space_target/0shot | 64 | 0.125 | 0.69 | 0.72 | 3.80 | -3.41 | 0.69 | [(' the', 19), (' Alice', 10), (' Bob', 8), (' Mary', 7), (' Lucy', 6)] |
| ioi_control/space_target/2shot | 64 | 0.125 | 1.00 | 1.00 | 5.27 | -4.68 | 1.00 | [(' Mike', 13), (' Bob', 12), (' Tom', 10), (' John', 8), (' Anna', 7)] |
| ioi_control/space_target/4shot | 64 | 0.125 | 1.00 | 1.00 | 7.73 | -7.20 | 1.00 | [(' Alice', 13), (' Lucy', 9), (' John', 9), (' Tom', 9), (' Anna', 8)] |
| ioi_control/space_prompt/0shot | 0 (not single-token) | | | | | | | |
| ioi_control/space_prompt/2shot | 0 (not single-token) | | | | | | | |
| ioi_control/space_prompt/4shot | 0 (not single-token) | | | | | | | |

## How to read this
| hf sanity | tl_plain vs hf | ours vs tl_plain | ours vs ours_gpu | tasks | Meaning |
|---|---|---|---|---|---|
| fails | - | - | - | - | Weights / tokenizer / environment problem: nothing to do with the harness |
| ok | disagree | - | - | - | TransformerLens (or its transformers version) converts Gemma-2 wrongly: pin TL / transformers versions |
| ok | agree | disagree | disagree | - | Bug in src/models/runtime.py head / forward path |
| ok | agree | disagree | agree | - | Bug in the CPU-embedding path only: set `runtime.offload_embed: false` and report |
| ok | agree | agree | agree | at chance | Model is fine; the task prompts are too hard: change prompts / few-shot, not code |


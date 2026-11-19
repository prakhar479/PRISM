# HF agreement on pilot task prompts 

96 pairs, 192 prompts, overall top-1 agreement 0.9896; near-tie threshold 0.25 logits; 79 s

| task | prompt | n | top-1 agree | disagreements | of which near-ties | median abs dD | max abs dD | median max abs logit diff |
|---|---|---|---|---|---|---|---|---|
| add_chain | A | 24 | 1.0000 | 0 | 0 | 0.065 | 0.178 | 0.413 |
| add_chain | B | 24 | 1.0000 | 0 | 0 | 0.083 | 0.251 | 0.431 |
| entity_track | A | 24 | 1.0000 | 0 | 0 | 0.054 | 0.170 | 0.401 |
| entity_track | B | 24 | 0.9167 | 2 | 2 | 0.108 | 0.240 | 0.522 |
| ioi_control | A | 24 | 1.0000 | 0 | 0 | 0.044 | 0.159 | 0.340 |
| ioi_control | B | 24 | 1.0000 | 0 | 0 | 0.079 | 0.380 | 0.350 |
| var_binding | A | 24 | 1.0000 | 0 | 0 | 0.066 | 0.350 | 0.592 |
| var_binding | B | 24 | 1.0000 | 0 | 0 | 0.085 | 0.317 | 0.527 |

## Pilot filter decisions, ours vs HF scores

| task | mode | pairs | kept (ours) | kept (hf) | flips |
|---|---|---|---|---|---|
| entity_track | clean_only | 24 | 24 | 24 | 0 |
| add_chain | strict | 24 | 23 | 24 | 1 |
| var_binding | strict | 24 | 24 | 24 | 0 |
| ioi_control | strict | 24 | 24 | 24 | 0 |

## Largest disagreements

| task | pair | prompt | ours top-1 (margin) | hf top-1 (margin) | D ours | D hf |
|---|---|---|---|---|---|---|
| entity_track | entity_track-test-00060 | B | ' C' (0.05) | ' D' (0.12) | 0.05 | -0.12 |
| entity_track | entity_track-test-00037 | B | ' B' (0.00) | ' A' (0.12) | 0.00 | 0.12 |

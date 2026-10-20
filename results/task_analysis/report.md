# Task capability analysis (2026-10-20 23:38)

model `gemma-2-2b` on `cuda`, n=128 pairs per cell, shots [0, 2, 4, 8], viable = top-1 >= 0.6, keep >= 0.3, Wilson lower bound > best baseline.

## Verdicts

| id | hypothesis | evidence | verdict |
|---|---|---|---|
| C0 | pipeline is sound (positive control) | IOI >=2-shot best top-1 1.00; sanity 6/7 | **PASS** |
| H1 | original tasks non-viable at the configured difficulty | mod_sum: keep<=0.00, acc 0.13 vs baseline 0.21; dyck_depth: keep<=0.00, acc 0.41 vs baseline 0.41; graph_reach: keep<=0.01, acc 0.30 vs baseline 0.48 | **CONFIRMED** |
| H2 | failure is difficulty-limited (easier levels become viable) | mod_sum: flat: prompt/format-limited (best acc by level 2:0.16, 3:0.18, 4:0.17); dyck_depth: flat: prompt/format-limited (best acc by level 2:0.50, 4:0.42, 6:0.45, 8:0.44); graph_reach: difficulty-sensitive but never viable (best acc by level 1:0.86, 2:0.52, 3:0.34, 4:0.30) | **REJECTED** |
| H3 | wrong answers follow a surface heuristic | mod_sum: wrong answers match 'last_add' 0.18 (chance 0.14); dyck_depth: wrong answers match 'start' 0.83 (chance 0.17); graph_reach: wrong answers match 'n_edges' 0.25 (chance 0.20) | **PARTIAL** |
| H4 | state is represented in the residual beyond what the prompt tokens give | mod_sum: best layer 2 probe 0.21 vs shuffled 0.15, surface-token probe 0.14; dyck_depth: best layer 6 probe 0.99 vs shuffled 0.18, surface-token probe 1.00 -> confounded: surface tokens already decode it; graph_reach: best layer 14 probe 0.53 vs shuffled 0.24, surface-token probe 0.47 | **REJECTED** |
| H5 | replacement tasks are viable | var_binding: 2 viable cells, hardest 1; entity_track: 3 viable cells, hardest 2; add_chain: 11 viable cells, hardest 3 | **CONFIRMED** |

## Sanity prompts

| prompt | target | top-1 | ok |
|---|---|---|---|
| 'The capital of France is' | ' Paris' | ' a' | no |
| 'When Mary and John went to the store, John gave a drink to' | ' Mary' | ' Mary' | yes |
| 'When Tom and Anna went to the park, Anna gave a book to' | ' Tom' | ' Tom' | yes |
| 'Monday, Tuesday, Wednesday,' | ' Thursday' | ' Thursday' | yes |
| 'The opposite of hot is' | ' cold' | ' cold' | yes |
| '1, 2, 3, 4, ' | '5' | '5' | yes |
| 'A, B, C, D,' | ' E' | ' E' | yes |

## All cells

Configured = the difficulty the overnight run used. delta = (clean gap) - (corrupt gap), the quantity the pipeline's filter thresholds at 2.

| task | level | shots | n | top-1 [95% CI] | answer-set top-1 | baseline (which) | keep | strict | delta median | answer mass | top predictions | viable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mod_sum | 2 | 0 | 128 | 0.07 [0.04, 0.13] | 0.07 | 0.20 (start) | 0.00 | 0.00 | -0.11 | 0.94 | '1':84, '2':33, '4':7, '3':4 | no |
| mod_sum | 2 | 2 | 128 | 0.16 [0.11, 0.24] | 0.16 | 0.23 (start) | 0.00 | 0.00 | -0.02 | 0.99 | '1':41, '5':27, '2':26, '3':17 | no |
| mod_sum | 2 | 4 | 128 | 0.14 [0.09, 0.21] | 0.14 | 0.20 (mode) | 0.00 | 0.00 | -0.08 | 1.00 | '1':52, '2':37, '0':18, '4':17 | no |
| mod_sum | 2 | 8 | 128 | 0.10 [0.06, 0.17] | 0.10 | 0.22 (start) | 0.00 | 0.00 | -0.04 | 0.99 | '4':26, '5':25, '2':23, '0':17 | no |
| mod_sum | 3 | 0 | 128 | 0.11 [0.07, 0.18] | 0.11 | 0.21 (mode) | 0.00 | 0.00 | -0.02 | 0.93 | '1':107, '2':19, '4':2 | no |
| mod_sum | 3 | 2 | 128 | 0.10 [0.06, 0.17] | 0.10 | 0.21 (mode) | 0.00 | 0.00 | 0.00 | 0.97 | '3':25, '1':23, '2':22, '6':20 | no |
| mod_sum | 3 | 4 | 128 | 0.13 [0.08, 0.20] | 0.13 | 0.20 (last_add) | 0.00 | 0.00 | 0.02 | 0.98 | '4':37, '5':30, '1':20, '2':16 | no |
| mod_sum | 3 | 8 | 128 | 0.18 [0.12, 0.26] | 0.18 | 0.19 (mode) | 0.00 | 0.00 | -0.03 | 1.00 | '0':108, '1':10, '2':9, '6':1 | no |
| mod_sum (configured) | 4 | 0 | 128 | 0.13 [0.08, 0.20] | 0.13 | 0.21 (mode) | 0.00 | 0.00 | 0.02 | 0.93 | '1':114, '2':12, '4':1, '3':1 | no |
| mod_sum (configured) | 4 | 2 | 128 | 0.12 [0.07, 0.18] | 0.12 | 0.20 (mode) | 0.00 | 0.00 | -0.00 | 0.97 | '4':93, '6':26, '3':7, '5':2 | no |
| mod_sum (configured) | 4 | 4 | 128 | 0.17 [0.12, 0.25] | 0.17 | 0.20 (mode) | 0.00 | 0.00 | 0.00 | 1.00 | '4':64, '1':45, '2':17, '3':1 | no |
| mod_sum (configured) | 4 | 8 | 128 | 0.15 [0.10, 0.22] | 0.15 | 0.20 (start) | 0.00 | 0.00 | 0.00 | 0.99 | '3':101, '2':14, '5':10, '0':2 | no |
| dyck_depth | 2 | 0 | 12 | 0.50 [0.25, 0.75] | 0.50 | 0.50 (start) | 0.00 | 0.00 | -0.04 | 0.86 | '4':4, '3':4, '2':2, '5':2 | no |
| dyck_depth | 2 | 2 | 8 | 0.00 [0.00, 0.32] | 0.12 | 0.50 (start) | 0.00 | 0.00 | 0.11 | 0.50 | '1':5, '2':3 | no |
| dyck_depth | 4 | 0 | 72 | 0.42 [0.31, 0.53] | 0.42 | 0.42 (start) | 0.00 | 0.00 | 0.03 | 0.86 | '3':20, '4':20, '5':12, '2':12 | no |
| dyck_depth | 4 | 2 | 62 | 0.40 [0.29, 0.53] | 0.40 | 0.40 (start) | 0.00 | 0.00 | 0.22 | 0.93 | '3':24, '4':14, '2':8, '5':8 | no |
| dyck_depth | 4 | 4 | 56 | 0.12 [0.06, 0.24] | 0.12 | 0.41 (start) | 0.00 | 0.00 | 0.35 | 0.96 | '3':34, '1':20, '2':2 | no |
| dyck_depth | 4 | 8 | 44 | 0.34 [0.22, 0.49] | 0.34 | 0.36 (start) | 0.00 | 0.00 | 0.55 | 0.93 | '3':28, '4':14, '1':2 | no |
| dyck_depth | 6 | 0 | 128 | 0.40 [0.32, 0.49] | 0.40 | 0.40 (start) | 0.00 | 0.00 | 0.00 | 0.85 | '3':28, '4':26, '2':25, '5':22 | no |
| dyck_depth | 6 | 2 | 128 | 0.45 [0.36, 0.53] | 0.45 | 0.45 (start) | 0.00 | 0.00 | 0.27 | 0.99 | '3':32, '4':31, '2':23, '5':20 | no |
| dyck_depth | 6 | 4 | 128 | 0.41 [0.33, 0.49] | 0.41 | 0.36 (count_plus) | 0.00 | 0.00 | 0.28 | 0.97 | '2':74, '3':31, '4':12, '1':11 | no |
| dyck_depth | 6 | 8 | 128 | 0.39 [0.31, 0.48] | 0.39 | 0.41 (start) | 0.00 | 0.00 | 0.26 | 0.97 | '2':42, '5':32, '3':24, '6':12 | no |
| dyck_depth (configured) | 8 | 0 | 128 | 0.41 [0.33, 0.49] | 0.41 | 0.41 (start) | 0.00 | 0.00 | 0.00 | 0.85 | '2':28, '4':25, '5':23, '3':21 | no |
| dyck_depth (configured) | 8 | 2 | 128 | 0.35 [0.27, 0.44] | 0.35 | 0.35 (start) | 0.00 | 0.00 | 0.10 | 0.89 | '3':27, '5':25, '4':25, '2':19 | no |
| dyck_depth (configured) | 8 | 4 | 128 | 0.44 [0.35, 0.52] | 0.44 | 0.44 (start) | 0.00 | 0.00 | 0.20 | 0.99 | '2':26, '4':25, '5':25, '3':24 | no |
| dyck_depth (configured) | 8 | 8 | 128 | 0.38 [0.30, 0.47] | 0.38 | 0.40 (start) | 0.00 | 0.00 | 0.27 | 0.97 | '4':71, '5':28, '3':19, '2':9 | no |
| graph_reach | 1 | 0 | 128 | 0.66 [0.57, 0.73] | 0.69 | 0.62 (n_edges) | 0.00 | 0.00 | 0.01 | 0.40 | '1':119, '2':5, '3':4 | no |
| graph_reach | 1 | 2 | 128 | 0.69 [0.60, 0.76] | 0.69 | 0.61 (n_edges) | 0.00 | 0.00 | 0.18 | 0.74 | '1':118, '2':10 | no |
| graph_reach | 1 | 4 | 128 | 0.86 [0.79, 0.91] | 0.86 | 0.58 (n_edges) | 0.00 | 0.00 | 0.69 | 0.86 | '1':92, '2':36 | no |
| graph_reach | 1 | 8 | 96 | 0.50 [0.40, 0.60] | 0.50 | 0.50 (n_edges) | 0.00 | 0.00 | 1.17 | 0.96 | '1':96 | no |
| graph_reach | 2 | 0 | 128 | 0.39 [0.31, 0.48] | 0.39 | 0.88 (direct_neighbours_plus_1) | 0.00 | 0.00 | 0.02 | 0.57 | '3':48, '2':46, '1':34 | no |
| graph_reach | 2 | 2 | 128 | 0.33 [0.25, 0.41] | 0.33 | 0.84 (direct_neighbours_plus_1) | 0.00 | 0.00 | 0.01 | 0.84 | '2':79, '1':49 | no |
| graph_reach | 2 | 4 | 128 | 0.40 [0.32, 0.49] | 0.40 | 0.79 (direct_neighbours_plus_1) | 0.00 | 0.00 | 0.05 | 0.88 | '2':83, '1':33, '3':12 | no |
| graph_reach | 2 | 8 | 128 | 0.52 [0.44, 0.61] | 0.52 | 0.89 (direct_neighbours_plus_1) | 0.01 | 0.00 | 0.36 | 0.96 | '1':115, '3':13 | no |
| graph_reach | 3 | 0 | 128 | 0.23 [0.16, 0.31] | 0.23 | 0.62 (direct_neighbours_plus_1) | 0.00 | 0.00 | 0.01 | 0.71 | '3':89, '1':20, '2':17, '4':2 | no |
| graph_reach | 3 | 2 | 128 | 0.20 [0.14, 0.27] | 0.20 | 0.57 (direct_neighbours_plus_1) | 0.00 | 0.00 | -0.02 | 0.91 | '2':99, '3':21, '4':8 | no |
| graph_reach | 3 | 4 | 128 | 0.29 [0.22, 0.37] | 0.29 | 0.57 (direct_neighbours_plus_1) | 0.00 | 0.00 | -0.01 | 0.96 | '1':128 | no |
| graph_reach | 3 | 8 | 128 | 0.34 [0.26, 0.42] | 0.34 | 0.67 (direct_neighbours_plus_1) | 0.00 | 0.00 | 0.17 | 0.98 | '1':116, '2':12 | no |
| graph_reach (configured) | 4 | 0 | 128 | 0.16 [0.10, 0.23] | 0.16 | 0.48 (n_nodes_mentioned) | 0.00 | 0.00 | 0.02 | 0.82 | '3':51, '1':35, '2':34, '4':7 | no |
| graph_reach (configured) | 4 | 2 | 128 | 0.19 [0.13, 0.26] | 0.19 | 0.48 (n_nodes_mentioned) | 0.00 | 0.00 | 0.03 | 0.91 | '4':51, '3':46, '5':31 | no |
| graph_reach (configured) | 4 | 4 | 128 | 0.15 [0.10, 0.22] | 0.15 | 0.41 (n_nodes_mentioned) | 0.00 | 0.00 | 0.02 | 0.97 | '3':97, '4':27, '2':4 | no |
| graph_reach (configured) | 4 | 8 | 128 | 0.30 [0.22, 0.38] | 0.30 | 0.48 (direct_neighbours_plus_1) | 0.01 | 0.00 | 0.10 | 0.94 | '5':71, '4':53, '3':4 | no |
| ioi_control | 0 | 0 | 128 | 0.73 [0.65, 0.80] | 1.00 | 0.18 (mode) | 0.73 | 0.66 | 7.57 | 0.43 | 'the':30, 'Lucy':20, 'Mary':18, 'Mike':12 | **yes** |
| ioi_control | 0 | 2 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.17 (mode) | 1.00 | 1.00 | 12.34 | 0.99 | 'John':22, 'Tom':20, 'Lucy':19, 'Alice':16 | **yes** |
| ioi_control | 0 | 4 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.17 (mode) | 1.00 | 1.00 | 11.38 | 0.99 | 'Mike':22, 'Anna':20, 'Bob':17, 'Lucy':16 | **yes** |
| ioi_control | 0 | 8 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.16 (mode) | 1.00 | 1.00 | 15.61 | 1.00 | 'Alice':20, 'Anna':20, 'Bob':18, 'Lucy':18 | **yes** |
| var_binding | 1 | 0 | 128 | 0.12 [0.08, 0.19] | 0.12 | 0.14 (mode) | 0.09 | 0.05 | 1.29 | 0.97 | '7':26, '2':21, '3':18, '4':16 | no |
| var_binding | 1 | 2 | 128 | 0.22 [0.16, 0.30] | 0.22 | 0.16 (mode) | 0.20 | 0.06 | 2.60 | 0.99 | '2':27, '3':26, '1':22, '8':15 | no |
| var_binding | 1 | 4 | 128 | 0.65 [0.56, 0.73] | 0.65 | 0.16 (mode) | 0.65 | 0.51 | 8.36 | 1.00 | '3':22, '5':17, '4':16, '6':15 | **yes** |
| var_binding | 1 | 8 | 128 | 0.95 [0.89, 0.97] | 0.95 | 0.14 (mode) | 0.95 | 0.88 | 11.19 | 1.00 | '5':19, '2':18, '7':16, '6':16 | **yes** |
| var_binding | 2 | 0 | 128 | 0.12 [0.08, 0.19] | 0.12 | 0.15 (mode) | 0.03 | 0.00 | 0.85 | 0.97 | '2':40, '1':26, '6':15, '7':11 | no |
| var_binding | 2 | 2 | 128 | 0.23 [0.17, 0.31] | 0.23 | 0.19 (mode) | 0.12 | 0.03 | 2.10 | 0.99 | '6':72, '9':17, '7':10, '5':7 | no |
| var_binding | 2 | 4 | 128 | 0.18 [0.12, 0.26] | 0.18 | 0.16 (mode) | 0.02 | 0.00 | 0.59 | 0.99 | '8':46, '6':37, '7':14, '5':9 | no |
| var_binding | 2 | 8 | 128 | 0.23 [0.16, 0.31] | 0.23 | 0.15 (mode) | 0.00 | 0.00 | 0.10 | 1.00 | '3':61, '4':37, '2':8, '9':6 | no |
| var_binding | 3 | 0 | 128 | 0.09 [0.05, 0.16] | 0.09 | 0.16 (mode) | 0.00 | 0.00 | 0.46 | 0.97 | '1':29, '2':22, '4':17, '3':16 | no |
| var_binding | 3 | 2 | 128 | 0.10 [0.06, 0.17] | 0.10 | 0.14 (mode) | 0.00 | 0.00 | 0.21 | 0.99 | '7':79, '1':26, '3':9, '9':4 | no |
| var_binding | 3 | 4 | 128 | 0.05 [0.02, 0.10] | 0.05 | 0.16 (mode) | 0.00 | 0.00 | 0.21 | 1.00 | '4':31, '7':30, '3':14, '6':12 | no |
| var_binding | 3 | 8 | 128 | 0.05 [0.03, 0.11] | 0.05 | 0.15 (mode) | 0.00 | 0.00 | -0.13 | 1.00 | '8':76, '7':50, '3':1, '4':1 | no |
| entity_track | 1 | 0 | 128 | 0.02 [0.00, 0.06] | 0.02 | 0.52 (first_box) | 0.00 | 0.00 | 0.40 | 0.89 | 'B':34, 'C':30, 'D':14, 'A':13 | no |
| entity_track | 1 | 2 | 128 | 0.99 [0.96, 1.00] | 0.99 | 0.50 (first_box) | 0.20 | 0.02 | 1.43 | 0.98 | 'A':26, 'F':23, 'E':20, 'G':20 | no |
| entity_track | 1 | 4 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.47 (first_box) | 0.56 | 0.09 | 2.10 | 0.99 | 'E':24, 'A':21, 'D':19, 'C':19 | **yes** |
| entity_track | 1 | 8 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.52 (first_box) | 0.64 | 0.02 | 2.26 | 1.00 | 'G':25, 'C':21, 'B':21, 'F':19 | **yes** |
| entity_track | 2 | 0 | 128 | 0.18 [0.12, 0.26] | 0.18 | 0.67 (initial_box) | 0.00 | 0.00 | 0.37 | 0.94 | 'B':28, 'C':26, 'D':25, 'A':21 | no |
| entity_track | 2 | 2 | 128 | 0.20 [0.14, 0.28] | 0.20 | 0.64 (initial_box) | 0.00 | 0.00 | 0.03 | 0.99 | 'B':28, 'F':21, 'C':20, 'G':16 | no |
| entity_track | 2 | 4 | 128 | 0.70 [0.62, 0.78] | 0.70 | 0.59 (initial_box) | 0.15 | 0.00 | 1.26 | 0.99 | 'C':33, 'D':20, 'B':20, 'A':16 | no |
| entity_track | 2 | 8 | 128 | 0.98 [0.94, 1.00] | 0.98 | 0.65 (initial_box) | 0.84 | 0.50 | 3.42 | 1.00 | 'G':24, 'D':22, 'C':20, 'F':20 | **yes** |
| entity_track | 3 | 0 | 128 | 0.16 [0.11, 0.24] | 0.16 | 0.55 (initial_box) | 0.00 | 0.00 | 0.45 | 0.94 | 'C':34, 'B':33, 'A':18, 'F':16 | no |
| entity_track | 3 | 2 | 128 | 0.15 [0.10, 0.22] | 0.15 | 0.48 (initial_box) | 0.00 | 0.00 | -0.00 | 0.99 | 'B':40, 'C':28, 'A':22, 'D':15 | no |
| entity_track | 3 | 4 | 128 | 0.66 [0.57, 0.73] | 0.66 | 0.66 (initial_box) | 0.14 | 0.00 | 0.87 | 1.00 | 'D':25, 'F':21, 'E':19, 'A':18 | no |
| entity_track | 3 | 8 | 128 | 0.72 [0.64, 0.79] | 0.72 | 0.52 (initial_box) | 0.29 | 0.10 | 1.82 | 1.00 | 'C':25, 'B':22, 'D':21, 'E':20 | no |
| add_chain | 1 | 0 | 45 | 1.00 [0.92, 1.00] | 1.00 | 0.20 (last_add) | 1.00 | 1.00 | 11.10 | 0.99 | '3':9, '4':9, '5':9, '6':6 | **yes** |
| add_chain | 1 | 2 | 30 | 1.00 [0.89, 1.00] | 1.00 | 0.30 (mode) | 1.00 | 1.00 | 11.95 | 1.00 | '4':9, '5':7, '3':6, '6':4 | **yes** |
| add_chain | 1 | 4 | 22 | 1.00 [0.85, 1.00] | 1.00 | 0.41 (last_add) | 1.00 | 1.00 | 13.17 | 1.00 | '3':5, '4':4, '5':4, '1':3 | **yes** |
| add_chain | 1 | 8 | 11 | 1.00 [0.74, 1.00] | 1.00 | 0.82 (last_add) | 1.00 | 1.00 | 13.77 | 1.00 | '3':3, '2':3, '1':3, '4':1 | no |
| add_chain | 2 | 0 | 128 | 0.94 [0.88, 0.97] | 0.94 | 0.23 (mode) | 0.94 | 0.88 | 6.89 | 0.93 | '6':30, '5':28, '7':22, '4':18 | **yes** |
| add_chain | 2 | 2 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.27 (mode) | 1.00 | 1.00 | 9.72 | 0.99 | '6':34, '5':24, '7':21, '8':17 | **yes** |
| add_chain | 2 | 4 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.20 (mode) | 1.00 | 1.00 | 11.60 | 1.00 | '5':26, '6':25, '7':23, '4':19 | **yes** |
| add_chain | 2 | 8 | 128 | 1.00 [0.97, 1.00] | 1.00 | 0.24 (mode) | 1.00 | 1.00 | 10.45 | 0.99 | '5':31, '4':27, '6':19, '7':14 | **yes** |
| add_chain | 3 | 0 | 128 | 0.73 [0.64, 0.80] | 0.75 | 0.27 (mode) | 0.68 | 0.46 | 4.09 | 0.84 | '8':38, '6':28, '7':19, '5':12 | **yes** |
| add_chain | 3 | 2 | 128 | 0.99 [0.96, 1.00] | 0.99 | 0.28 (mode) | 0.99 | 0.89 | 7.26 | 0.98 | '8':36, '7':33, '6':20, '5':19 | **yes** |
| add_chain | 3 | 4 | 128 | 0.96 [0.91, 0.98] | 0.96 | 0.28 (mode) | 0.96 | 0.76 | 7.08 | 0.99 | '7':40, '8':26, '6':23, '9':20 | **yes** |
| add_chain | 3 | 8 | 128 | 0.98 [0.93, 0.99] | 0.98 | 0.26 (mode) | 0.98 | 0.88 | 7.15 | 0.99 | '7':36, '8':33, '6':22, '9':17 | **yes** |

## Which shortcut do wrong answers follow? (configured level, mean over shots)

Heuristics with accuracy >= 0.95 equal the true solution at that level and are not used as baselines.

| task | heuristic | heuristic accuracy | share of wrong answers matching it |
|---|---|---|---|
| mod_sum | start | 0.14 | 0.11 |
| mod_sum | last_add | 0.14 | 0.18 |
| mod_sum | first_add | 0.13 | 0.16 |
| mod_sum | start_plus_last_mod | 0.14 | 0.16 |
| mod_sum | sum_no_mod_last_digit | 0.00 | 0.13 |
| dyck_depth | start | 0.40 | 0.83 |
| dyck_depth | start_plus_last_op | 0.00 | 0.02 |
| dyck_depth | count_plus | 0.31 | 0.12 |
| dyck_depth | count_minus | 0.15 | 0.37 |
| graph_reach | direct_neighbours_plus_1 | 0.42 | 0.17 |
| graph_reach | n_edges | 0.27 | 0.25 |
| graph_reach | n_nodes_mentioned | 0.45 | 0.13 |
| var_binding | last_number | 0.00 | 0.33 |
| var_binding | first_number | 0.11 | 0.34 |
| entity_track | initial_box | 0.73 | 0.26 |
| entity_track | last_box_mentioned | 0.04 | 0.15 |
| entity_track | first_box | 0.32 | 0.08 |
| add_chain | start | 0.00 | 0.00 |
| add_chain | last_add | 0.12 | 0.00 |
| add_chain | start_plus_last | 0.33 | 0.08 |

## Does the corrupted step matter? (median delta by corrupted position, configured level, most shots)

- **mod_sum** (8 shot): {'0': -0.009, '1': 0.01, '2': -0.001}
- **dyck_depth** (8 shot): {'0': 0.177, '1': 0.451, '2': 0.285, '3': 0.268}
- **graph_reach** (8 shot): {'-1': 0.186}

## Linear probes on the residual (last position, clean state, 5-fold CV)

A residual probe only shows computation if it beats the surface-token probe (same classifier on token counts + position one-hots of the prompt).

| task | level | shots | layer | probe acc | shuffled | surface tokens | majority | n |
|---|---|---|---|---|---|---|---|---|
| mod_sum | 4 | 2 | 2 | 0.21 | 0.15 | 0.14 | 0.15 | 400 |
| mod_sum | 4 | 2 | 6 | 0.17 | 0.15 | 0.14 | 0.15 | 400 |
| mod_sum | 4 | 2 | 10 | 0.12 | 0.14 | 0.14 | 0.15 | 400 |
| mod_sum | 4 | 2 | 14 | 0.13 | 0.15 | 0.14 | 0.15 | 400 |
| mod_sum | 4 | 2 | 18 | 0.13 | 0.14 | 0.14 | 0.15 | 400 |
| mod_sum | 4 | 2 | 22 | 0.14 | 0.15 | 0.14 | 0.15 | 400 |
| dyck_depth | 8 | 2 | 2 | 0.99 | 0.21 | 1.00 | 0.23 | 400 |
| dyck_depth | 8 | 2 | 6 | 0.99 | 0.18 | 1.00 | 0.23 | 400 |
| dyck_depth | 8 | 2 | 10 | 0.95 | 0.17 | 1.00 | 0.23 | 400 |
| dyck_depth | 8 | 2 | 14 | 0.96 | 0.18 | 1.00 | 0.23 | 400 |
| dyck_depth | 8 | 2 | 18 | 0.94 | 0.18 | 1.00 | 0.23 | 400 |
| dyck_depth | 8 | 2 | 22 | 0.93 | 0.19 | 1.00 | 0.23 | 400 |
| graph_reach | 4 | 2 | 2 | 0.42 | 0.25 | 0.47 | 0.29 | 400 |
| graph_reach | 4 | 2 | 6 | 0.44 | 0.21 | 0.47 | 0.29 | 400 |
| graph_reach | 4 | 2 | 10 | 0.48 | 0.25 | 0.47 | 0.29 | 400 |
| graph_reach | 4 | 2 | 14 | 0.53 | 0.24 | 0.47 | 0.29 | 400 |
| graph_reach | 4 | 2 | 18 | 0.50 | 0.24 | 0.47 | 0.29 | 400 |
| graph_reach | 4 | 2 | 22 | 0.51 | 0.26 | 0.47 | 0.29 | 400 |
| var_binding | 2 | 2 | 2 | 0.36 | 0.10 | 0.71 | 0.13 | 400 |
| var_binding | 2 | 2 | 6 | 0.33 | 0.11 | 0.71 | 0.13 | 400 |
| var_binding | 2 | 2 | 10 | 0.30 | 0.11 | 0.71 | 0.13 | 400 |
| var_binding | 2 | 2 | 14 | 0.36 | 0.10 | 0.71 | 0.13 | 400 |
| var_binding | 2 | 2 | 18 | 0.50 | 0.10 | 0.71 | 0.13 | 400 |
| var_binding | 2 | 2 | 22 | 0.50 | 0.12 | 0.71 | 0.13 | 400 |
| entity_track | 2 | 2 | 2 | 0.27 | 0.10 | 0.32 | 0.17 | 400 |
| entity_track | 2 | 2 | 6 | 0.39 | 0.12 | 0.32 | 0.17 | 400 |
| entity_track | 2 | 2 | 10 | 0.51 | 0.13 | 0.32 | 0.17 | 400 |
| entity_track | 2 | 2 | 14 | 0.80 | 0.11 | 0.32 | 0.17 | 400 |
| entity_track | 2 | 2 | 18 | 0.89 | 0.12 | 0.32 | 0.17 | 400 |
| entity_track | 2 | 2 | 22 | 0.87 | 0.16 | 0.32 | 0.17 | 400 |
| add_chain | 2 | 2 | 2 | 1.00 | 0.20 | 1.00 | 0.23 | 208 |
| add_chain | 2 | 2 | 6 | 1.00 | 0.21 | 1.00 | 0.23 | 208 |
| add_chain | 2 | 2 | 10 | 1.00 | 0.19 | 1.00 | 0.23 | 208 |
| add_chain | 2 | 2 | 14 | 1.00 | 0.20 | 1.00 | 0.23 | 208 |
| add_chain | 2 | 2 | 18 | 1.00 | 0.19 | 1.00 | 0.23 | 208 |
| add_chain | 2 | 2 | 22 | 1.00 | 0.21 | 1.00 | 0.23 | 208 |

## Recommended settings (hardest viable level, fewest shots)

```json
{
 "mod_sum": {
  "viable": false
 },
 "dyck_depth": {
  "viable": false
 },
 "graph_reach": {
  "viable": false
 },
 "ioi_control": {
  "viable": true,
  "level": 0,
  "n_shots": 0,
  "filter_mode": "clean_only",
  "expected_retention": 0.734,
  "candidate_multiplier": 3,
  "acc": 0.734,
  "baseline": 0.18
 },
 "var_binding": {
  "viable": true,
  "depth": 1,
  "n_shots": 4,
  "filter_mode": "clean_only",
  "expected_retention": 0.648,
  "candidate_multiplier": 3,
  "acc": 0.648,
  "baseline": 0.156
 },
 "entity_track": {
  "viable": true,
  "depth": 2,
  "n_shots": 8,
  "filter_mode": "clean_only",
  "expected_retention": 0.836,
  "candidate_multiplier": 2,
  "acc": 0.984,
  "baseline": 0.648
 },
 "add_chain": {
  "viable": true,
  "depth": 3,
  "n_shots": 0,
  "filter_mode": "clean_only",
  "expected_retention": 0.68,
  "candidate_multiplier": 3,
  "acc": 0.727,
  "baseline": 0.273
 }
}
```

Figures: `figures/accuracy_vs_difficulty.png`, `figures/keep_vs_difficulty.png` (dotted line = configured level).

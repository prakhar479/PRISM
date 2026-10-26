# PRISM summary: `paper_widths`

device `cuda` dtype `torch.bfloat16` mock=False timings {'exp1': 376.5, 'exp2': 81.1, 'exp3': 2604.5} total Nones

## Tasks

| task | n_train | n_test | skipped | n_shots | format | filter | dL clean | dL corrupt | clean top1 |
|---|---|---|---|---|---|---|---|---|---|
| entity_track | 300 | 150 | False | 8 | auto | clean_only | 2.77 | -0.15 | 1.00 |
| add_chain | 300 | 150 | False | 8 | auto | strict | 3.48 | -3.02 | 1.00 |
| ioi_control | 300 | 150 | False | 2 | auto | strict | 6.75 | -6.39 | 1.00 |

## Exp1: causal split (NLDR, 95% bootstrap CI)

| task | layer | width | pos | rel err | clean_full | sae_only | error_only | sae_splice | noise_control |
|---|---|---|---|---|---|---|---|---|---|
| add_chain | 12 | 131k | from_diff | 0.12 | 1.00 [1.00,1.00] | 0.95 [0.94,0.96] | 0.04 [0.03,0.06] | 1.03 [1.01,1.04] | 0.05 [0.02,0.07] |
| add_chain | 12 | 16k | from_diff | 0.17 | 1.00 [1.00,1.00] | 0.94 [0.93,0.95] | 0.06 [0.05,0.07] | 1.02 [1.00,1.04] | 0.07 [0.04,0.09] |
| add_chain | 12 | 262k | from_diff | 0.11 | 1.00 [1.00,1.00] | 0.95 [0.93,0.96] | 0.05 [0.04,0.06] | 1.01 [0.99,1.02] | 0.05 [0.03,0.07] |
| add_chain | 12 | 32k | from_diff | 0.15 | 1.00 [1.00,1.00] | 0.94 [0.92,0.95] | 0.05 [0.04,0.07] | 1.03 [1.01,1.05] | 0.05 [0.03,0.08] |
| add_chain | 12 | 65k | from_diff | 0.15 | 1.00 [1.00,1.00] | 0.93 [0.91,0.94] | 0.07 [0.05,0.09] | 1.05 [1.03,1.07] | 0.03 [0.01,0.07] |
| entity_track | 12 | 131k | from_diff | 0.13 | 1.00 [1.00,1.00] | 0.77 [0.75,0.78] | 0.54 [0.50,0.57] | 0.92 [0.91,0.93] | 0.49 [0.45,0.53] |
| entity_track | 12 | 16k | from_diff | 0.18 | 1.00 [1.00,1.00] | 0.74 [0.73,0.76] | 0.62 [0.59,0.65] | 1.09 [1.07,1.11] | 0.51 [0.47,0.55] |
| entity_track | 12 | 262k | from_diff | 0.13 | 1.00 [1.00,1.00] | 0.85 [0.84,0.86] | 0.46 [0.44,0.49] | 0.99 [0.97,1.01] | 0.48 [0.44,0.51] |
| entity_track | 12 | 32k | from_diff | 0.18 | 1.00 [1.00,1.00] | 0.80 [0.79,0.82] | 0.52 [0.49,0.55] | 1.13 [1.12,1.15] | 0.60 [0.56,0.63] |
| entity_track | 12 | 65k | from_diff | 0.17 | 1.00 [1.00,1.00] | 0.76 [0.74,0.77] | 0.59 [0.55,0.62] | 0.98 [0.96,0.99] | 0.45 [0.41,0.49] |
| ioi_control | 12 | 131k | from_diff | 0.10 | 1.00 [1.00,1.00] | 0.98 [0.98,0.99] | 0.04 [0.03,0.05] | 0.98 [0.98,0.99] | 0.06 [0.05,0.07] |
| ioi_control | 12 | 16k | from_diff | 0.14 | 1.00 [1.00,1.00] | 0.96 [0.95,0.97] | 0.13 [0.11,0.14] | 0.97 [0.97,0.98] | 0.10 [0.08,0.11] |
| ioi_control | 12 | 262k | from_diff | 0.10 | 1.00 [1.00,1.00] | 1.00 [0.99,1.00] | 0.05 [0.04,0.06] | 0.99 [0.99,1.00] | 0.04 [0.03,0.05] |
| ioi_control | 12 | 32k | from_diff | 0.14 | 1.00 [1.00,1.00] | 0.97 [0.96,0.98] | 0.09 [0.08,0.11] | 0.97 [0.96,0.98] | 0.09 [0.08,0.11] |
| ioi_control | 12 | 65k | from_diff | 0.13 | 1.00 [1.00,1.00] | 0.98 [0.97,0.99] | 0.06 [0.05,0.07] | 0.98 [0.97,0.98] | 0.07 [0.06,0.08] |

## Exp2: test probe accuracy (selectivity) / R^2 (circular for mod_sum)

`surface` = the same probe on the prompt's tokens only: eps must beat it for a state claim.

| task | layer | width | x | x_hat | eps | latents | surface | chance |
|---|---|---|---|---|---|---|---|---|
| add_chain | 12 | 131k | 0.57 (0.35) / 0.99 | 0.44 (0.20) / 0.86 | 0.52 (0.31) / 0.98 | - | 0.45 (0.22) / 1.00 | 0.25 |
| add_chain | 12 | 16k | 0.57 (0.35) / 0.99 | 0.42 (0.21) / 0.81 | 0.53 (0.33) / 0.98 | - | 0.45 (0.22) / 1.00 | 0.25 |
| add_chain | 12 | 262k | 0.57 (0.35) / 0.99 | 0.46 (0.23) / 0.88 | 0.53 (0.30) / 0.97 | - | 0.45 (0.22) / 1.00 | 0.25 |
| add_chain | 12 | 32k | 0.57 (0.35) / 0.99 | 0.40 (0.16) / 0.78 | 0.50 (0.24) / 0.98 | - | 0.45 (0.22) / 1.00 | 0.25 |
| add_chain | 12 | 65k | 0.57 (0.35) / 0.99 | 0.40 (0.17) / 0.78 | 0.49 (0.29) / 0.97 | - | 0.45 (0.22) / 1.00 | 0.25 |
| entity_track | 12 | 131k | 0.91 (0.73) / 0.75 | 0.55 (0.37) / 0.52 | 0.81 (0.63) / 0.69 | - | 0.33 (0.18) / 0.08 | 0.18 |
| entity_track | 12 | 16k | 0.91 (0.73) / 0.75 | 0.48 (0.32) / 0.44 | 0.84 (0.65) / 0.71 | - | 0.33 (0.18) / 0.08 | 0.18 |
| entity_track | 12 | 262k | 0.91 (0.73) / 0.75 | 0.55 (0.34) / 0.55 | 0.82 (0.63) / 0.68 | - | 0.33 (0.18) / 0.08 | 0.18 |
| entity_track | 12 | 32k | 0.91 (0.73) / 0.75 | 0.44 (0.28) / 0.43 | 0.85 (0.69) / 0.72 | - | 0.33 (0.18) / 0.08 | 0.18 |
| entity_track | 12 | 65k | 0.91 (0.73) / 0.75 | 0.47 (0.29) / 0.44 | 0.81 (0.61) / 0.69 | - | 0.33 (0.18) / 0.08 | 0.18 |
| ioi_control | 12 | 131k | 1.00 (0.94) / 0.78 | 0.29 (0.24) / 0.00 | 0.88 (0.83) / 0.43 | - | 1.00 (0.92) / 1.00 | 0.11 |
| ioi_control | 12 | 16k | 1.00 (0.94) / 0.78 | 0.20 (0.16) / 0.03 | 0.97 (0.93) / 0.55 | - | 1.00 (0.92) / 1.00 | 0.11 |
| ioi_control | 12 | 262k | 1.00 (0.94) / 0.78 | 0.31 (0.26) / 0.02 | 0.90 (0.86) / 0.43 | - | 1.00 (0.92) / 1.00 | 0.11 |
| ioi_control | 12 | 32k | 1.00 (0.94) / 0.78 | 0.19 (0.14) / 0.03 | 0.96 (0.90) / 0.51 | - | 1.00 (0.92) / 1.00 | 0.11 |
| ioi_control | 12 | 65k | 1.00 (0.94) / 0.78 | 0.18 (0.13) / 0.01 | 0.98 (0.93) / 0.57 | - | 1.00 (0.92) / 1.00 | 0.11 |

## Exp3: hybrid patch NLDR at each rank (method rows; reference = full error_only)

| task | layer | width | pos | method | r=0 | r=1 | r=2 | r=4 | r=8 | r=16 | r=32 | ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| add_chain | 12 | 131k | from_diff | random_subspace | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.04 |
| add_chain | 12 | 131k | from_diff | supervised_centroid_svd | 0.95 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.04 |
| add_chain | 12 | 131k | from_diff | unsupervised_svd | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.96 | 0.04 |
| add_chain | 12 | 16k | from_diff | random_subspace | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 12 | 16k | from_diff | supervised_centroid_svd | 0.94 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.06 |
| add_chain | 12 | 16k | from_diff | unsupervised_svd | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.06 |
| add_chain | 12 | 262k | from_diff | random_subspace | 0.95 | 0.95 | 0.94 | 0.94 | 0.95 | 0.94 | 0.95 | 0.05 |
| add_chain | 12 | 262k | from_diff | supervised_centroid_svd | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.05 |
| add_chain | 12 | 262k | from_diff | unsupervised_svd | 0.95 | 0.95 | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.05 |
| add_chain | 12 | 32k | from_diff | random_subspace | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.05 |
| add_chain | 12 | 32k | from_diff | supervised_centroid_svd | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.05 |
| add_chain | 12 | 32k | from_diff | unsupervised_svd | 0.94 | 0.94 | 0.94 | 0.93 | 0.94 | 0.94 | 0.95 | 0.05 |
| add_chain | 12 | 65k | from_diff | random_subspace | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.07 |
| add_chain | 12 | 65k | from_diff | supervised_centroid_svd | 0.93 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.07 |
| add_chain | 12 | 65k | from_diff | unsupervised_svd | 0.93 | 0.93 | 0.93 | 0.93 | 0.94 | 0.94 | 0.94 | 0.07 |
| entity_track | 12 | 131k | from_diff | random_subspace | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.54 |
| entity_track | 12 | 131k | from_diff | supervised_centroid_svd | 0.77 | 0.86 | 0.86 | 0.86 | 0.92 | 0.92 | 0.93 | 0.54 |
| entity_track | 12 | 131k | from_diff | unsupervised_svd | 0.77 | 0.91 | 0.91 | 0.92 | 0.92 | 0.92 | 0.93 | 0.54 |
| entity_track | 12 | 16k | from_diff | random_subspace | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.62 |
| entity_track | 12 | 16k | from_diff | supervised_centroid_svd | 0.74 | 0.84 | 0.84 | 0.84 | 0.89 | 0.89 | 0.90 | 0.62 |
| entity_track | 12 | 16k | from_diff | unsupervised_svd | 0.74 | 0.88 | 0.88 | 0.88 | 0.88 | 0.89 | 0.91 | 0.62 |
| entity_track | 12 | 262k | from_diff | random_subspace | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.46 |
| entity_track | 12 | 262k | from_diff | supervised_centroid_svd | 0.85 | 0.90 | 0.90 | 0.89 | 0.93 | 0.94 | 0.95 | 0.46 |
| entity_track | 12 | 262k | from_diff | unsupervised_svd | 0.85 | 0.93 | 0.92 | 0.93 | 0.93 | 0.94 | 0.94 | 0.46 |
| entity_track | 12 | 32k | from_diff | random_subspace | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.52 |
| entity_track | 12 | 32k | from_diff | supervised_centroid_svd | 0.80 | 0.90 | 0.90 | 0.89 | 0.95 | 0.95 | 0.95 | 0.52 |
| entity_track | 12 | 32k | from_diff | unsupervised_svd | 0.80 | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.95 | 0.52 |
| entity_track | 12 | 65k | from_diff | random_subspace | 0.76 | 0.76 | 0.75 | 0.76 | 0.76 | 0.75 | 0.76 | 0.59 |
| entity_track | 12 | 65k | from_diff | supervised_centroid_svd | 0.76 | 0.85 | 0.85 | 0.85 | 0.92 | 0.94 | 0.94 | 0.59 |
| entity_track | 12 | 65k | from_diff | unsupervised_svd | 0.76 | 0.91 | 0.92 | 0.92 | 0.93 | 0.94 | 0.94 | 0.59 |
| ioi_control | 12 | 131k | from_diff | random_subspace | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.04 |
| ioi_control | 12 | 131k | from_diff | supervised_centroid_svd | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.04 |
| ioi_control | 12 | 131k | from_diff | unsupervised_svd | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.04 |
| ioi_control | 12 | 16k | from_diff | random_subspace | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.13 |
| ioi_control | 12 | 16k | from_diff | supervised_centroid_svd | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.96 | 0.95 | 0.13 |
| ioi_control | 12 | 16k | from_diff | unsupervised_svd | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.95 | 0.13 |
| ioi_control | 12 | 262k | from_diff | random_subspace | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.05 |
| ioi_control | 12 | 262k | from_diff | supervised_centroid_svd | 1.00 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.05 |
| ioi_control | 12 | 262k | from_diff | unsupervised_svd | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.05 |
| ioi_control | 12 | 32k | from_diff | random_subspace | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.09 |
| ioi_control | 12 | 32k | from_diff | supervised_centroid_svd | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | 0.97 | 0.96 | 0.09 |
| ioi_control | 12 | 32k | from_diff | unsupervised_svd | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | 0.96 | 0.09 |

## Failures (1)

- exp3 `ioi_control|12|65k|from_diff`: OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of 7.53 GiB of which 23.75 MiB is free. Including non-PyTorch memory, this process has 7.48 GiB memory in use. Of the allocated memory 7.24 GiB is allocated by PyTorch, and 44.77 MiB is reserved by PyTorch but unallocated.

## Figures

- figures/fig1_causal_split.png
- figures/fig2_error_manifolds.png
- figures/fig3_spectrum_and_probes.png
- figures/fig7_subspace_methods.png

# PRISM summary: `paper_residue`

device `cuda` dtype `torch.bfloat16` mock=False timings {'exp1': 370.8, 'exp2': 47.1, 'exp3': 2174.4} total Nones

## Tasks

| task | n_train | n_test | skipped | n_shots | format | filter | dL clean | dL corrupt | clean top1 |
|---|---|---|---|---|---|---|---|---|---|
| entity_track | 300 | 150 | False | 8 | auto | clean_only | 2.77 | -0.15 | 1.00 |
| add_chain | 300 | 150 | False | 8 | auto | strict | 3.48 | -3.02 | 1.00 |
| ioi_control | 300 | 150 | False | 2 | auto | strict | 6.75 | -6.39 | 1.00 |

## Exp1: causal split (NLDR, 95% bootstrap CI)

| task | layer | width | pos | rel err | clean_full | sae_only | error_only | sae_splice | noise_control |
|---|---|---|---|---|---|---|---|---|---|
| add_chain | 10 | 16k | from_diff | 0.17 | 1.00 [1.00,1.00] | 0.95 [0.94,0.96] | 0.06 [0.04,0.07] | 1.03 [1.01,1.04] | 0.06 [0.04,0.08] |
| add_chain | 12 | 16k | from_diff | 0.17 | 1.00 [1.00,1.00] | 0.94 [0.93,0.95] | 0.06 [0.05,0.07] | 1.02 [1.00,1.04] | 0.07 [0.05,0.10] |
| add_chain | 14 | 16k | from_diff | 0.14 | 1.00 [1.00,1.00] | 0.94 [0.93,0.96] | 0.06 [0.05,0.08] | 0.94 [0.92,0.96] | 0.16 [0.13,0.19] |
| add_chain | 18 | 16k | from_diff | 0.15 | 1.00 [1.00,1.00] | 0.87 [0.86,0.89] | 0.14 [0.12,0.17] | 0.81 [0.79,0.83] | 0.21 [0.18,0.24] |
| entity_track | 10 | 16k | from_diff | 0.16 | 1.00 [1.00,1.00] | 0.96 [0.95,0.97] | 0.30 [0.27,0.32] | 1.02 [1.01,1.04] | 0.45 [0.42,0.48] |
| entity_track | 12 | 16k | from_diff | 0.18 | 1.00 [1.00,1.00] | 0.74 [0.73,0.76] | 0.62 [0.59,0.65] | 1.09 [1.07,1.11] | 0.51 [0.48,0.55] |
| entity_track | 14 | 16k | from_diff | 0.16 | 1.00 [1.00,1.00] | 0.96 [0.94,0.98] | 0.41 [0.38,0.44] | 0.98 [0.96,1.01] | 0.28 [0.24,0.31] |
| entity_track | 18 | 16k | from_diff | 0.13 | 1.00 [1.00,1.00] | 0.86 [0.85,0.88] | 0.17 [0.15,0.19] | 1.06 [1.03,1.08] | 0.14 [0.11,0.17] |
| ioi_control | 10 | 16k | from_diff | 0.14 | 1.00 [1.00,1.00] | 0.96 [0.95,0.96] | 0.07 [0.06,0.07] | 0.99 [0.98,0.99] | 0.03 [0.02,0.04] |
| ioi_control | 12 | 16k | from_diff | 0.14 | 1.00 [1.00,1.00] | 0.96 [0.95,0.97] | 0.13 [0.11,0.14] | 0.97 [0.97,0.98] | 0.09 [0.07,0.10] |
| ioi_control | 14 | 16k | from_diff | 0.14 | 1.00 [1.00,1.00] | 0.90 [0.89,0.92] | 0.18 [0.16,0.20] | 0.84 [0.83,0.84] | 0.18 [0.16,0.19] |
| ioi_control | 18 | 16k | from_diff | 0.12 | 1.00 [1.00,1.00] | 0.76 [0.75,0.77] | 0.26 [0.25,0.28] | 0.82 [0.81,0.83] | 0.21 [0.20,0.23] |

## SAE fidelity on generic text (CE nats/token)

| layer | width | CE clean | CE spliced | CE mean-ablated | delta | CE recovered |
|---|---|---|---|---|---|---|
| 10 | 16k | 2.48 | 2.56 | 10.70 | 0.08 | 0.99 |
| 12 | 16k | 2.48 | 2.58 | 10.14 | 0.10 | 0.99 |
| 14 | 16k | 2.48 | 2.60 | 10.20 | 0.11 | 0.99 |
| 18 | 16k | 2.48 | 2.64 | 10.30 | 0.16 | 0.98 |

## Exp2: test probe accuracy (selectivity) / R^2 (circular for mod_sum)

`surface` = the same probe on the prompt's tokens only: eps must beat it for a state claim.

| task | layer | width | x | x_hat | eps | latents | surface | chance |
|---|---|---|---|---|---|---|---|---|
| add_chain | 10 | 16k | 0.52 (0.30) / 0.98 | 0.33 (0.10) / 0.65 | 0.47 (0.25) / 0.98 | 0.34 (0.09) / 0.62 | 0.45 (0.22) / 1.00 | 0.25 |
| add_chain | 12 | 16k | 0.57 (0.35) / 0.99 | 0.42 (0.21) / 0.81 | 0.53 (0.33) / 0.98 | 0.38 (0.13) / 0.78 | 0.45 (0.22) / 1.00 | 0.25 |
| add_chain | 14 | 16k | 0.69 (0.42) / 0.99 | 0.50 (0.22) / 0.88 | 0.57 (0.30) / 0.98 | 0.49 (0.24) / 0.87 | 0.45 (0.18) / 1.00 | 0.25 |
| add_chain | 18 | 16k | 0.92 (0.73) / 1.00 | 0.87 (0.67) / 0.96 | 0.91 (0.71) / 0.99 | 0.83 (0.60) / 0.94 | 0.45 (0.21) / 1.00 | 0.25 |
| entity_track | 10 | 16k | 0.78 (0.57) / 0.60 | 0.37 (0.19) / 0.32 | 0.67 (0.46) / 0.44 | 0.36 (0.18) / 0.29 | 0.33 (0.19) / 0.08 | 0.18 |
| entity_track | 12 | 16k | 0.91 (0.73) / 0.75 | 0.48 (0.32) / 0.44 | 0.84 (0.65) / 0.71 | 0.45 (0.27) / 0.39 | 0.33 (0.18) / 0.08 | 0.18 |
| entity_track | 14 | 16k | 0.97 (0.79) / 0.83 | 0.59 (0.41) / 0.59 | 0.94 (0.75) / 0.77 | 0.45 (0.29) / 0.49 | 0.33 (0.19) / 0.08 | 0.18 |
| entity_track | 18 | 16k | 0.99 (0.89) / 0.90 | 0.88 (0.80) / 0.77 | 0.99 (0.90) / 0.89 | 0.89 (0.80) / 0.75 | 0.33 (0.15) / 0.08 | 0.18 |
| ioi_control | 10 | 16k | 1.00 (0.94) / 0.76 | 0.15 (0.09) / 0.03 | 0.98 (0.92) / 0.51 | 0.16 (0.13) / 0.03 | 1.00 (0.93) / 1.00 | 0.11 |
| ioi_control | 12 | 16k | 1.00 (0.94) / 0.78 | 0.20 (0.16) / 0.03 | 0.97 (0.93) / 0.55 | 0.19 (0.13) / -0.00 | 1.00 (0.92) / 1.00 | 0.11 |
| ioi_control | 14 | 16k | 1.00 (0.93) / 0.88 | 0.52 (0.48) / 0.16 | 0.99 (0.92) / 0.76 | 0.41 (0.36) / 0.07 | 1.00 (0.94) / 1.00 | 0.11 |
| ioi_control | 18 | 16k | 1.00 (0.96) / 0.99 | 1.00 (0.95) / 0.98 | 1.00 (0.96) / 0.99 | 1.00 (0.91) / 0.96 | 1.00 (0.94) / 1.00 | 0.11 |

## Exp3: hybrid patch NLDR at each rank (method rows; reference = full error_only)

| task | layer | width | pos | method | r=0 | r=1 | r=2 | r=4 | r=8 | r=16 | r=32 | ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| add_chain | 12 | 16k | from_diff | generic_text_svd | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 12 | 16k | from_diff | random_subspace | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 12 | 16k | from_diff | supervised_centroid_svd | 0.94 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.06 |
| add_chain | 12 | 16k | from_diff | transfer_from_entity_track | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 12 | 16k | from_diff | transfer_from_ioi_control | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 12 | 16k | from_diff | unsupervised_svd | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.06 |
| add_chain | 14 | 16k | from_diff | generic_text_svd | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 | 0.06 |
| add_chain | 14 | 16k | from_diff | random_subspace | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 14 | 16k | from_diff | supervised_centroid_svd | 0.94 | 0.95 | 0.96 | 0.95 | 0.96 | 0.96 | 0.97 | 0.06 |
| add_chain | 14 | 16k | from_diff | transfer_from_entity_track | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.06 |
| add_chain | 14 | 16k | from_diff | transfer_from_ioi_control | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.06 |
| add_chain | 14 | 16k | from_diff | unsupervised_svd | 0.94 | 0.94 | 0.94 | 0.95 | 0.95 | 0.96 | 0.96 | 0.06 |
| add_chain | 18 | 16k | from_diff | generic_text_svd | 0.87 | 0.87 | 0.87 | 0.88 | 0.88 | 0.88 | 0.88 | 0.14 |
| add_chain | 18 | 16k | from_diff | random_subspace | 0.87 | 0.88 | 0.88 | 0.87 | 0.88 | 0.88 | 0.88 | 0.14 |
| add_chain | 18 | 16k | from_diff | supervised_centroid_svd | 0.87 | 0.93 | 0.95 | 0.97 | 0.97 | 0.98 | 0.98 | 0.14 |
| add_chain | 18 | 16k | from_diff | transfer_from_entity_track | 0.87 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.89 | 0.14 |
| add_chain | 18 | 16k | from_diff | transfer_from_ioi_control | 0.87 | 0.88 | 0.87 | 0.87 | 0.88 | 0.88 | 0.88 | 0.14 |
| add_chain | 18 | 16k | from_diff | unsupervised_svd | 0.87 | 0.90 | 0.91 | 0.91 | 0.97 | 0.97 | 0.98 | 0.14 |
| entity_track | 12 | 16k | from_diff | generic_text_svd | 0.74 | 0.74 | 0.74 | 0.75 | 0.75 | 0.74 | 0.75 | 0.62 |
| entity_track | 12 | 16k | from_diff | random_subspace | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.62 |
| entity_track | 12 | 16k | from_diff | supervised_centroid_svd | 0.74 | 0.84 | 0.84 | 0.84 | 0.89 | 0.89 | 0.90 | 0.62 |
| entity_track | 12 | 16k | from_diff | transfer_from_add_chain | 0.74 | 0.74 | 0.74 | 0.74 | 0.75 | 0.75 | 0.78 | 0.62 |
| entity_track | 12 | 16k | from_diff | transfer_from_ioi_control | 0.74 | 0.74 | 0.75 | 0.75 | 0.75 | 0.76 | 0.76 | 0.62 |
| entity_track | 12 | 16k | from_diff | unsupervised_svd | 0.74 | 0.88 | 0.88 | 0.88 | 0.88 | 0.89 | 0.91 | 0.62 |
| entity_track | 14 | 16k | from_diff | generic_text_svd | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.95 | 0.95 | 0.41 |
| entity_track | 14 | 16k | from_diff | random_subspace | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.41 |
| entity_track | 14 | 16k | from_diff | supervised_centroid_svd | 0.96 | 0.96 | 0.97 | 0.96 | 1.01 | 1.01 | 1.01 | 0.41 |
| entity_track | 14 | 16k | from_diff | transfer_from_add_chain | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.97 | 0.41 |
| entity_track | 14 | 16k | from_diff | transfer_from_ioi_control | 0.96 | 0.96 | 0.96 | 0.95 | 0.95 | 0.96 | 0.96 | 0.41 |
| entity_track | 14 | 16k | from_diff | unsupervised_svd | 0.96 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 0.41 |
| entity_track | 18 | 16k | from_diff | generic_text_svd | 0.86 | 0.86 | 0.86 | 0.87 | 0.86 | 0.86 | 0.85 | 0.17 |
| entity_track | 18 | 16k | from_diff | random_subspace | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.87 | 0.17 |
| entity_track | 18 | 16k | from_diff | supervised_centroid_svd | 0.86 | 0.90 | 0.91 | 0.95 | 0.94 | 0.95 | 0.96 | 0.17 |
| entity_track | 18 | 16k | from_diff | transfer_from_add_chain | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.87 | 0.17 |
| entity_track | 18 | 16k | from_diff | transfer_from_ioi_control | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.87 | 0.87 | 0.17 |
| entity_track | 18 | 16k | from_diff | unsupervised_svd | 0.86 | 0.86 | 0.89 | 0.90 | 0.93 | 0.95 | 0.96 | 0.17 |
| ioi_control | 12 | 16k | from_diff | generic_text_svd | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.13 |
| ioi_control | 12 | 16k | from_diff | random_subspace | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.13 |
| ioi_control | 12 | 16k | from_diff | supervised_centroid_svd | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.96 | 0.95 | 0.13 |
| ioi_control | 12 | 16k | from_diff | transfer_from_add_chain | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.13 |
| ioi_control | 12 | 16k | from_diff | transfer_from_entity_track | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.13 |
| ioi_control | 12 | 16k | from_diff | unsupervised_svd | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.95 | 0.13 |
| ioi_control | 14 | 16k | from_diff | generic_text_svd | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.18 |
| ioi_control | 14 | 16k | from_diff | random_subspace | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.18 |
| ioi_control | 14 | 16k | from_diff | supervised_centroid_svd | 0.90 | 0.90 | 0.90 | 0.91 | 0.91 | 0.91 | 0.91 | 0.18 |
| ioi_control | 14 | 16k | from_diff | transfer_from_add_chain | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.18 |
| ioi_control | 14 | 16k | from_diff | transfer_from_entity_track | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.91 | 0.18 |
| ioi_control | 14 | 16k | from_diff | unsupervised_svd | 0.90 | 0.90 | 0.90 | 0.90 | 0.91 | 0.91 | 0.92 | 0.18 |
| ioi_control | 18 | 16k | from_diff | generic_text_svd | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.26 |
| ioi_control | 18 | 16k | from_diff | random_subspace | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.26 |
| ioi_control | 18 | 16k | from_diff | supervised_centroid_svd | 0.76 | 0.77 | 0.78 | 0.80 | 0.82 | 0.87 | 0.89 | 0.26 |
| ioi_control | 18 | 16k | from_diff | transfer_from_add_chain | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.26 |
| ioi_control | 18 | 16k | from_diff | transfer_from_entity_track | 0.76 | 0.75 | 0.76 | 0.76 | 0.76 | 0.76 | 0.77 | 0.26 |
| ioi_control | 18 | 16k | from_diff | unsupervised_svd | 0.76 | 0.77 | 0.80 | 0.80 | 0.83 | 0.87 | 0.89 | 0.26 |

## Failures (1)

- exp3 `fit|10|16k`: LinAlgError: SVD did not converge

## Figures

- figures/fig1_causal_split.png
- figures/fig2_error_manifolds.png
- figures/fig3_spectrum_and_probes.png
- figures/fig7_subspace_methods.png

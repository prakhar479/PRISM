# Batch-2 atlas (exploratory, post hoc; scripts/make_atlas.py)

Capture = NLDR(feat_A_along_U) / NLDR(full_along_U), DAS rank 8, mean over independent seeds; differences < ~0.1 are ties (S10).

## P20
156 cells, 3 DAS seeds; capture range 0.37-1.07; largest seed sd 0.041; mean n_seeds 3.0

- entity_track: capture minimum 0.429 at layer 15 (depth 0.60); outer-layer median 0.899; dip +0.470 vs seed sd 0.023; in the middle third -> dip holds
- ioi_control: capture minimum 0.606 at layer 13 (depth 0.52); outer-layer median 0.900; dip +0.294 vs seed sd 0.018; in the middle third -> dip holds
- people_cities: capture minimum 0.496 at layer 15 (depth 0.60); outer-layer median 0.891; dip +0.395 vs seed sd 0.013; in the middle third -> dip holds
- H3 (P20): 3 of 3 tasks have a mid-depth dip -> GO

Width / sparsity effects inside each (task, layer) (P20): width 16k -0.000; l0v big +0.045; l0v small -0.045

## P28
108 cells, 3 DAS seeds; capture range 0.46-0.99; largest seed sd 0.028; mean n_seeds 3.0


Width / sparsity effects inside each (task, layer) (P28): width 16k -0.037; width 262k +0.034; width 65k +0.003; l0v big +0.048; l0v medium +0.009; l0v small -0.057

## P21: no complete run

## P22
156 cells, 2 DAS seeds; capture range 0.37-1.03; largest seed sd 0.274; mean n_seeds 1.3

- refusal/das1v2_kl: capture minimum 0.734 at layer 12 (depth 0.48); outer-layer median 0.945; dip +0.212 vs seed sd 0.061; in the middle third -> dip holds
- refusal/das4v2_kl: capture minimum 0.773 at layer 12 (depth 0.48); outer-layer median 0.943; dip +0.170 vs seed sd 0.000; in the middle third -> dip holds
- refusal/diffmean: capture minimum 0.558 at layer 10 (depth 0.40); outer-layer median 0.922; dip +0.365 vs seed sd 0.000; in the middle third -> dip holds
- H3 (P22): 3 of 3 tasks have a mid-depth dip -> GO

Width / sparsity effects inside each (task, layer) (P22): width 16k -0.000; l0v big +0.035; l0v small -0.035

## H4: which cheap metric predicts capture?

- **P20**: UNCLEAR (no metric reaches |rho| 0.7)
  - spearman_pooled: splice_over_noise -0.68, fvu -0.47, recovered_mean +0.47, ce_delta -0.45, l0 +0.09, eps_over_x +0.08
- **P20 per dictionary**: GO (splice_over_noise predicts capture better than FVU)
  - spearman_pooled: splice_over_noise -0.77, fvu -0.54, recovered_mean +0.54, ce_delta -0.50, l0 +0.12, eps_over_x +0.10
- **P22**: UNCLEAR (no metric reaches |rho| 0.7)
  - spearman_pooled: splice_over_noise -0.57, fvu -0.54, ce_delta -0.49, recovered_mean +0.49, eps_over_x -0.10, l0 +0.09
- **P22 per dictionary**: UNCLEAR (no metric reaches |rho| 0.7)
  - spearman_pooled: splice_over_noise -0.57, fvu -0.54, ce_delta -0.49, recovered_mean +0.49, eps_over_x -0.10, l0 +0.09
- **P28**: UNCLEAR (no metric reaches |rho| 0.7)
  - spearman_pooled: recovered_mean +0.34, fvu -0.32, ce_delta -0.31, l0 +0.23, splice_over_noise -0.14, eps_over_x +0.02
  - spearman_within_layer: eps_over_x -0.72, recovered_mean +0.70, ce_delta -0.70, fvu -0.67, l0 +0.49, splice_over_noise -0.25
- **P28 per dictionary**: GO (eps_over_x predicts capture better than FVU)
  - spearman_pooled: recovered_mean +0.37, fvu -0.35, ce_delta -0.34, l0 +0.28, splice_over_noise -0.16, eps_over_x +0.02
  - spearman_within_layer: eps_over_x -0.86, ce_delta -0.83, recovered_mean +0.82, fvu -0.79, l0 +0.58, splice_over_noise -0.27

## Independently trained dictionaries (P20 all-layers vs P28 subset, same layer / width / L0)

- 24 pairs: median |capture difference| 0.031, max 0.232 (the dictionary-training noise floor; compare any dictionary difference with it)


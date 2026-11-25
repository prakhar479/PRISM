# Breadth pilot verdicts (exploratory, post hoc; scripts/make_pilot.py)

## P0: NO-GO as is: adopt kl + held-out validity; report the sensitivity
run c35acd1113a3 (2df949e6c2)
- add_chain rank 4: capture spread over objective x seed = 0.126; max |code - subspace| = 0.427; share of fits with held-out IIA >= 0.5 = 1.00 -> NOT robust
- add_chain rank 8: capture spread over objective x seed = 0.111; max |code - subspace| = 0.316; share of fits with held-out IIA >= 0.5 = 1.00 -> NOT robust
- add_chain rank 16: capture spread over objective x seed = 0.076; max |code - subspace| = 0.189; share of fits with held-out IIA >= 0.5 = 1.00 -> NOT robust
- people_cities rank 4: capture spread over objective x seed = 0.060; max |code - subspace| = 1.785; share of fits with held-out IIA >= 0.5 = 0.00 -> NOT robust
- people_cities rank 8: capture spread over objective x seed = 0.080; max |code - subspace| = 0.916; share of fits with held-out IIA >= 0.5 = 0.33 -> NOT robust
- people_cities rank 16: capture spread over objective x seed = 0.118; max |code - subspace| = 0.162; share of fits with held-out IIA >= 0.5 = 1.00 -> NOT robust

## P11: GO (OT-v2 usable)
run 5d1b285d70e1 (2df949e6c2)
- keep rate (test, f_pilot): entity_track_L8_16k_l071__fit-ot_placed_dest+ot_bin_dest+ot_drawer_dest__eval-ot_placed_dest+ot_bin_dest+ot_drawer_dest: 0.7448, entity_track_L12_16k_l082__fit-ot_placed_dest+ot_bin_dest+ot_drawer_dest__eval-ot_placed_dest+ot_bin_dest+ot_drawer_dest: 0.7448, entity_track_L16_16k_l078__fit-ot_placed_dest+ot_bin_dest+ot_drawer_dest__eval-ot_placed_dest+ot_bin_dest+ot_drawer_dest: 0.7448
- L12 das4v2_kl: DAS full-patch flip 0.129 vs random p95 0.003
- L12 das8v2_kl: DAS full-patch flip 0.409 vs random p95 0.003
- L16 das4v2_kl: DAS full-patch flip 0.080 vs random p95 0.000
- L16 das8v2_kl: DAS full-patch flip 0.231 vs random p95 0.000
- L8 das4v2_kl: DAS full-patch flip 0.105 vs random p95 0.003
- L8 das8v2_kl: DAS full-patch flip 0.398 vs random p95 0.003

## P1: NO-GO / unclear (no layer-robust content-relational gap)
run 431a8b4f4cc2 (9cc390664e)
- L8 L8_16k_l071 das16v2_kl: content 0.818 vs pc_en_swap 0.817 (diff +0.001, CIs separate: False)
- L8 L8_16k_l071 das16v2_kl: content 0.818 vs pc_en_query 0.947 (diff -0.129, CIs separate: False)
- L8 L8_16k_l071 das4v2_kl: content 0.765 vs pc_en_swap 0.779 (diff -0.014, CIs separate: False)
- L8 L8_16k_l071 das4v2_kl: content 0.765 vs pc_en_query 0.834 (diff -0.069, CIs separate: False)
- L8 L8_16k_l071 das8v2_kl: content 0.801 vs pc_en_swap 0.792 (diff +0.009, CIs separate: False)
- L8 L8_16k_l071 das8v2_kl: content 0.801 vs pc_en_query 0.932 (diff -0.131, CIs separate: False)
- L8 L8_65k_l059 das16v2_kl: content 0.910 vs pc_en_swap 0.905 (diff +0.005, CIs separate: False)
- L8 L8_65k_l059 das16v2_kl: content 0.910 vs pc_en_query 0.939 (diff -0.028, CIs separate: False)
- L8 L8_65k_l059 das4v2_kl: content 0.881 vs pc_en_swap 0.872 (diff +0.009, CIs separate: False)
- L8 L8_65k_l059 das4v2_kl: content 0.881 vs pc_en_query 0.833 (diff +0.049, CIs separate: True)
- L8 L8_65k_l059 das8v2_kl: content 0.898 vs pc_en_swap 0.882 (diff +0.016, CIs separate: False)
- L8 L8_65k_l059 das8v2_kl: content 0.898 vs pc_en_query 0.933 (diff -0.034, CIs separate: False)
- L12 L12_16k_l082 das16v2_kl: content 0.724 vs pc_en_swap 0.686 (diff +0.038, CIs separate: True)
- L12 L12_16k_l082 das16v2_kl: content 0.724 vs pc_en_query 0.748 (diff -0.024, CIs separate: False)
- L12 L12_16k_l082 das4v2_kl: content 0.678 vs pc_en_swap 0.738 (diff -0.060, CIs separate: False)
- L12 L12_16k_l082 das4v2_kl: content 0.678 vs pc_en_query 0.657 (diff +0.021, CIs separate: False)
- L12 L12_16k_l082 das8v2_kl: content 0.721 vs pc_en_swap 0.724 (diff -0.002, CIs separate: False)
- L12 L12_16k_l082 das8v2_kl: content 0.721 vs pc_en_query 0.743 (diff -0.021, CIs separate: False)
- L12 L12_65k_l072 das16v2_kl: content 0.824 vs pc_en_swap 0.790 (diff +0.034, CIs separate: True)
- L12 L12_65k_l072 das16v2_kl: content 0.824 vs pc_en_query 0.830 (diff -0.006, CIs separate: False)
- L12 L12_65k_l072 das4v2_kl: content 0.774 vs pc_en_swap 0.811 (diff -0.037, CIs separate: False)
- L12 L12_65k_l072 das4v2_kl: content 0.774 vs pc_en_query 0.778 (diff -0.004, CIs separate: False)
- L12 L12_65k_l072 das8v2_kl: content 0.814 vs pc_en_swap 0.818 (diff -0.004, CIs separate: False)
- L12 L12_65k_l072 das8v2_kl: content 0.814 vs pc_en_query 0.819 (diff -0.005, CIs separate: False)
- L16 L16_16k_l078 das16v2_kl: content 0.649 vs pc_en_swap 0.636 (diff +0.013, CIs separate: False)
- L16 L16_16k_l078 das16v2_kl: content 0.649 vs pc_en_query 0.817 (diff -0.168, CIs separate: False)
- L16 L16_16k_l078 das4v2_kl: content 0.608 vs pc_en_swap 0.731 (diff -0.124, CIs separate: False)
- L16 L16_16k_l078 das4v2_kl: content 0.608 vs pc_en_query 0.683 (diff -0.075, CIs separate: False)
- L16 L16_16k_l078 das8v2_kl: content 0.653 vs pc_en_swap 0.675 (diff -0.022, CIs separate: False)
- L16 L16_16k_l078 das8v2_kl: content 0.653 vs pc_en_query 0.792 (diff -0.139, CIs separate: False)
- L16 L16_65k_l069 das16v2_kl: content 0.739 vs pc_en_swap 0.738 (diff +0.001, CIs separate: False)
- L16 L16_65k_l069 das16v2_kl: content 0.739 vs pc_en_query 0.806 (diff -0.067, CIs separate: False)
- L16 L16_65k_l069 das4v2_kl: content 0.714 vs pc_en_swap 0.808 (diff -0.094, CIs separate: False)
- L16 L16_65k_l069 das4v2_kl: content 0.714 vs pc_en_query 0.668 (diff +0.045, CIs separate: False)
- L16 L16_65k_l069 das8v2_kl: content 0.737 vs pc_en_swap 0.761 (diff -0.024, CIs separate: False)
- L16 L16_65k_l069 das8v2_kl: content 0.737 vs pc_en_query 0.774 (diff -0.037, CIs separate: False)

## P2: no complete run

## P4: no complete run

## P5: NO-GO / unclear
run 4d85e3b586c8 (c566c86485)
- L16 L16_16k_l078: capture magnitude 0.849 vs period-10 plane 0.697; magnitude carries 0.06 of the resampled-error effect -> no
- L16 L16_65k_l069: capture magnitude 0.780 vs period-10 plane 0.533; magnitude carries 0.01 of the resampled-error effect -> no
- L18 L18_16k_l074: capture magnitude 0.690 vs period-10 plane 0.715; magnitude carries 0.08 of the resampled-error effect -> no
- L18 L18_65k_l064: capture magnitude 0.761 vs period-10 plane 0.769; magnitude carries -0.04 of the resampled-error effect -> no
- L19 L19_16k_l073: capture magnitude 0.616 vs period-10 plane 0.659; magnitude carries 0.13 of the resampled-error effect -> no
- L19 L19_1m_l088: capture magnitude 0.773 vs period-10 plane 0.745; magnitude carries 0.09 of the resampled-error effect -> no
- L19 L19_65k_l063: capture magnitude 0.626 vs period-10 plane 0.721; magnitude carries 0.03 of the resampled-error effect -> no

## P7: GO (sensitivity beats variance)
run f0dd2ac58142 (c566c86485)
- add_chain L18 r=1: grad_err closes 0.482, pca_pooled 0.225 (repair above random p95: True) -> yes
- add_chain L18 r=2: grad_err closes 0.436, pca_pooled 0.225 (repair above random p95: True) -> no
- add_chain L18 r=4: grad_err closes 0.591, pca_pooled 0.282 (repair above random p95: True) -> yes
- add_chain L18 r=8: grad_err closes 0.718, pca_pooled 0.730 (repair above random p95: True) -> no
- entity_track L12 r=1: grad_err closes 0.475, pca_pooled -0.001 (repair above random p95: True) -> yes
- entity_track L12 r=2: grad_err closes 0.745, pca_pooled 0.015 (repair above random p95: True) -> yes
- entity_track L12 r=4: grad_err closes 1.400, pca_pooled 0.001 (repair above random p95: True) -> yes
- entity_track L12 r=8: grad_err closes 2.457, pca_pooled 0.005 (repair above random p95: True) -> yes
- people_cities L12 r=1: grad_err closes 0.208, pca_pooled -0.001 (repair above random p95: True) -> yes
- people_cities L12 r=2: grad_err closes 0.359, pca_pooled -0.003 (repair above random p95: True) -> yes
- people_cities L12 r=4: grad_err closes 0.628, pca_pooled 0.001 (repair above random p95: True) -> yes
- people_cities L12 r=8: grad_err closes 0.946, pca_pooled 0.007 (repair above random p95: True) -> yes

## P8: GO (second model usable)
run 1e285481c666 (2df949e6c2)
- tasks with keep >= 0.30: ['entity_track', 'people_cities']
- any positive feature-only gap: True

## P12: GO (dictionary family matters, replacement-model error nodes carry task state)
run a445d715ee36 (a8ebaab725)
- people_cities L13: median capture by family xc 0.192, clt+attn 0.582, resid 0.695, clt 0.717, resid_seed2 0.729, mlpsae_all 0.800, tc_all+attn 0.831, tc_all 0.864, tc_single 0.980
  spread 0.846 (L13_gs2_1bc_L13_xc65k_medium 0.135 .. L13_gs2_1bc_L13_tc16k_medium_single 0.981; CIs separate: True)
  seed floor (262k medium, two seeds): |delta capture| = 0.024
- people_cities L17: median capture by family xc 0.618, resid 0.768, tc_all 1.012, clt 1.018
  spread 0.416 (L17_gs2_1bc_L17_xc262k_medium 0.602 .. L17_gs2_1bc_L17_clt262k_medium 1.018; CIs separate: True)
- people_cities L13: replacement models: largest feature-only gap 0.298 (L13_gs2_1bc_L13_clt262k_medium+attn16k_small; error-only NLDR 0.283)
- people_cities L17: replacement models: largest feature-only gap 0.038 (L17_gs2_1bc_L17_clt262k_medium; error-only NLDR 0.031)

## P13: GO (refusal state partly in the SAE error)
run f2350346ccbf (6cf4cea778)
- usable test pairs (median keep rate over cells): 1.00
- L7 L7_gs2_1bit_L7_16k_medium: full_A 1.000, feature-only gap 0.096, error-only 0.505, n 150
- L7 L7_gs2_1bit_L7_262k_medium: full_A 1.000, feature-only gap 0.062, error-only 0.183, n 150
- L13 L13_gs2_1bit_L13_16k_medium: full_A 1.000, feature-only gap 0.082, error-only 0.129, n 150
- L13 L13_gs2_1bit_L13_262k_medium: full_A 1.000, feature-only gap 0.067, error-only 0.117, n 150
- L13 L13_gs2_1bitc_L13_clt262k_medium: full_A 1.000, feature-only gap 0.113, error-only 0.195, n 150
- L13 L13_gs2_1bitc_L13_tc16k_small_all: full_A 1.000, feature-only gap 0.036, error-only 0.115, n 150
- L13 L13_gs2_1bitc_L13_xc65k_medium: full_A 1.000, feature-only gap 0.147, error-only 0.303, n 150
- L17 L17_gs2_1bit_L17_16k_medium: full_A 1.000, feature-only gap 0.063, error-only 0.187, n 150
- L17 L17_gs2_1bit_L17_262k_medium: full_A 1.000, feature-only gap 0.085, error-only -0.020, n 150
- L22 L22_gs2_1bit_L22_16k_medium: full_A 1.000, feature-only gap 0.086, error-only 0.013, n 150
- L22 L22_gs2_1bit_L22_262k_medium: full_A 1.000, feature-only gap 0.051, error-only -0.083, n 150
  L7 L7_gs2_1bit_L7_16k_medium das1v2_kl r1: capture 0.994 [0.976, 1.012] (full NLDR 0.982, random p95 0.002)
  L7 L7_gs2_1bit_L7_16k_medium diffmean r1: capture 1.013 [0.983, 1.046] (full NLDR 0.337, random p95 0.002)
  L7 L7_gs2_1bit_L7_262k_medium das1v2_kl r1: capture 0.987 [0.973, 1.000] (full NLDR 0.982, random p95 0.002)
  L7 L7_gs2_1bit_L7_262k_medium diffmean r1: capture 0.957 [0.925, 0.986] (full NLDR 0.337, random p95 0.002)
  L13 L13_gs2_1bit_L13_16k_medium das1v2_kl r1: capture 0.902 [0.897, 0.908] (full NLDR 0.871, random p95 0.004)
  L13 L13_gs2_1bit_L13_16k_medium diffmean r1: capture 0.922 [0.911, 0.932] (full NLDR 0.682, random p95 0.004)
  L13 L13_gs2_1bit_L13_262k_medium das1v2_kl r1: capture 0.915 [0.910, 0.921] (full NLDR 0.871, random p95 0.004)
  L13 L13_gs2_1bit_L13_262k_medium diffmean r1: capture 0.916 [0.907, 0.924] (full NLDR 0.682, random p95 0.004)
  L13 L13_gs2_1bitc_L13_clt262k_medium das1v2_kl r1: capture 0.854 [0.844, 0.864] (full NLDR 0.871, random p95 0.004)
  L13 L13_gs2_1bitc_L13_clt262k_medium diffmean r1: capture 0.823 [0.801, 0.846] (full NLDR 0.682, random p95 0.004)
  L13 L13_gs2_1bitc_L13_tc16k_small_all das1v2_kl r1: capture 0.927 [0.921, 0.932] (full NLDR 0.871, random p95 0.004)
  L13 L13_gs2_1bitc_L13_tc16k_small_all diffmean r1: capture 0.941 [0.930, 0.951] (full NLDR 0.682, random p95 0.004)
  L13 L13_gs2_1bitc_L13_xc65k_medium das1v2_kl r1: capture 0.679 [0.670, 0.688] (full NLDR 0.871, random p95 0.004)
  L13 L13_gs2_1bitc_L13_xc65k_medium diffmean r1: capture 0.587 [0.568, 0.607] (full NLDR 0.682, random p95 0.004)
  L17 L17_gs2_1bit_L17_16k_medium das1v2_kl r1: capture 0.950 [0.947, 0.953] (full NLDR 0.823, random p95 0.016)
  L17 L17_gs2_1bit_L17_16k_medium diffmean r1: capture 0.947 [0.942, 0.952] (full NLDR 0.722, random p95 0.016)
  L17 L17_gs2_1bit_L17_262k_medium das1v2_kl r1: capture 0.975 [0.972, 0.977] (full NLDR 0.823, random p95 0.016)
  L17 L17_gs2_1bit_L17_262k_medium diffmean r1: capture 0.985 [0.982, 0.988] (full NLDR 0.722, random p95 0.016)
  L22 L22_gs2_1bit_L22_16k_medium das1v2_kl r1: capture 0.981 [0.980, 0.983] (full NLDR 1.086, random p95 0.028)
  L22 L22_gs2_1bit_L22_16k_medium diffmean r1: capture 0.927 [0.924, 0.931] (full NLDR 0.761, random p95 0.028)
  L22 L22_gs2_1bit_L22_262k_medium das1v2_kl r1: capture 0.995 [0.993, 0.997] (full NLDR 1.086, random p95 0.028)
  L22 L22_gs2_1bit_L22_262k_medium diffmean r1: capture 0.965 [0.961, 0.968] (full NLDR 0.761, random p95 0.028)

- trained klft_L18_16k: held-out CE delta 0.1712920236975197 -> 0.12586561621689216
- trained frozendec_L19_16k: held-out CE delta 7.275492350260417 -> 5.372371339875508
- trained frozendec_L12_16k: held-out CE delta 7.618050834841845 -> 4.925670915309007
- trained klft_L12_16k: held-out CE delta 0.1162407876611725 -> 0.08545016529114266

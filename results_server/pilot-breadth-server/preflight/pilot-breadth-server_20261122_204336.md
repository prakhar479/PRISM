# Preflight pilot-breadth-server 20261122_204336: OK

| # | status | check | detail |
|---|---|---|---|
| 1 | PASS | packages match requirements/harness.lock | 0 errors, 0 warnings |
| 1 | PASS | accelerator | NVIDIA GeForce RTX 2080 Ti, free 10.41 / 10.57 GB |
| 1 | PASS | RAM and disk | RAM available 122.4 GB; disk free over cache tiers 2067 GB (/ssd_scratch/raveeshvyas1410/prism/cache: 463, /scratch/raveeshvyas1410/prism/cache: 1604; need > 10); at runs root 49 GB (need > 2) |
| 2 | PASS | prefetch targets cached with matching sha256 | 123 files, 62.73 GB, 0 errors |
| 3 | PASS | model load | 27 s, 4.92 GB allocated |
| 3 | PASS | last-position head == full forward last logits | max /dlogit/ 0.121 (tol 0.25 = 2 bfloat16 ulp at max /logit/ 28.4); argmax equal True |
| 4 | PASS | hook identity (OT L12, 4 pairs): full_B == clean_B, resumed clean_B == stage 1 | max /dD/ full_B 0, resumed clean_B 0 |
| 4 | PASS | full_A reproduces clean A (NLDR = 1) | max /dD/ 0 |
| 5 | PASS | entity_track_L12: golden pair ids and token ids == harness prompt set | 16 pairs |
| 5 | PASS | entity_track_L12: span residuals vs golden (relative) | max relative row error 0.0314 (median 0.0153; tol 0.05); pilot's own batch-1 vs batched: max 0.0269 |
| 5 | PASS | entity_track_L12: SAE variant == the one the golden export loaded | harness layer_12/width_16k/average_l0_82/params.npz; golden layer_12/width_16k/average_l0_82/params.npz |
| 5 | PASS | entity_track_L12: SAE x_hat on golden residuals | max /dx_hat/ 0.000214 (GPU fp32 vs golden) |
| 5 | PASS | entity_track_L12: per-condition D vs golden | max /dD/ feat_A_errB 0.122, feat_B_errA 0.0868, feat_A_zero 0.0994, full_A 0.115, repair_pca_pooled_r1 0.104, repair_pca_pooled_r2 0.134 |
| 7 | PASS | throughput | forward tokens/s by rows {1: 1396.1, 2: 1548.5, 4: 1668.7, 24: 1778.5} -> throughput_pilot-breadth-server_20261122_204336.json |
| 6 | SKIP | HF top-1 agreement (informational) | {'add_chain/A': 1.0, 'add_chain/B': 1.0, 'entity_track/A': 1.0, 'entity_track/B': 0.917, 'ioi_control/A': 1.0, 'ioi_control/B': 1.0, 'var_binding/A': 1.0, 'var_binding/B': 1.0} (see pilot-breadth-server_20261122_204336_hf/report.md) |
| 9 | PASS | GPU 0 | budget 9216 tokens x 24 rows, peak 5.75 GB, 1779.8 tok/s |
| 9 | PASS | GPU 1 | budget 9216 tokens x 24 rows, peak 5.75 GB, 1634.1 tok/s |
| 9 | PASS | GPU 2 | budget 9216 tokens x 24 rows, peak 5.75 GB, 1748.2 tok/s |
| 9 | PASS | GPU 3 | budget 9216 tokens x 24 rows, peak 5.75 GB, 1621.7 tok/s |
| 3 | PASS | per-GPU checks: at least one GPU usable | 4/4 GPUs passed -> runs/_queue/pilot-breadth-server/gpus.json |
| 8 | PASS | smoke E1_split | ok |
| 8 | PASS | smoke E2 | ok |

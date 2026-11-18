# Release 2: EXPLORATORY analyses (not preregistered; no multiplicity correction)

## E. E4: gap vs width and realized L0, OT L12, filter f_pilot (gap = M(full_A) - M(feat_A_errB))
           sae  width_k  l0_test  l0_generic  generic_err_energy  test_rel_err_norm  gap_nldr  gap_nldr_lo  gap_nldr_hi  gap_prob  gap_flip  meanPos_gap_nldr
  L12_16k_l082       16   78.211      84.455               0.156              0.425     0.252        0.237        0.267     0.210     0.000             0.060
  L12_32k_l076       32   75.395      79.221               0.143              0.416     0.197        0.181        0.211     0.230     0.000             0.052
 L12_32k_l0155       32  146.073     159.636               0.115              0.368     0.177        0.164        0.191     0.244     0.000             0.051
  L12_65k_l072       65   76.071      75.613               0.133              0.407     0.241        0.224        0.257     0.322     0.000             0.046
 L12_65k_l0141       65  134.406     142.286               0.109              0.367     0.134        0.122        0.147     0.168     0.000             0.031
 L12_131k_l067      131   69.973      69.797               0.126              0.401     0.186        0.169        0.201     0.170     0.000             0.031
L12_131k_l0129      131  130.849     134.585               0.103              0.358     0.232        0.217        0.248     0.326     0.011             0.069
 L12_262k_l067      262   67.054      67.891               0.120              0.411     0.147        0.130        0.164     0.103     0.011             0.022
L12_262k_l0121      262  124.403     128.232               0.098              0.362     0.145        0.133        0.158     0.150     0.000             0.036
 L12_524k_l065      524   67.537      68.526               0.114              0.406     0.190        0.173        0.209     0.160     0.000             0.039
L12_524k_l0115      524  114.401     116.175               0.096              0.367     0.177        0.163        0.189     0.208     0.011             0.051

OLS gap_nldr ~ 1 + log2(width) + log2(L0_test): intercept 0.440, per doubling width -0.0105, per doubling L0 -0.0274, R^2 0.28 (n=11; descriptive)
OLS gap_prob ~ 1 + log2(width) + log2(L0_test): intercept 0.071, per doubling width -0.0153, per doubling L0 +0.0366, R^2 0.21 (n=11; descriptive)
corr(gap_nldr, generic_err_energy) = 0.57; corr(gap_nldr, test_rel_err_norm) = 0.41


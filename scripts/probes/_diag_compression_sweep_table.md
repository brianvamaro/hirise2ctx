# Compression-fix sweep — 20260529T211211Z

Composite metric tables from the v2-dev within-image sweep (20 folds).

## Per-bin mean prediction (linear scale)

                    variant  S   zero  0_to_1e-4  1e-4_to_1e-3  1e-3_to_1e-2  1e-2_to_max  spearman    AUC
         lightgbm_two_stage 32 0.0028     0.0016        0.0030        0.0101       0.0247    0.1867 0.5500
         lightgbm_two_stage 64 0.0024     0.0019        0.0024        0.0091       0.0259    0.2632 0.5377
lightgbm_two_stage_balanced 32 0.0029     0.0017        0.0032        0.0105       0.0257    0.2083 0.5514
lightgbm_two_stage_balanced 64 0.0026     0.0020        0.0026        0.0094       0.0260    0.2804 0.5556
lightgbm_two_stage_weighted 32 0.0057     0.0049        0.0069        0.0179       0.0325    0.0796 0.5119
lightgbm_two_stage_weighted 64 0.0048     0.0043        0.0054        0.0168       0.0316    0.1599 0.4730
   lightgbm_two_stage_gamma 32 0.0026     0.0015        0.0029        0.0096       0.0241    0.1851 0.5518
   lightgbm_two_stage_gamma 64 0.0023     0.0018        0.0022        0.0085       0.0255    0.2550 0.5126
lightgbm_two_stage_combined 32 0.0062     0.0049        0.0072        0.0187       0.0328    0.0980 0.5010
lightgbm_two_stage_combined 64 0.0055     0.0046        0.0056        0.0159       0.0309    0.1602 0.4400

## Per-bin mean_pred / mean_true ratio

                    variant  S  zero  0_to_1e-4  1e-4_to_1e-3  1e-3_to_1e-2  1e-2_to_max
         lightgbm_two_stage 32   NaN     21.611         6.405         2.455        0.769
         lightgbm_two_stage 64   NaN     27.516         5.951         2.252        0.830
lightgbm_two_stage_balanced 32   NaN     22.923         6.717         2.552        0.798
lightgbm_two_stage_balanced 64   NaN     28.754         6.274         2.333        0.833
lightgbm_two_stage_weighted 32   NaN     66.818        14.435         4.351        1.011
lightgbm_two_stage_weighted 64   NaN     61.183        13.180         4.179        1.011
   lightgbm_two_stage_gamma 32   NaN     21.073         6.028         2.347        0.749
   lightgbm_two_stage_gamma 64   NaN     26.238         5.322         2.126        0.817
lightgbm_two_stage_combined 32   NaN     66.865        15.237         4.560        1.021
lightgbm_two_stage_combined 64   NaN     65.728        13.632         3.949        0.991

## Decision table

                    variant  S  Spearman    AUC  compression_score  zero_pred  high_pred  high_ratio
         lightgbm_two_stage 32    0.1867 0.5500             0.6613     0.0028     0.0247      0.7692
         lightgbm_two_stage 64    0.2632 0.5377             0.6619     0.0024     0.0259      0.8303
lightgbm_two_stage_balanced 32    0.2083 0.5514             0.6731     0.0029     0.0257      0.7980
lightgbm_two_stage_balanced 64    0.2804 0.5556             0.6759     0.0026     0.0260      0.8330
lightgbm_two_stage_weighted 32    0.0796 0.5119             0.9069     0.0057     0.0325      1.0105
lightgbm_two_stage_weighted 64    0.1599 0.4730             0.8831     0.0048     0.0316      1.0113
   lightgbm_two_stage_gamma 32    0.1851 0.5518             0.6500     0.0026     0.0241      0.7486
   lightgbm_two_stage_gamma 64    0.2550 0.5126             0.6401     0.0023     0.0255      0.8168
lightgbm_two_stage_combined 32    0.0980 0.5010             0.9190     0.0062     0.0328      1.0205
lightgbm_two_stage_combined 64    0.1602 0.4400             0.8881     0.0055     0.0309      0.9914

# W1 Rung 1a — label-shift rescore test

Recipe: two_stage_balanced × boulder_count @ S=64 (`models\lightgbm_two_stage_balanced\8c7523615964f5cb\scale_S64_target_boulder_count`); meaningful threshold bc > 50 (strict, matching evaluate.py); offsets di,dj ∈ [-2,+2] (25 cells; 1 tile = 320 m).

Question: does any anti-signal image's per-image AUC recover when its
label grid is shifted? Recovery at a nonzero offset = geometric
misalignment (rung 1 cause), not absent signal. Healthy images give the
null for max-over-25-offsets inflation.

```
                 auc_center  auc_best  best_di  best_dj  gain  n_overlap_best  anti_signal  recovers_to_gt_0.5  recovers_to_gt_0.6
obs_id                                                                                                                            
ESP_055978_2270       0.303     0.499        2       -2 0.196            1135         True               False               False
ESP_076499_1160       0.367     0.445       -2       -1 0.078             983         True               False               False
ESP_047976_2020       0.378     0.560        1       -1 0.181             698         True                True               False
ESP_046328_2180       0.400     0.519       -1       -2 0.119             375         True                True               False
ESP_071699_2260       0.403     0.438        1        0 0.035             598         True               False               False
ESP_054000_2255       0.430     0.468        1       -2 0.038             679         True               False               False
ESP_055017_2055       0.434     0.554        2        2 0.120             606         True                True               False
ESP_049242_2115       0.457     0.507        0        2 0.050             497         True                True               False
ESP_054622_2240       0.461     0.848        1        2 0.387            1049         True                True                True
ESP_064510_2260       0.475     0.591       -1       -2 0.116             962         True                True               False
ESP_055253_2245       0.492     0.519        2        2 0.027             524         True                True               False
ESP_045983_2270       0.513     0.556       -2        2 0.044             487        False               False               False
ESP_054397_2105       0.516     0.544       -1        0 0.029            1423        False               False               False
ESP_059421_2170       0.523     0.584        2        2 0.061             935        False               False               False
ESP_071093_2210       0.525     0.576        0       -1 0.050             745        False               False               False
ESP_069669_2220       0.531     0.573        1        0 0.042            1028        False               False               False
ESP_076565_2215       0.567     0.654        2        1 0.088             517        False               False               False
ESP_048688_2085       0.578     0.721        2        2 0.143             244        False               False               False
ESP_046959_2225       0.592     0.613        2       -2 0.021             860        False               False               False
ESP_051943_2270       0.596     0.596        0        0 0.000             632        False               False               False
ESP_045390_2215       0.620     0.660        1        0 0.040             571        False               False               False
ESP_063429_2240       0.627     0.661       -1       -1 0.034             373        False               False               False
ESP_045878_2235       0.637     0.719        2        0 0.083            1676        False               False               False
ESP_052576_2250       0.653     0.708        1        0 0.055            1025        False               False               False
ESP_068402_2240       0.654     0.663        0        1 0.008            1364        False               False               False
ESP_059686_2235       0.657     0.738       -2        0 0.081             954        False               False               False
ESP_066634_2210       0.675     0.750        1        0 0.076             842        False               False               False
ESP_054134_2265       0.687     0.687        0        0 0.000             798        False               False               False
ESP_076723_2265       0.691     0.769        1        0 0.078            1369        False               False               False
ESP_017355_2260       0.720     0.741        1        0 0.022            2562        False               False               False
ESP_055055_2255       0.721     0.738       -1        2 0.016             912        False               False               False
ESP_055690_2200       0.735     0.741        0       -1 0.006            1123        False               False               False
ESP_045139_2270       0.737     0.737        2        0 0.000            1099        False               False               False
ESP_068483_2280       0.801     0.842        2       -1 0.041            1095        False               False               False
ESP_053989_2260       0.813     0.828        1        0 0.015             625        False               False               False
ESP_042964_2160       0.843     0.876        1        0 0.033             581        False               False               False
ESP_045550_2180       0.945     0.973        2        2 0.027             611        False               False               False
ESP_069763_2235       0.979     0.979        0        0 0.000            1015        False               False               False
```

- Healthy-image (n=27) best-offset gain: median 0.034, max 0.143
- Anti-signal (n=11) best-offset gain: median 0.116, max 0.387
- Anti-signal recovering past 0.5: 7/11; past 0.6: 1/11

Note: best-offset AUC is selected post hoc over 25 cells, so small gains
are expected by chance — judge anti-signal gains against the healthy
null above, and treat only recoveries well past it as geometry evidence.
# W1 Rung 1a — label-shift rescore test

Recipe: two_stage_balanced × boulder_count @ S=64 (`models\lightgbm_two_stage_balanced\8c7523615964f5cb\scale_S64_target_boulder_count`); meaningful threshold bc > 50 (strict, matching evaluate.py); offsets di,dj ∈ [-2,+2] (25 cells; 1 tile = 320 m).

Question: does any anti-signal image's per-image AUC recover when its
label grid is shifted? Recovery at a nonzero offset = geometric
misalignment (rung 1 cause), not absent signal. Healthy images give the
null for max-over-25-offsets inflation.

```
                 auc_center  auc_best  best_di  best_dj  gain  n_overlap_best  anti_signal  recovers_to_gt_0.5  recovers_to_gt_0.6
obs_id                                                                                                                            
ESP_076499_1160       0.224     0.366       -2        1 0.142             956         True               False               False
ESP_055978_2270       0.245     0.368        2       -2 0.123            1135         True               False               False
ESP_054000_2255       0.394     0.458        1       -2 0.064             679         True               False               False
ESP_046328_2180       0.396     0.550       -2       -2 0.154             359         True                True               False
ESP_064510_2260       0.404     0.519       -2       -2 0.114             948         True                True               False
ESP_047976_2020       0.421     0.564        0       -1 0.143             716         True                True               False
ESP_049242_2115       0.470     0.491       -1        2 0.022             487         True               False               False
ESP_059686_2235       0.487     0.758       -2        2 0.271             872         True                True                True
ESP_045983_2270       0.503     0.582       -1        2 0.079             513        False               False               False
ESP_054622_2240       0.515     0.915        0       -2 0.400            1059        False               False               False
ESP_054397_2105       0.525     0.525        0        0 0.000            1459        False               False               False
ESP_071699_2260       0.566     0.603        2        2 0.037             520        False               False               False
ESP_055253_2245       0.571     0.606       -2       -2 0.035             524        False               False               False
ESP_048688_2085       0.574     0.770       -1        2 0.196             261        False               False               False
ESP_055017_2055       0.579     0.587       -2        2 0.007             590        False               False               False
ESP_046959_2225       0.580     0.585        2       -2 0.006             860        False               False               False
ESP_059421_2170       0.583     0.583        0        0 0.000            1084        False               False               False
ESP_045390_2215       0.595     0.639       -2        2 0.044             498        False               False               False
ESP_069669_2220       0.600     0.600        0        0 0.000            1062        False               False               False
ESP_051943_2270       0.607     0.629       -1        0 0.023             585        False               False               False
ESP_054134_2265       0.654     0.677        0       -1 0.023             694        False               False               False
ESP_063429_2240       0.671     0.707       -2        2 0.036             330        False               False               False
ESP_068402_2240       0.684     0.684        0        0 0.000            1433        False               False               False
ESP_076565_2215       0.685     0.685        0        0 0.000             581        False               False               False
ESP_045878_2235       0.696     0.696        0        0 0.000            1766        False               False               False
ESP_055055_2255       0.734     0.751        2       -1 0.018             923        False               False               False
ESP_017355_2260       0.735     0.735        0        0 0.000            2927        False               False               False
ESP_071093_2210       0.737     0.737        0        0 0.000             792        False               False               False
ESP_052576_2250       0.738     0.738        0        0 0.000            1104        False               False               False
ESP_066634_2210       0.744     0.744        0        0 0.000             914        False               False               False
ESP_055690_2200       0.762     0.774        0       -2 0.012            1067        False               False               False
ESP_045139_2270       0.795     0.795        0        0 0.000            1342        False               False               False
ESP_053989_2260       0.805     0.820       -1        0 0.014             625        False               False               False
ESP_076723_2265       0.808     0.808        0        0 0.000            1461        False               False               False
ESP_068483_2280       0.823     0.848        1        0 0.025            1167        False               False               False
ESP_042964_2160       0.890     0.890        0        0 0.000             608        False               False               False
ESP_045550_2180       0.931     0.967        1        2 0.036             625        False               False               False
ESP_069763_2235       0.991     0.991        0        0 0.000            1015        False               False               False
```

- Healthy-image (n=30) best-offset gain: median 0.007, max 0.400
- Anti-signal (n=8) best-offset gain: median 0.132, max 0.271
- Anti-signal recovering past 0.5: 4/8; past 0.6: 1/8

Note: best-offset AUC is selected post hoc over 25 cells, so small gains
are expected by chance — judge anti-signal gains against the healthy
null above, and treat only recoveries well past it as geometry evidence.
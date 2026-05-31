# Stage 6b H3 mechanism check -- full-v2 LOIO

Sweep: `models\_sweep_stage6b\20260531T020308Z`  | Dataset: `dataset_v2`  | Scale: S=64

## Acceptance summary

- PR-AUC mean: baseline 0.5431 -> +Stage 6b 0.5601  (delta +0.0170; pass = +>= 0.03)
- Spearman mean: baseline +0.1431 -> +Stage 6b +0.1507  (delta +0.0076)
- mean_ctx_incidence vs per-image presence_auc: rho = -0.103 (p=0.618, n=26)  -- H3 prediction: rho < -0.30
- mean_ctx_incidence vs per-image pr_auc: rho = +0.050 (p=0.765, n=38)  -- H3 prediction: rho < -0.30
- mean_ctx_incidence vs per-image spearman_rho: rho = -0.213 (p=0.199, n=38)  -- H3 prediction: rho < -0.30

## H3 correlation table

Spearman rho of per-image feature vs per-image **baseline** metric across 38 images. ** marks p < 0.05.

```
metric                     normalised_lift_meaningful  pr_auc  precision_at_top_5pct  presence_auc  spearman_rho
feature                                                                                                         
dominant_source_frac_mean                      +0.376  +0.361                 +0.393        +0.321        +0.394
mean_ctx_incidence                             +0.025  +0.050                 +0.042        -0.103        -0.213
mean_n_sources                                 -0.342  -0.326                 -0.357        -0.279        -0.405
std_ctx_incidence                              -0.400  -0.370                 -0.361        -0.340        -0.342
```

## Per-image deltas (top winners)

```
          ObsId  mean_ctx_incidence  mean_n_sources  delta_pr_auc  delta_spearman_rho  delta_presence_auc  delta_precision_at_top_5pct
ESP_071699_2260             +42.989          +1.084        +0.292              +0.398              -0.175                       +0.647
ESP_064510_2260             +43.033          +1.053        +0.207              +0.246              +0.031                       +0.556
ESP_076499_1160             +59.167          +1.026        +0.176              +0.497              +0.194                       +0.531
ESP_046959_2225             +56.724          +1.029        +0.116              +0.029              -0.070                       +0.320
ESP_055055_2255             +44.280          +1.000        +0.107              +0.079              -0.018                       +0.250
ESP_051943_2270             +51.884          +1.093        +0.106              +0.086              +0.089                       +0.062
ESP_054000_2255             +57.670          +1.111        +0.055              +0.204              +0.067                       +0.244
ESP_052576_2250             +51.548          +1.000        +0.047              +0.145              +0.048                       -0.218
ESP_076723_2265             +42.658          +1.000        +0.040              +0.112                 NaN                       -0.014
ESP_071093_2210             +54.829          +1.000        +0.033              +0.386              -0.008                       +0.000
```

## Per-image deltas (largest regressions)

```
          ObsId  mean_ctx_incidence  mean_n_sources  delta_pr_auc  delta_spearman_rho  delta_presence_auc  delta_precision_at_top_5pct
ESP_045878_2235             +52.908          +1.000        -0.011              -0.191              -0.063                       +0.023
ESP_068483_2280              +4.276          +1.000        -0.020              -0.303                 NaN                       -0.016
ESP_049242_2115             +46.018          +1.049        -0.038              -0.130              -0.014                       -0.069
ESP_042964_2160             +57.681          +1.081        -0.039              -0.064              -0.037                       -0.100
ESP_045983_2270             +62.476          +1.085        -0.040              -0.040              +0.101                       -0.379
ESP_066634_2210             +54.829          +1.000        -0.047              -0.138              -0.072                       +0.000
ESP_068402_2240             +46.951          +1.075        -0.059              +0.095              +0.060                       -0.319
ESP_059421_2170             +40.481          +1.138        -0.084              -0.156              -0.053                       -0.185
ESP_055690_2200             +48.404          +1.074        -0.104              -0.780              -0.336                       -0.203
ESP_017355_2260             +52.441          +1.003        -0.108              -0.218              -0.138                       -0.308
```


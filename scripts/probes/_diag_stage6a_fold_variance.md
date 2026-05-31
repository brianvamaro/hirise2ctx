# Stage 6a per-fold variance probe

Source: `models\_sweep_stage6a\20260530T213424Z\summary.parquet`
n_folds = 20; schemes = ['within_image_4fold', 'within_image_4fold_nbr']

## Per-scheme aggregates

```
                       spearman_rho                         presence_auc                         pr_auc                         normalised_lift_meaningful                      precision_at_top_5pct                   recall_at_top_5pct                     
                               mean     std     min     max         mean     std     min    max    mean     std     min     max                       mean     std  min     max                  mean     std  min  max               mean     std  min     max
scheme                                                                                                                                                                                                                                                         
within_image_4fold           0.2826  0.2440 -0.2447  0.7246       0.5642  0.0936  0.3728  0.689  0.6396  0.3992  0.0117  0.9997                     0.6188  0.3942  0.0  0.9934                0.6600  0.4204  0.0  1.0             0.0577  0.0391  0.0  0.1667
within_image_4fold_nbr       0.2760  0.2796 -0.1503  0.7868       0.5433  0.1801  0.1661  0.826  0.6494  0.3831  0.0107  0.9999                     0.6256  0.3729  0.0  0.9967                0.7036  0.3790  0.0  1.0             0.0782  0.0584  0.0  0.2143
```

## Per-fold deltas (nbr - base)

| metric | mean delta | std delta | wins (nbr>base) | losses | ties | 
|--------|-----------:|----------:|----------------:|-------:|-----:|
| spearman_rho                        | -0.0065 | 0.2234 |  12 |   8 |   0 |
| presence_auc                        | -0.0210 | 0.1449 |   7 |   5 |   0 |
| pr_auc                              | +0.0098 | 0.0988 |  11 |   6 |   0 |
| normalised_lift_meaningful          | +0.0068 | 0.0940 |   7 |   4 |   6 |
| precision_at_top_5pct               | +0.0436 | 0.1906 |   3 |   1 |  13 |
| recall_at_top_5pct                  | +0.0204 | 0.0662 |   3 |   1 |  13 |

## Per-held-out-image deltas (Spearman + precision@top-5%)

| obs_id | rho_base | rho_nbr | rho_delta | prec5_base | prec5_nbr | prec5_delta |
|--------|---------:|--------:|----------:|-----------:|----------:|------------:|
| ESP_055978_2270 | +0.1681 | +0.1709 | +0.0028 | 0.1538 | 0.1538 | +0.0000 |
| ESP_064510_2260 | +0.2794 | -0.0437 | -0.3231 | 0.5342 | 0.4509 | -0.0833 |
| ESP_068483_2280 | +0.5408 | +0.5367 | -0.0040 | 1.0000 | 1.0000 | +0.0000 |
| ESP_069669_2220 | +0.0008 | +0.1450 | +0.1442 | 0.2325 | 0.5010 | +0.2685 |
| ESP_071093_2210 | +0.4238 | +0.5712 | +0.1474 | 1.0000 | 1.0000 | +0.0000 |
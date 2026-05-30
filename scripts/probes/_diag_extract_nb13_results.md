# Notebook 13 key outputs


## lbl-extract

```
LBL augmentation: 38/38 images have IncidenceAngle

Distribution of illumination angles:
       IncidenceAngle  EmissionAngle  SubSolarAzimuth  PhaseAngle
count           38.00          38.00            38.00       38.00
mean            53.24           6.71           155.50       52.12
std              8.96           7.72            17.94       13.17
min             40.24           0.09           124.43       21.14
25%             45.76           0.58           143.29       43.03
50%             52.12           4.86           157.63       51.80
75%             59.04           8.30           165.26       60.73
max             72.44          28.94           219.92       79.07

Augmented df: 38 rows, 19 cols

```

## ctx-seam-extract

```
CTX source attribution: 38/38 images mapped

Distribution of n_ctx_sources per HiRISE footprint:
count    38.0
mean     24.1
std      11.5
min       4.0
25%      13.5
50%      25.5
75%      32.0
max      46.0

First 3 examples (dominant CTX sources per HiRISE footprint):

```

## corr-table

```
Spearman ρ (cells with p < 0.05 marked ** in the table below):


Significant correlations (p < 0.05):

```

## anti-shadow

```
features for ESP_054000_2255 S=64: 812 tiles, 58 columns
shadow_fraction columns: ['shadow_fraction', 'shadow_fraction_strict', 'lacunarity_shadow_b2', 'lacunarity_shadow_b4']

shadow_fraction stats by truth bin (boulder-rich vs not):

```

## anti-topk

```
Image: ESP_054000_2255, n=812 tiles, base_rate(fa>1e-2) = 0.183

Top-10% predicted (81 tiles):
  mean truth fractional_area: 0.0025
  fraction boulder-rich: 0.049 (vs base rate 0.183)
  ==> precision@top-10%: 0.049

Top-1% predicted (8 tiles):
  mean truth fractional_area: 0.0010
  fraction boulder-rich: 0.000

```
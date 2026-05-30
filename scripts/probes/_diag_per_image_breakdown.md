# Per-image breakdown — which v2 images worked, which didn't

Source data: full-v2 regression sweep `models/_sweep/20260529T061553Z/` and binary sweep 
`models/_sweep_binary/20260529T075754Z/` at fa_gt_1e-2 S=64.  Manifest: hirise_40_vclaire.csv.

Total images joined: **38**

## Top 10 by boulder-rich lift@top-K

          ObsId BoulderLabel  CenterLat  IncidenceAngle  NPolygons  bin_rich_base_rate  bin_rich_auc  bin_rich_lift  reg_spearman  reg_presence_auc
ESP_055978_2270 Boulder rich     46.586             NaN       9628               0.013         0.759          9.066         0.164             0.532
ESP_042964_2160 Boulder rich     35.816             NaN      34237               0.082         0.911          5.350         0.670             0.813
ESP_055055_2255 Boulder rich     44.996             NaN      22334               0.035         0.637          3.809         0.254             0.622
ESP_059421_2170 Boulder rich     36.743             NaN      20060               0.030         0.575          3.176         0.042             0.548
ESP_045878_2235 Boulder rich     43.297             NaN      21483               0.025         0.607          1.824         0.087             0.524
ESP_046959_2225 Boulder rich     42.137             NaN      73931               0.249         0.599          1.548         0.205             0.602
ESP_052576_2250 Boulder rich     44.635             NaN      79362               0.251         0.641          1.496         0.362             0.654
ESP_064510_2260 Boulder rich     45.429             NaN      80918               0.288         0.661          1.486         0.010             0.508
ESP_066634_2210 Boulder rich     40.535             NaN      78438               0.365         0.653          1.450         0.372               NaN
ESP_051943_2270 Boulder rich     46.840             NaN      14744               0.033         0.522          1.433         0.065             0.510

## Bottom 10 by boulder-rich lift@top-K

          ObsId BoulderLabel  CenterLat  IncidenceAngle  NPolygons  bin_rich_base_rate  bin_rich_auc  bin_rich_lift  reg_spearman  reg_presence_auc
ESP_045390_2215 Boulder rich     42.078             NaN      20456               0.095         0.589          0.926         0.375             0.734
ESP_045983_2270 Boulder rich     46.881             NaN      99740               0.114         0.511          0.785        -0.013             0.761
ESP_055253_2245 Boulder rich     44.208             NaN      47901               0.049         0.419          0.686         0.050               NaN
ESP_049242_2115 Boulder rich     31.269             NaN      41626               0.136         0.486          0.658        -0.047             0.974
ESP_054000_2255 Boulder rich     45.345             NaN      36686               0.183         0.398          0.293        -0.253             0.288
ESP_048688_2085 Boulder rich     28.411             NaN      42235               0.016         0.629          0.000         0.033               NaN
ESP_055017_2055 Boulder rich     25.318             NaN      16730               0.011         0.672          0.000         0.002               NaN
ESP_054397_2105 Boulder rich     30.426             NaN      18632               0.006         0.504          0.000        -0.123             0.418
ESP_055690_2200 Boulder rich     39.741             NaN      26524               0.010         0.566          0.000         0.556             0.784
ESP_069669_2220 Boulder rich     41.816             NaN      35238               0.007         0.499          0.000        -0.026             0.534

## Spearman correlations: per-image features vs performance

```
metric              bin_rich_auc  bin_rich_lift  reg_presence_auc  reg_spearman
feature                                                                        
CenterLat                 +0.208         +0.237            +0.192        +0.280
EmissionAngle                NaN            NaN               NaN           NaN
IncidenceAngle               NaN            NaN               NaN           NaN
NPolygons                 +0.265         -0.064            +0.186        +0.238
bin_rich_base_rate        +0.362         +0.083            +0.102        +0.289
reg_mean_true_fa          +0.334         -0.049            +0.221        +0.316
```

## Performance by manifest BoulderLabel

```
             bin_rich_auc              bin_rich_lift              reg_spearman              reg_presence_auc             
                     mean median count          mean median count         mean median count             mean median count
BoulderLabel                                                                                                             
Boulder rich        0.617  0.607    35         1.451  1.110    35        0.171  0.094    36            0.577  0.541    23
unknown             0.577  0.577     2         1.059  1.059     2        0.123  0.123     2            0.602  0.602     2
```
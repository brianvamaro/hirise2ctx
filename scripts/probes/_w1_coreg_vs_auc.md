# W1 Rung 1b — coreg shift/quality vs per-image AUC

Coreg solves: `cache_v2\coregistration` (block-median, applied to polygons in Stage 4).
Recipe: two_stage_balanced × boulder_count @ S=64. Tile = 320 m.

## Correlations
- `peak` vs meaningful_auc: Spearman rho=-0.168 p=0.3135 (n=38)
- `mag_m` vs meaningful_auc: Spearman rho=-0.003 p=0.9839 (n=38)
- `confident_frac` vs meaningful_auc: Spearman rho=-0.021 p=0.9010 (n=38)
- `mad_dy_px` vs meaningful_auc: Spearman rho=-0.183 p=0.2708 (n=38)
- `mad_dx_px` vs meaningful_auc: Spearman rho=-0.187 p=0.2603 (n=38)
- `n_confident` vs meaningful_auc: Spearman rho=+0.204 p=0.2204 (n=38)
- sign +1: dy_tiles vs best_di rho=-0.313 p=0.0559; dx_tiles vs best_dj rho=+0.075 p=0.6545
- sign -1: dy_tiles vs best_di rho=+0.313 p=0.0559; dx_tiles vs best_dj rho=-0.075 p=0.6545

## Per-image table (sorted by AUC)
```
                 meaningful_auc  anti_signal  best_di  best_dj   gain     dy_m     dx_m   mag_m  peak  confident_frac  mad_dy_px  mad_dx_px        method
obs_id                                                                                                                                                   
ESP_055978_2270           0.303         True        2       -2  0.196 -135.249   -1.000 135.253 0.626           0.562      1.400      1.100  block_median
ESP_076499_1160           0.367         True       -2       -1  0.078 -198.749  259.999 327.262 0.586           0.520      0.700      0.950  block_median
ESP_047976_2020           0.378         True        1       -1  0.181 -126.999    1.000 127.003 0.667           1.000      0.300      1.300  block_median
ESP_046328_2180           0.400         True       -1       -2  0.119  -46.500  -85.250  97.107 0.688           1.000      0.350      0.400  block_median
ESP_071699_2260           0.403         True        1        0  0.035 -142.999   42.250 149.110 0.731           0.946      0.700      1.150  block_median
ESP_054000_2255           0.430         True        1       -2  0.038 -224.499   46.000 229.163 0.753           1.000      0.650      1.650  block_median
ESP_055017_2055           0.434         True        2        2  0.120 -234.249   24.625 235.540 0.781           0.973      0.500      0.975  block_median
ESP_049242_2115           0.457         True        0        2  0.050 -102.499  -27.125 106.028 0.717           0.828      0.725      0.900  block_median
ESP_054622_2240           0.461         True        1        2  0.387 -186.249    3.000 186.273 0.751           1.000      1.350      0.825  block_median
ESP_064510_2260           0.475         True       -1       -2  0.116 -192.749    6.500 192.859 0.748           0.875      1.100      3.700  block_median
ESP_055253_2245           0.492         True        2        2  0.027 -262.999   67.750 271.585 0.795           1.000      0.200      0.600  block_median
ESP_045983_2270           0.513        False       -2        2  0.044 -116.499   53.750 128.301 0.727           1.000      0.550      0.750  block_median
ESP_054397_2105           0.516        False       -1        0  0.029 -182.749   41.500 187.402 0.759           1.000      0.250      0.800  block_median
ESP_059421_2170           0.523        False        2        2  0.061  -94.000   66.750 115.288 0.621           0.732      0.800      1.050  block_median
ESP_071093_2210           0.525        False        0       -1  0.050 -244.874   18.375 245.562 0.578           0.821      0.600      0.550  block_median
ESP_069669_2220           0.531        False        1        0  0.042 -240.124  121.999 269.339 0.836           1.000      0.425      0.550  block_median
ESP_076565_2215           0.567        False        2        1  0.088 -285.499   29.000 286.968 0.705           1.000      0.250      0.775  block_median
ESP_048688_2085           0.578        False        2        2  0.143 -209.249  -21.250 210.325 0.642           0.846      0.200      1.150  block_median
ESP_046959_2225           0.592        False        2       -2  0.021 -119.999    9.875 120.405 0.875           0.941      0.500      0.675  block_median
ESP_051943_2270           0.596        False        0        0  0.000 -158.999  112.374 194.702 0.689           0.824      0.375      1.475  block_median
ESP_045390_2215           0.620        False        1        0  0.040 -201.749   41.000 205.873 0.852           1.000      0.400      0.750  block_median
ESP_063429_2240           0.627        False       -1       -1  0.034 -137.249   51.500 146.593 0.719           0.962      0.500      1.100  block_median
ESP_045878_2235           0.637        False        2        0  0.083 -266.499   87.000 280.340 0.829           1.000      0.350      1.000  block_median
ESP_052576_2250           0.653        False        1        0  0.055 -209.499  105.999 234.789 0.751           0.820      0.450      0.775  block_median
ESP_068402_2240           0.654        False        0        1  0.008 -194.374   66.750 205.516 0.722           0.933      0.475      0.800  block_median
ESP_059686_2235           0.657        False       -2        0  0.081  -77.375   52.125  93.294 0.632           0.963      0.475      0.625  block_median
ESP_066634_2210           0.675        False        1        0  0.076 -222.249   45.250 226.809 0.585           0.833      0.900      1.575  block_median
ESP_054134_2265           0.687        False        0        0 -0.000 -116.499  -37.500 122.386 0.773           0.977      0.600      0.650  block_median
ESP_076723_2265           0.691        False        1        0  0.078 -209.249    2.750 209.267 0.617           0.880      0.650      1.100  block_median
ESP_017355_2260           0.720        False        1        0  0.022 -182.999  175.499 253.552 0.601           0.616      0.600      0.900  block_median
ESP_055055_2255           0.721        False       -1        2  0.016 -248.749   59.500 255.766 0.747           0.870      0.500      0.800  block_median
ESP_055690_2200           0.735        False        0       -1  0.006   -6.000 -162.999 163.110 0.673           0.869      1.000      0.700  block_median
ESP_045139_2270           0.737        False        2        0  0.000 -193.249   96.000 215.780 0.702           0.989      0.300      0.925  block_median
ESP_068483_2280           0.801        False        2       -1  0.041 -186.249   98.624 210.750 0.646           0.462      0.725      1.750  block_median
ESP_053989_2260           0.813        False        1        0  0.015 -114.874  -18.125 116.296 0.666           1.000      0.325      1.375  block_median
ESP_042964_2160           0.843        False        1        0  0.033 -179.499  106.374 208.652 0.657           0.968      0.500      0.475  block_median
ESP_045550_2180           0.945        False        2        2  0.027 -136.249  -69.750 153.065 0.707           1.000      0.250      0.450  block_median
ESP_069763_2235           0.979        False        0        0 -0.000  -77.250   20.500  79.923 0.670           1.000      0.450      0.700  block_median
```
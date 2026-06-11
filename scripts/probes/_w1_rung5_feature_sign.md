# W1 Rung 5 — within-image feature-label correlation signs

```
         obs_id  anti    n  shadow_fraction  grad_mag_mean  glcm_contrast_d1  glcm_energy_d1  intensity_std  edge_density  intensity_mean
ESP_017355_2260 False 2927            +0.63          +0.27             +0.47           -0.17          -0.01         +0.55           -0.61
ESP_042964_2160 False  608            +0.66          +0.65             +0.79           -0.51          +0.14         +0.82           -0.67
ESP_045139_2270 False 1342            +0.31          +0.67             +0.82           -0.61          +0.53         +0.67           -0.28
ESP_045390_2215 False  602            -0.01          +0.38             +0.49           -0.11          +0.01         +0.50           +0.14
ESP_045550_2180 False  719            +0.47          +0.55             +0.61           -0.50          +0.37         +0.55           -0.47
ESP_045878_2235 False 1766            -0.10          +0.50             +0.47           -0.50          +0.47         +0.46           +0.48
ESP_045983_2270 False  587            +0.03          -0.51             -0.22           +0.44          -0.52         -0.44           -0.34
ESP_046328_2180  True  439              NaN          +0.48             +0.60           -0.42          +0.10         +0.60           -0.35
ESP_046959_2225 False 1000            +0.10          +0.03             +0.24           +0.15          -0.30         +0.01           -0.31
ESP_047976_2020  True  773            -0.04          -0.01             -0.20           -0.11          +0.30         -0.33           +0.12
ESP_048688_2085 False  316            +0.17          +0.37             +0.35           -0.41          +0.26         +0.29           -0.17
ESP_049242_2115  True  572            +0.10          -0.09             -0.13           -0.06          +0.06         -0.08           -0.08
ESP_051943_2270 False  632            +0.09          +0.09             +0.08           -0.01          +0.03         +0.09           -0.11
ESP_052576_2250 False 1104            +0.33          +0.57             +0.53           -0.56          +0.53         +0.57           -0.30
ESP_053989_2260 False  656            +0.69          +0.43             +0.48           -0.39          +0.17         +0.48           -0.72
ESP_054000_2255  True  812            -0.13          -0.12             +0.25           -0.06          +0.05         +0.07           +0.08
ESP_054134_2265 False  798            +0.32          +0.25             +0.39           -0.11          -0.09         +0.36           -0.40
ESP_054397_2105 False 1459            +0.51          +0.66             +0.66           -0.64          +0.64         +0.53           -0.24
ESP_054622_2240 False 1204            +0.14          -0.38             -0.23           +0.30          -0.32         +0.08           -0.22
ESP_055017_2055 False  727            +0.10          -0.17             -0.22           +0.06          +0.07         -0.30           -0.12
ESP_055055_2255 False 1043            +0.06          +0.54             +0.35           -0.55          +0.56         +0.52           +0.18
ESP_055253_2245 False  617            +0.17          +0.55             +0.53           -0.50          +0.47         +0.51           +0.26
ESP_055690_2200 False 1181            +0.84          +0.84             +0.86           -0.85          +0.78         +0.85           -0.83
ESP_055978_2270  True 1310            -0.02          +0.19             +0.31           -0.16          +0.07         +0.19           +0.10
ESP_059421_2170 False 1084            -0.09          -0.09             +0.05           +0.20          -0.20         -0.16           +0.00
ESP_059686_2235  True 1035            +0.57          +0.01             -0.38           +0.06          +0.15         -0.42           -0.57
ESP_063429_2240 False  459            -0.05          +0.37             +0.45           -0.33          +0.24         +0.35           +0.26
ESP_064510_2260  True 1081              NaN          -0.16             -0.05           +0.10          -0.24         -0.09           +0.17
ESP_066634_2210 False  914            +0.26          +0.28             +0.54           -0.22          +0.12         +0.45           -0.30
ESP_068402_2240 False 1433            +0.07          +0.40             +0.27           -0.38          +0.40         +0.32           +0.25
ESP_068483_2280 False 1248            +0.61          +0.56             +0.64           -0.48          +0.29         +0.64           -0.63
ESP_069669_2220 False 1062            +0.41          +0.10             +0.08           -0.11          +0.13         +0.02           -0.26
ESP_069763_2235 False 1015            +0.35          +0.12             -0.08           -0.15          +0.23         -0.05           -0.34
ESP_071093_2210 False  792            +0.35          +0.20             +0.47           -0.13          -0.02         +0.38           -0.52
ESP_071699_2260 False  670            +0.42          +0.09             -0.09           -0.03          +0.10         -0.07           -0.43
ESP_076499_1160  True 1286            +0.73          +0.27             +0.19           -0.25          +0.30         +0.22           -0.63
ESP_076565_2215 False  581            +0.16          +0.33             +0.43           -0.26          +0.17         +0.34           -0.08
ESP_076723_2265 False 1461            +0.44          +0.80             +0.86           -0.74          +0.69         +0.80           -0.22
```

- `shadow_fraction`: cohort-majority sign + (healthy median +0.287, 30 imgs); anti median +0.042, sign-flipped in 5/8 anti images
- `grad_mag_mean`: cohort-majority sign + (healthy median +0.373, 30 imgs); anti median +0.002, sign-flipped in 4/8 anti images
- `glcm_contrast_d1`: cohort-majority sign + (healthy median +0.459, 30 imgs); anti median +0.072, sign-flipped in 4/8 anti images
- `glcm_energy_d1`: cohort-majority sign - (healthy median -0.296, 30 imgs); anti median -0.088, sign-flipped in 2/8 anti images
- `intensity_std`: cohort-majority sign + (healthy median +0.174, 30 imgs); anti median +0.084, sign-flipped in 1/8 anti images
- `edge_density`: cohort-majority sign + (healthy median +0.414, 30 imgs); anti median -0.004, sign-flipped in 4/8 anti images
- `intensity_mean`: cohort-majority sign - (healthy median -0.270, 30 imgs); anti median +0.003, sign-flipped in 4/8 anti images

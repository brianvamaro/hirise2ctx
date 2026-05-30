# Compression diagnostic — v2 LOIO `lightgbm_two_stage` S=64

Source: models\lightgbm_two_stage\629276139c22da68\scale_S64\predictions.parquet
Figure: reports\figures\12_compression_diagnostic.png

## Per-bin mean prediction (raw vs LOIO-isotonic recalibration)

              n_tiles  mean_true  mean_pred_raw  ratio_raw  mean_pred_iso  ratio_iso
truth_bin                                                                           
zero             2627     0.0000         0.0074        NaN         0.0099        NaN
0_to_1e-4        1192     0.0001         0.0079   111.8442         0.0105   148.9659
1e-4_to_1e-3     7493     0.0004         0.0082    18.5617         0.0109    24.7013
1e-3_to_1e-2    12820     0.0043         0.0106     2.4808         0.0134     3.1416
1e-2_to_max     13183     0.0347         0.0146     0.4197         0.0167     0.4814

## Per-fold (LOIO) headline

      spearman_raw  spearman_iso  auc_raw  auc_iso
mean        0.1689        0.1568   0.5792   0.5716
std         0.2285        0.2134   0.2267   0.1745

**Interpretation cheatsheet:**
- Panel A/B/C: how much does iso-recalibration close the bin-mean gap?
- Panel D: is p_pos collapsing toward 0 on true-zero tiles, or is it spreading mass everywhere?
- Panel E: does the magnitude head produce a flat distribution across truth bins (= the squash)?
- Panel F: do iso preds reach the high-bin diagonal that raw preds miss?
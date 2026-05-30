# Notebook 13 §4 correlation table

## Spearman rho (per-image features vs performance metrics)

metric              bin_rich_auc  bin_rich_ece  bin_rich_lift  reg_spearman
feature                                                                    
CenterLat                  0.208         0.060          0.237         0.280
EmissionAngle              0.273         0.164         -0.089         0.269
IncidenceAngle            -0.136        -0.273         -0.003        -0.198
NPolygons                  0.265         0.044         -0.064         0.238
PhaseAngle                 0.078        -0.243          0.070        -0.062
SubSolarAzimuth            0.032         0.061         -0.246        -0.058
bin_rich_base_rate         0.362         0.061          0.083         0.289
reg_mean_true_fa           0.334         0.070         -0.049         0.316

## p-values

metric              bin_rich_auc  bin_rich_ece  bin_rich_lift  reg_spearman
feature                                                                    
CenterLat                  0.216         0.722          0.158         0.088
EmissionAngle              0.102         0.331          0.602         0.103
IncidenceAngle             0.421         0.103          0.988         0.233
NPolygons                  0.113         0.798          0.705         0.149
PhaseAngle                 0.645         0.148          0.679         0.712
SubSolarAzimuth            0.852         0.720          0.143         0.728
bin_rich_base_rate         0.027         0.719          0.626         0.083
reg_mean_true_fa           0.044         0.682          0.774         0.053

## Significant (p < 0.05) only:

           feature       metric  n   rho     p
bin_rich_base_rate bin_rich_auc 37 0.362 0.027
  reg_mean_true_fa bin_rich_auc 37 0.334 0.044
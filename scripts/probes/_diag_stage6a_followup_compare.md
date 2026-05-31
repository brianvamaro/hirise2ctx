# Stage 6a follow-up sweep -- combined comparison

Source: `models\_sweep_stage6a\20260531T004356Z\aggregate.parquet`

Variant: `lightgbm_two_stage_balanced` (P1) + `target_col=boulder_count` (P2).  Dev = within-image 4-fold on 5 dataset_v2_dev images (20 folds).  Acceptance criteria (PROMOTION_QUEUE.md Stage 6a): Spearman delta >= +0.05 AND PR-AUC delta >= +0.03 vs the P1+P2 baseline.

## S = 32 (scale_idx=2)

| metric | P1+P2 baseline<br>(delta) | +6a default (3x3, mean+max+std)<br>(delta) | +6a 5x5 stencil (mean+max+std)<br>(delta) | +6a max-only (3x3, max)<br>(delta) |
|--------|:---:|:---:|:---:|:---:|
| Spearman rho | +0.2226 | +0.2440<br>(+0.0214) | +0.2760<br>(+0.0534) | +0.2433<br>(+0.0207) |
| PR-AUC | 0.4934 | 0.5467<br>(+0.0533) | 0.5460<br>(+0.0526) | 0.5275<br>(+0.0341) |
| normalised lift @top-K | 0.4693 | 0.5263<br>(+0.0570) | 0.5240<br>(+0.0547) | 0.5086<br>(+0.0393) |
| precision @top-5% | 0.5160 | 0.5789<br>(+0.0629) | 0.5882<br>(+0.0722) | 0.5736<br>(+0.0575) |
| recall @top-5% | 0.0679 | 0.1473<br>(+0.0794) | 0.1225<br>(+0.0546) | 0.1326<br>(+0.0647) |

Acceptance verdict at this scale:

- **+6a default (3x3, mean+max+std)**: Spearman +0.0214 (< +0.05), PR-AUC +0.0533 (>= +0.03) --> **FAIL**
- **+6a 5x5 stencil (mean+max+std)**: Spearman +0.0534 (>= +0.05), PR-AUC +0.0526 (>= +0.03) --> **PASS**
- **+6a max-only (3x3, max)**: Spearman +0.0207 (< +0.05), PR-AUC +0.0341 (>= +0.03) --> **FAIL**

## S = 64 (scale_idx=3)

| metric | P1+P2 baseline<br>(delta) | +6a default (3x3, mean+max+std)<br>(delta) | +6a 5x5 stencil (mean+max+std)<br>(delta) | +6a max-only (3x3, max)<br>(delta) |
|--------|:---:|:---:|:---:|:---:|
| Spearman rho | +0.2826 | +0.2760<br>(-0.0065) | +0.3095<br>(+0.0269) | +0.2442<br>(-0.0383) |
| PR-AUC | 0.6396 | 0.6494<br>(+0.0098) | 0.6435<br>(+0.0039) | 0.6517<br>(+0.0120) |
| normalised lift @top-K | 0.6188 | 0.6256<br>(+0.0068) | 0.6257<br>(+0.0069) | 0.6191<br>(+0.0003) |
| precision @top-5% | 0.6600 | 0.7036<br>(+0.0436) | 0.6550<br>(-0.0050) | 0.6787<br>(+0.0186) |
| recall @top-5% | 0.0577 | 0.0782<br>(+0.0204) | 0.0585<br>(+0.0008) | 0.0798<br>(+0.0221) |

Acceptance verdict at this scale:

- **+6a default (3x3, mean+max+std)**: Spearman -0.0065 (< +0.05), PR-AUC +0.0098 (< +0.03) --> **FAIL**
- **+6a 5x5 stencil (mean+max+std)**: Spearman +0.0269 (< +0.05), PR-AUC +0.0039 (< +0.03) --> **FAIL**
- **+6a max-only (3x3, max)**: Spearman -0.0383 (< +0.05), PR-AUC +0.0120 (< +0.03) --> **FAIL**

## Best absolute numbers across the whole grid

| metric | scheme | scale_idx | S | value |
|--------|--------|----------:|--:|------:|
| Spearman rho | within_image_4fold_nbr_s5 | 3 | 64 | +0.3095 |
| PR-AUC | within_image_4fold_nbr_max | 3 | 64 | 0.6517 |
| normalised lift @top-K | within_image_4fold_nbr_s5 | 3 | 64 | 0.6257 |
| precision @top-5% | within_image_4fold_nbr | 3 | 64 | 0.7036 |
| recall @top-5% | within_image_4fold_nbr | 2 | 32 | 0.1473 |

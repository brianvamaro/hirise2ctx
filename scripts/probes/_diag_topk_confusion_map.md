# Top-K confusion-map probe

Figure: reports\figures\13_topk_confusion_map.png

Per-image confusion counts at K = n_pos:

          ObsId  n_tiles  n_pos  base_rate  TP  FP  FN  precision@K  lift@K
ESP_042964_2160      608     50      0.082  22  28  28        0.440   5.350
ESP_046959_2225     1000    249      0.249  96 153 153        0.386   1.548
ESP_054000_2255      812    149      0.183   8 141 141        0.054   0.293

**Reading the figure:**
- Green tiles dominate ⇒ the model correctly identified boulder-rich tiles.
- A green-dominant cluster ⇒ AUC is high and that's *operationally* meaningful.
- Red dominant ⇒ the model's top-K confidently flagged tiles that aren't boulder-rich.
- Orange dominant ⇒ the model missed many true boulder-rich tiles.
- Anti-signal cases (AUC < 0.5) have nearly all red/orange and almost no green.
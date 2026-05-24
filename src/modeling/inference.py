"""Off-HiRISE prediction stub.

Defines the I/O contract for running a trained model across an arbitrary CTX raster
region (i.e. Murray Lab mosaic tiles where no HiRISE detections exist). This is the
project's whole point per CLAUDE.md and PLAN_modeling.md §7, but the actual global
sweep across Murray Lab tiles is deferred -- here we land the contract so today's
training artifacts don't paint us into a corner.

The contract is intentionally narrow:

  predict_over_ctx_region(
      ctx_raster: Path,             # a downloaded Murray Lab tile, on its native CRS
      feature_parquet: Path,        # output of a Stage-4b-like feature extractor over
                                    # the same CTX raster (no label dependency!)
      *,
      model: Model,
      tile_size_px: int,
      out_parquet: Path,
      patch_stack_npy: Path | None = None,  # only required for CNN models
  ) -> dict

The function:
  1. Reads `feature_parquet` (expected schema: same X-columns as packaged training
     parquets + tile-key columns + optional patch_idx_S*).
  2. Applies the trained model's `predict`. For CNN models, the caller supplies the
     pre-computed patch stack via `patch_stack_npy`.
  3. Writes a tidy parquet: (ti, tj, tile_size_px) + predicted abundance + model
     provenance (`config_hash`, `model_hash`).

This file is intentionally a stub. Implementation goes alongside building the
off-HiRISE feature extractor, which itself requires a thin wrapper around
src.features that operates on a CTX raster without consuming labels. Stage 4b's
shadow detector already uses image-percentile thresholds (no label leak), so the
seam is genuinely clean; what's missing is the wrapper script and a Murray Lab
mosaic-tile iterator. Both belong in a separate follow-up phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceRequest:
    """Self-documenting argument bundle for the eventual off-HiRISE sweep."""

    ctx_raster: Path
    feature_parquet: Path
    tile_size_px: int
    out_parquet: Path
    patch_stack_npy: Path | None = None


def predict_over_ctx_region(request: InferenceRequest, model) -> dict:
    """Run a trained model across an arbitrary CTX region. Not implemented in Week 3.

    Returns a dict describing what would be written:
        {'rows_predicted': int, 'config_hash': str, 'model_hash': str,
         'out_parquet': str}
    """
    raise NotImplementedError(
        "Off-HiRISE inference is a follow-up phase. See PLAN_modeling.md §7 for "
        "the sketch and src.modeling.inference's module docstring for the contract."
    )

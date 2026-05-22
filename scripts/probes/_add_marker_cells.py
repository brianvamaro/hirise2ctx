"""Insert marker-overlay cells into notebooks/05_coregistration_qa.ipynb.

Idempotent: re-running checks for the marker cell tag and skips if present.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "05_coregistration_qa.ipynb"
TAG = "stage3-marker-overlay"

MARKDOWN_SOURCE = """\
## Per-image feature crosshairs (BEFORE / AFTER shift)

Hard to read sub-pixel alignment from the red/blue overlays alone, especially on
boulder-poor / weakly-textured scenes (ESP_056165_2200, peak=0.28). This second
visualization pins down ~6 distinctive features detected in the CTX sub-window and
plots crosshairs at the SAME pixel coords on all three panels.

The eye should read it as:
- **CTX panel:** crosshairs ON the bright features (definition of correct).
- **BEFORE panel:** the same screen coords miss the corresponding HiRISE features
  by the magnitude of the un-corrected offset.
- **AFTER panel:** the same screen coords land back ON the corresponding HiRISE
  features — confirming that the rigid translation does what we measured.

Bland scenes still get crosshairs at *something* — `peak_local_max` returns the
strongest local extrema even when those are dim. ESP_056165_2200 is the cleanest
test of this: low peak correlation, but the markers should still snap into place.
"""

CODE_SOURCE = """\
from src.coregister import find_tracking_features


def plot_one_with_markers(obs_id: str) -> Path | None:
    \"\"\"Render CTX | BEFORE | AFTER panels with identical feature crosshairs.\"\"\"
    coreg = load_shift(obs_id, cfg.cache_dir)
    if coreg is None:
        return None
    fft = coreg[\"fft_window\"]
    size = fft[\"size_px\"]
    r0, c0 = fft[\"row_off\"], fft[\"col_off\"]
    dy_px, dx_px = coreg[\"shift_px\"][\"dy\"], coreg[\"shift_px\"][\"dx\"]

    ctx_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f\"{obs_id}.tif\"
    with rasterio.open(ctx_tif) as ds:
        ctx = ds.read(1).astype(np.float32)
    row = df.set_index(\"ObsId\").loc[obs_id]
    hi, _, _ = _warp_hirise_to_ctx_grid(
        obs_id, jp2_url=str(row[\"JP2_URL\"]), cache_dir=cfg.cache_dir, ctx_window_tif=ctx_tif,
    )
    ctx_sub = ctx[r0:r0+size, c0:c0+size]
    hi_sub = hi[r0:r0+size, c0:c0+size]
    hi_shifted = nd_shift(hi_sub, shift=(dy_px, dx_px), order=1, mode=\"constant\", cval=0.0)

    # Find the marker coordinates on the CTX sub-window once; the same coords are
    # reused on the HiRISE panels so any misalignment is immediately legible.
    feats = find_tracking_features(ctx_sub, n_features=6, min_distance=size // 6, edge_margin=size // 16)

    def norm(a):
        a = a.astype(np.float32)
        if (a > 0).any():
            lo, hi_ = np.percentile(a[a > 0], (2, 98))
        else:
            lo, hi_ = 0.0, 1.0
        return np.clip((a - lo) / max(hi_ - lo, 1e-9), 0, 1)

    def draw_markers(ax, coords, color, label_offset=True):
        for i, (r, c) in enumerate(coords):
            # Crosshair: short tick marks + small circle, leaving the central feature visible.
            ax.plot([c - 7, c - 2], [r, r], color=color, lw=1.3)
            ax.plot([c + 2, c + 7], [r, r], color=color, lw=1.3)
            ax.plot([c, c], [r - 7, r - 2], color=color, lw=1.3)
            ax.plot([c, c], [r + 2, r + 7], color=color, lw=1.3)
            ax.add_patch(plt.Circle((c, r), radius=9, edgecolor=color, facecolor=\"none\", lw=1.0))
            if label_offset:
                ax.text(c + 11, r - 11, str(i + 1), color=color, fontsize=8,
                        weight=\"bold\", path_effects=[])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(norm(ctx_sub), cmap=\"gray\")
    axes[0].set_title(\"CTX (fixed)  —  crosshairs ON features\")
    draw_markers(axes[0], feats, color=\"#ffe600\")

    axes[1].imshow(norm(hi_sub), cmap=\"gray\")
    axes[1].set_title(\"HiRISE BEFORE shift  —  crosshairs OFF features\")
    draw_markers(axes[1], feats, color=\"#ffe600\")

    axes[2].imshow(norm(hi_shifted), cmap=\"gray\")
    axes[2].set_title(\"HiRISE AFTER shift  —  crosshairs back ON features\")
    draw_markers(axes[2], feats, color=\"#ffe600\")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, size); ax.set_ylim(size, 0)

    sm = coreg[\"shift_m\"]
    fig.suptitle(
        f\"{obs_id}  —  fft={size}px  shift=({sm['dx']:+.0f}, {sm['dy']:+.0f}) m  \"
        f\"|{sm['magnitude']:.0f}| m  peak={coreg['peak_correlation']:.2f}  \"
        f\"({len(feats)} tracked features)\",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = FIG_DIR / f\"05_markers_{obs_id}.png\"
    fig.savefig(out_path, dpi=110)
    plt.show()
    return out_path


for obs in summary.index:
    plot_one_with_markers(obs)
"""


def _new_markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {"tags": [TAG]},
        "source": source.splitlines(keepends=True),
    }


def _new_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    if any(TAG in (c.get("metadata") or {}).get("tags", []) for c in nb["cells"]):
        print("Marker cells already present — nothing to do.")
        return
    # Insert before the "Decision space" markdown cell (last cell).
    insert_at = len(nb["cells"]) - 1
    nb["cells"].insert(insert_at, _new_markdown(MARKDOWN_SOURCE))
    nb["cells"].insert(insert_at + 1, _new_code(CODE_SOURCE))
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted 2 cells at position {insert_at}. Total cells now: {len(nb['cells'])}")


if __name__ == "__main__":
    main()

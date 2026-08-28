"""Make a regional mosaic fast to open in QGIS/ArcGIS -- WITHOUT touching the mosaic itself.

`scripts/map_mosaics.py` already writes the archival mosaic tiled 256x256 with DEFLATE, so the
packaging is fine. Two things are missing for a viewer, and both can be supplied as *sidecars*:

* **overviews.** Without them, a 10,370 x 7,407 float32 raster must be read in full at every
  zoom level. `gdaladdo -ro` writes them to an external `<name>.tif.ovr`, leaving the GeoTIFF
  byte-for-byte unchanged (asserted here by sha256 before and after).
* **statistics.** With none, a viewer defaults to a min/max stretch, and on a layer where ~42 %
  of valid cells are exactly 0 and p98 of the non-zero cells is ~0.045 against a max of 0.13,
  that renders a near-black rectangle. `gdalinfo -stats` writes them to `<name>.tif.aux.xml`.

⚠ **An earlier version of this script made a "display copy" instead, and it was a bad trade:**
the source is already compressed, so re-encoding with `predictor=3` compressed *worse*, and the
copy came out at 235 MB against the original's 103 MB -- a duplicate of the product, larger than
the product, that a reader could mistake for it. Sidecars have none of those problems: nothing
is duplicated, nothing can be mis-cited, and deleting them costs nothing.

The suggested stretch is printed for you to type into the viewer. It is `0 -> p98 of the
NON-ZERO cells`: the target is heavily zero-inflated by nature (CLAUDE.md), so a stretch that
includes the zeros in its percentile is dominated by them.

    python scripts/make_display_geotiff.py reports/map_extended/regional_abundance_mosaic.tif
    python scripts/make_display_geotiff.py --clean reports/map_extended/*_mosaic.tif
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                                        # noqa: E402
import rasterio                                                           # noqa: E402

#: conda puts the GDAL binaries in the env's Library/bin, which is not on PATH under `conda run`.
_GDAL_BIN = Path(sys.prefix) / "Library" / "bin"


def gdal_tool(name: str) -> str:
    """Absolute path to a GDAL CLI tool, because `conda run` does not put them on PATH."""
    for cand in (_GDAL_BIN / f"{name}.exe", _GDAL_BIN / name, Path(sys.prefix) / "bin" / name):
        if cand.exists():
            return str(cand)
    return name                                    # fall back to PATH (Linux/Sherlock)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def suggested_stretch(arr: np.ndarray) -> dict:
    """Percentiles for a usable default stretch, from the finite cells only."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    nz = finite[finite > 0]
    out = {"min": float(finite.min()), "max": float(finite.max()),
           "mean": float(finite.mean()), "valid_pct": 100.0 * finite.size / arr.size,
           "zero_fraction": float((finite == 0).mean())}
    if nz.size:
        out["p98_nonzero"] = float(np.percentile(nz, 98))
    return out


def sidecars(tif: Path) -> list[Path]:
    return [tif.with_name(tif.name + ".ovr"), tif.with_name(tif.name + ".aux.xml")]


def build(tif: Path, *, levels=(2, 4, 8, 16, 32, 64)) -> None:
    before = hashlib.sha256(tif.read_bytes()).hexdigest()
    with rasterio.open(tif) as src:
        arr = src.read(1)
        shape = src.shape
    st = suggested_stretch(arr)
    del arr

    env = {**os.environ, "PATH": f"{_GDAL_BIN}{os.pathsep}{os.environ.get('PATH', '')}"}
    # -ro forces EXTERNAL overviews; DEFLATE keeps the .ovr small. average skips nodata cells.
    env["COMPRESS_OVERVIEW"] = "DEFLATE"
    subprocess.run([gdal_tool("gdaladdo"), "-ro", "-r", "average", str(tif),
                    *(str(x) for x in levels)], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    subprocess.run([gdal_tool("gdalinfo"), "-stats", str(tif)], check=True, env=env,
                   stdout=subprocess.DEVNULL)

    after = hashlib.sha256(tif.read_bytes()).hexdigest()
    if before != after:
        raise SystemExit(f"{tif.name}: the GeoTIFF itself changed -- it must not. "
                         "Delete the sidecars and investigate before trusting this file.")
    made = [p for p in sidecars(tif) if p.exists()]
    with rasterio.open(tif) as chk:
        n_ovr = len(chk.overviews(1))

    print(f"  {_rel(tif)}   {shape[0]} x {shape[1]}")
    print(f"    GeoTIFF UNCHANGED (sha256 {before[:16]}...), {n_ovr} external overview level(s)")
    for p in made:
        print(f"      + {p.name}  {p.stat().st_size / 1e6:,.1f} MB")
    if st:
        print(f"    values [{st['min']:.6g}, {st['max']:.6g}]  mean {st['mean']:.6g}  "
              f"valid {st['valid_pct']:.3f}%  zeros {100 * st['zero_fraction']:.1f}% of valid")
        if "p98_nonzero" in st:
            print(f"    SUGGESTED STRETCH in QGIS/ArcGIS:  min 0   max {st['p98_nonzero']:.6g}"
                  "   (p98 of the non-zero cells)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rasters", nargs="+", type=Path)
    ap.add_argument("--clean", action="store_true",
                    help="remove the .ovr/.aux.xml sidecars instead of building them")
    args = ap.parse_args()

    for p in args.rasters:
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
    if args.clean:
        print("=== removing display sidecars ===")
        for p in args.rasters:
            for s in sidecars(p):
                if s.exists():
                    s.unlink()
                    print(f"  removed {_rel(s)}")
        return 0

    print("=== display sidecars (the GeoTIFFs are not modified) ===", flush=True)
    for p in args.rasters:
        build(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# scripts/probes

Throwaway debug/probe scripts (`_`-prefix convention). Useful when an upstream
convention surprises us and the canonical pipeline drivers in `scripts/` need to
stay clean. Each file's docstring explains when it was useful — re-run only if
you hit the same class of issue. Inventory:

- `_probe_jp2_crs.py` — compare JP2 embedded CRS vs Stage 1 sidecar (SP1 bug check).
- `_check_decimated_sp1.py` — dump SP1 of every cached decimated TIFF after the JP2-side fix.
- `_verify_sp1_fix.py` — force re-read of a decimated cache through the override path.
- `_probe_pyproj_sp1.py`, `_probe_sp1_regex.py` — pyproj WKT canonicalization debugging.
- `_probe_murray_url_variants.py` — HEAD-probe Murray Lab URLs across sign quadrants.
- `_extract_allowlist_candidates.py` — tabulate transcript Bash/MCP frequencies (used by `/fewer-permission-prompts`).
- `_add_marker_cells.py` — one-off notebook cell injection for the Stage 3 marker overlay.

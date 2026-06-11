"""One-shot migration: rewrite cached coregistration JSONs with the corrected
row->world-y sign (W1 rung-1 bug, DECISIONS.md 2026-06-10).

The solves (shift_px) are correct; only the metre conversion was wrong, so we
recompute dy_m = -dy_px * px_y (and the single_window copy) instead of
re-solving. Idempotent: files already carrying the marker are skipped.
"""
import json
from pathlib import Path

DIRS = [Path("cache/coregistration"), Path("cache_v2/coregistration"), Path("cache_v2_dev/coregistration")]
MARKER = "y_sign_fix_applied"

for d in DIRS:
    if not d.exists():
        print(f"{d}: missing, skipped")
        continue
    n_fixed = n_skipped = 0
    for f in sorted(d.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get(MARKER):
            n_skipped += 1
            continue
        px_y = abs(rec["ctx_transform"][4])
        old_dy = rec["shift_m"]["dy"]
        rec["shift_m"]["dy"] = -rec["shift_px"]["dy"] * px_y
        assert abs(rec["shift_m"]["dy"] + old_dy) < 1e-6, f"{f}: unexpected stored dy_m"
        sw = rec.get("single_window")
        if sw is not None:
            sw["dy_m"] = -sw["dy_px"] * px_y
        rec[MARKER] = "2026-06-10"
        f.write_text(json.dumps(rec, indent=2))
        n_fixed += 1
    print(f"{d}: fixed {n_fixed}, already-fixed {n_skipped}")

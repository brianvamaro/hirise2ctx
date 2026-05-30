"""Pull text outputs from executed notebook 13 to confirm key findings.

Writes to a markdown file (avoids Windows cp1252 stdout encoding issues with
unicode characters in notebook outputs).
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import nbformat

nb = nbformat.read(REPO / "notebooks" / "13_per_image_heterogeneity.ipynb", as_version=4)
WANT = {"lbl-extract", "ctx-seam-extract", "corr-table", "anti-shadow", "anti-topk"}
out = Path(__file__).with_suffix(".md")
lines = ["# Notebook 13 key outputs", ""]
for c in nb.cells:
    if c.cell_type != "code" or c.id not in WANT:
        continue
    lines.append(f"\n## {c.id}\n")
    lines.append("```")
    for o in c.outputs:
        if "text" in o:
            lines.append(o["text"])
        elif o.get("output_type") == "execute_result" and "data" in o:
            tp = o["data"].get("text/plain")
            if tp:
                lines.append(tp)
    lines.append("```")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out}", flush=True)

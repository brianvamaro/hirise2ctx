"""Print the executed Stage 7d notebook's verdict + setup outputs to stdout."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent.parent / "notebooks" / "15_stage7d_pooled.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    txts: list[str] = []
    for out in c.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            txts.append("".join(out.get("text", [])))
        elif kind in {"execute_result", "display_data"}:
            data = out.get("data", {})
            if "text/plain" in data:
                txts.append("".join(data["text/plain"]))
    joined = "\n".join(txts).strip()
    if joined:
        print(f"=== cell {i} ===")
        print(joined[:2500])
        print()

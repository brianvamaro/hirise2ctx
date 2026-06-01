"""Extract the verdict + numbers from the executed notebook 14."""
from __future__ import annotations

import json
from pathlib import Path

nb = json.loads(Path("notebooks/14_compositional_feasibility.ipynb").read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    cid = cell.get("id", "")
    if cid not in ("verdict-code", "dust-code", "test-b-code", "test-a-code"):
        continue
    print(f"\n===== cell: {cid} =====")
    for out in cell.get("outputs", []):
        ot = out.get("output_type")
        if ot in ("stream",):
            print("".join(out.get("text", [])))
        elif ot == "execute_result":
            data = out.get("data", {})
            for mime, val in data.items():
                if mime == "text/plain":
                    print("".join(val) if isinstance(val, list) else val)
        elif ot == "error":
            print("ERROR:", out.get("ename"), out.get("evalue"))
            for tb in out.get("traceback", []):
                print(tb)

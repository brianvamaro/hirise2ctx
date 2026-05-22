"""Verify notebook 06 produced sensible outputs by reading back its executed cells."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
nb_path = REPO_ROOT / "notebooks/06_labeling_qa.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Grab the text outputs from each code cell (errors will be visible in 'output_type'='error').
errors = []
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    for o in cell.get("outputs", []):
        if o.get("output_type") == "error":
            errors.append((i, o.get("ename"), o.get("evalue")))
        elif o.get("output_type") == "stream":
            txt = o.get("text", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            txt = txt.strip()
            if txt:
                print(f"--- cell {i} stream ---\n{txt}\n")

if errors:
    print("ERRORS FOUND:")
    for i, name, val in errors:
        print(f"  cell {i}: {name}: {val}")
    sys.exit(1)
print("OK -- no errors in executed notebook")

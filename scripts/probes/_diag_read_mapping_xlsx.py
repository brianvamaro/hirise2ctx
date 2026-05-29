"""One-off: dump the structure of the vClaire mapping spreadsheet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

XLSX = Path(r"C:\Users\brian\Downloads\Mapping_Images_33_36.xlsx")

xl = pd.ExcelFile(XLSX)
print(f"sheets: {xl.sheet_names}")
for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print(f"\n===== sheet '{sheet}' : {df.shape[0]} rows x {df.shape[1]} cols =====")
    print(f"columns: {df.columns.tolist()}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(8).to_string())

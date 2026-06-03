"""Dump the user's pre-existing terrain classification spreadsheet."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

PATH = Path("C:/Users/brian/Downloads/Mapping_Images_33_36.xlsx")
xl = pd.ExcelFile(PATH)
for sheet in xl.sheet_names:
    df = pd.read_excel(PATH, sheet_name=sheet)
    print(f"\n=== sheet: {sheet}  ({len(df)} rows, cols: {list(df.columns)}) ===")
    print(df.to_string(index=False, max_colwidth=80))

"""One-shot: list the 52 feature columns in dataset/packaged/loio_9fold/."""
import pandas as pd

df = pd.read_parquet("dataset/packaged/loio_9fold/X_test_fold0.parquet")
print(f"{len(df.columns)} columns:")
for i, c in enumerate(df.columns):
    print(f"  Column_{i}  ->  {c}")

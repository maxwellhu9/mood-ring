"""Phase 1a: look at the data before modeling anything.

Rule of the project: never train on a dataset you haven't looked at.
"""

import pandas as pd
from datasets import load_dataset

ds = load_dataset("dair-ai/emotion")  # downloads once, then cached in ~/.cache/huggingface

print(ds)  # splits and sizes

labels = ds["train"].features["label"].names
print(f"\nlabels: {labels}")

df = ds["train"].to_pandas()
df["label_name"] = df["label"].map(dict(enumerate(labels)))

print("\nclass balance (train):")
print(df["label_name"].value_counts(normalize=True).round(3))

print("\ntext length (words):")
print(df["text"].str.split().str.len().describe().round(1))

print("\none example per class:")
for name, group in df.groupby("label_name"):
    print(f"  [{name}] {group['text'].iloc[0]}")

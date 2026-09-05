"""Check R3iwan Kazakh dataset + inspect downloaded Russian CSV (tab-separated)."""
import pandas as pd
from datasets import load_dataset

print("=" * 70)
print("RU CSV: rureviews (tab-separated)")
ru = pd.read_csv(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\data\raw\rureviews.csv", sep="\t")
print("shape", ru.shape)
print("cols", list(ru.columns))
print("sentiment counts:", ru["sentiment"].value_counts().to_dict())
print("null reviews:", ru["review"].isna().sum())
print()

print("=" * 70)
print("KZ dataset: R3iwan/entertainment-reviews-kazakh")
try:
    ds = load_dataset("R3iwan/entertainment-reviews-kazakh")
    print("splits:", {k: len(v) for k, v in ds.items()})
    split = ds[list(ds.keys())[0]]
    print("features:", split.features)
    for i in range(3):
        row = split[i]
        print(f"--- sample {i} ---")
        for k, v in row.items():
            print(f"  {k}: {str(v)[:140]}")
    # class distribution
    for k in split.features:
        if split.features[k].dtype in ("class_label",):
            print(f"label dist {k}:", {l: int((split[k] == l).sum()) for l in range(len(split.features[k].names))})
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:300])

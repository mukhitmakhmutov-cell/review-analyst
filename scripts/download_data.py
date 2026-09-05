"""Download the full Yelp reviews dataset and build a balanced 3-class sample.

Mapping: 1-2 stars -> negative, 3 stars -> neutral, 4-5 stars -> positive.
Loads the FULL dataset (so the sample is random across businesses, not
business-clustered), then samples 10k rows per class with a fixed seed.
Saves to data/raw/yelp_reviews.csv.
"""
import pandas as pd
from datasets import load_dataset

PER_CLASS = 10_000
SEED = 42

print("Loading full dataset...")
ds = load_dataset("Yelp/yelp_review_full", split="train")
df = ds.to_pandas().rename(columns={"text": "review", "label": "rating"})
df["rating"] = df["rating"] + 1
print("Full dataset:", df.shape)


def label_for(star: int) -> str:
    if star <= 2:
        return "negative"
    if star == 3:
        return "neutral"
    return "positive"


df["sentiment"] = df["rating"].map(label_for)
print(df["sentiment"].value_counts())

parts = []
for lab in ("negative", "neutral", "positive"):
    sub = df[df["sentiment"] == lab]
    parts.append(sub.sample(n=PER_CLASS, random_state=SEED))
sample = pd.concat(parts, ignore_index=True)
sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
sample = sample[["review", "rating", "sentiment"]]

out = r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\data\raw\yelp_reviews.csv"
sample.to_csv(out, index=False)
print(sample["sentiment"].value_counts())
print("saved:", out)

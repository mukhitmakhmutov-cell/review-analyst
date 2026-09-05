"""Inspect the RU and KZ sentiment datasets: availability, features, samples."""
from datasets import load_dataset

def peek(name, **kw):
    print("=" * 70)
    print("DATASET:", name)
    try:
        ds = load_dataset(name, **kw)
        print("configs/splits:", {k: len(v) for k, v in ds.items()})
        split = ds[list(ds.keys())[0]]
        print("features:", split.features)
        for i in range(3):
            row = split[i]
            print(f"--- sample {i} ---")
            for k, v in row.items():
                sv = str(v)
                print(f"  {k}: {sv[:120]}")
    except Exception as e:
        print("ERROR:", type(e).__name__, str(e)[:300])
    print()

# Russian
peek("sismetanin/rureviews")
# Kazakh (3-class movie reviews)
peek("yeshpanovrustem/100k_movie_reviews_from_kz")

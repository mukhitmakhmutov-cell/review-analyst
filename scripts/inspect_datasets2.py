"""Check alternative accessible datasets: issai/kazsandra (KZ) and MonoHime RU."""
from datasets import load_dataset

def peek(name, config=None, **kw):
    print("=" * 70)
    print("DATASET:", name, "| config:", config)
    try:
        ds = load_dataset(name, config, **kw) if config else load_dataset(name, **kw)
        print("splits:", {k: len(v) for k, v in ds.items()})
        split = ds[list(ds.keys())[0]]
        print("features:", split.features)
        for i in range(2):
            row = split[i]
            print(f"--- sample {i} ---")
            for k, v in row.items():
                print(f"  {k}: {str(v)[:140]}")
    except Exception as e:
        print("ERROR:", type(e).__name__, str(e)[:300])
    print()

# Kazakh - KazSAnDRA (score_classification = 1..5 stars -> can map to 3-class)
peek("issai/kazsandra", "score_classification")
# Russian alternative
peek("MonoHime/ru_sentiment_dataset")

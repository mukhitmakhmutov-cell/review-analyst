"""Build a balanced multilingual (EN/RU/KZ) 3-class sentiment dataset.

Sources:
  EN: data/raw/yelp_reviews.csv                 (review, rating, sentiment)
  RU: data/raw/rureviews.csv                    (tab-sep, product/clothing domain)
      + clapAI/MultiLingualSentiment (language=ru)  (general domain: restaurant/
        organization/product reviews) — added so the model generalises beyond
        the narrow clothing domain of rureviews.
  KZ: R3iwan/entertainment-reviews-kazakh       (text, category, sentiment)

The RU *test* set is drawn from the general-domain source only, because that is
what the app is actually asked to classify (business reviews, not clothing).

Output: data/processed/train.jsonl + test.jsonl
  each line: {"text": str, "label": 0/1/2, "lang": "en"/"ru"/"kz"}
label map (matches the base model's head order): 0=positive, 1=neutral, 2=negative
"""
import json
import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

BASE = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

LABEL = {"positive": 0, "neutral": 1, "negative": 2}
RU_GENERAL_CAP = 8000  # per-label cap when streaming clapAI Russian


def norm_label(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("neautral", "neutural", "neuter"):
        return "neutral"
    return s


def clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"^\[\d+\]\s*", "", text)  # strip leading "[1] " index
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_en():
    df = pd.read_csv(RAW / "yelp_reviews.csv")
    out = []
    for _, r in df.iterrows():
        lab = norm_label(r["sentiment"])
        if lab in LABEL:
            out.append((clean(r["review"]), LABEL[lab]))
    return out


def load_ru_product():
    df = pd.read_csv(RAW / "rureviews.csv", sep="\t")
    out = []
    for _, r in df.iterrows():
        lab = norm_label(r["sentiment"])
        if lab in LABEL:
            out.append((clean(r["review"]), LABEL[lab]))
    return out


def load_ru_general():
    """Stream clapAI/MultiLingualSentiment and collect Russian rows per label."""
    ds = load_dataset("clapAI/MultiLingualSentiment", split="train", streaming=True)
    by_label = {0: [], 1: [], 2: []}
    for row in ds:
        if row["language"] != "ru":
            continue
        lab = norm_label(row["label"])
        if lab not in LABEL:
            continue
        text = clean(row["text"])
        if not text or len(text) > 1000:
            continue
        l = LABEL[lab]
        if len(by_label[l]) < RU_GENERAL_CAP:
            by_label[l].append(text)
        if all(len(v) >= RU_GENERAL_CAP for v in by_label.values()):
            break
    out = [(t, l) for l in (0, 1, 2) for t in by_label[l]]
    return out


def load_kz():
    ds = load_dataset("R3iwan/entertainment-reviews-kazakh")
    split = ds[list(ds.keys())[0]]
    out = []
    for row in split:
        lab = norm_label(row["sentiment"])
        if lab in LABEL:
            out.append((clean(row["text"]), LABEL[lab]))
    return out


def stratified_sample(pairs, n, seed=42):
    """Sample up to n (text,label) pairs, balanced across the 3 classes."""
    rng = random.Random(seed)
    by_class = {0: [], 1: [], 2: []}
    for t, l in pairs:
        by_class[l].append(t)
    per = n // 3
    rem = n - per * 3
    res = []
    for c in range(3):
        pool = by_class[c][:]
        rng.shuffle(pool)
        res.extend((t, c) for t in pool[: per + (1 if c < rem else 0)])
    rng.shuffle(res)
    return res


def split_lang(pairs, n_train, n_test, seed=42):
    """Disjoint stratified train/test split, balanced across the 3 classes."""
    rng = random.Random(seed)
    by_class = {0: [], 1: [], 2: []}
    for t, l in pairs:
        by_class[l].append(t)
    train, test = [], []
    for c in range(3):
        pool = by_class[c][:]
        rng.shuffle(pool)
        n_te = min(n_test // 3 + (1 if c < n_test % 3 else 0), len(pool))
        test_pool = pool[:n_te]
        train_pool = pool[n_te:]
        n_tr = min(n_train // 3 + (1 if c < n_train % 3 else 0), len(train_pool))
        train_pool = train_pool[:n_tr]
        test.extend((t, c) for t in test_pool)
        train.extend((t, c) for t in train_pool)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    train_rows, test_rows = [], []

    # EN + KZ: straightforward disjoint split
    for lang, fn, n_tr, n_te in (
        ("en", load_en, 10000, 2000),
        ("kz", load_kz, 1000, 250),
    ):
        pairs = fn()
        dist = {0: 0, 1: 0, 2: 0}
        for _, l in pairs:
            dist[l] += 1
        print(f"{lang}: total={len(pairs)}  dist(neg/neu/pos)={dist[0]}/{dist[1]}/{dist[2]}")
        tr, te = split_lang(pairs, n_tr, n_te, seed=42)
        for t, l in tr:
            train_rows.append({"text": t, "label": l, "lang": lang})
        for t, l in te:
            test_rows.append({"text": t, "label": l, "lang": lang})

    # RU: train from product + general, test from general only (real use case)
    ru_product = load_ru_product()
    ru_general = load_ru_general()
    for name, pairs in (("ru-product", ru_product), ("ru-general", ru_general)):
        dist = {0: 0, 1: 0, 2: 0}
        for _, l in pairs:
            dist[l] += 1
        print(f"{name}: total={len(pairs)}  dist(neg/neu/pos)={dist[0]}/{dist[1]}/{dist[2]}")

    ru_te = stratified_sample(ru_general, 2000, seed=1234)
    ru_te_texts = {t for t, _ in ru_te}
    ru_train_pool = ru_product + [(t, l) for t, l in ru_general if t not in ru_te_texts]
    ru_tr = stratified_sample(ru_train_pool, 20000, seed=42)
    for t, l in ru_tr:
        train_rows.append({"text": t, "label": l, "lang": "ru"})
    for t, l in ru_te:
        test_rows.append({"text": t, "label": l, "lang": "ru"})

    with open(PROC / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(PROC / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nWROTE:")
    print("  train:", len(train_rows), "rows")
    print("  test: ", len(test_rows), "rows")
    for lang in ("en", "ru", "kz"):
        d = {0: 0, 1: 0, 2: 0}
        for r in test_rows:
            if r["lang"] == lang:
                d[r["label"]] += 1
        print(f"  test {lang}: {d[0]}/{d[1]}/{d[2]}")


if __name__ == "__main__":
    main()

"""Verify two improvement resources:
1. lxyuan/distilbert-base-multilingual-cased-sentiments-student (sentiment-pretrained base)
2. clapAI/MultiLingualSentiment (does it have Russian? label format?)
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

print("=" * 70)
print("1) sentiment-student base model")
try:
    name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    print("labels:", model.config.id2label)
    n_params = sum(p.numel() for p in model.parameters())
    print("params (M):", round(n_params / 1e6, 1))
    # quick sanity prediction
    for t in ["I love this place, amazing food!", "Terrible service, never again.", "The door is blue."]:
        enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, -1).numpy()[0]
        pred = model.config.id2label[int(probs.argmax())]
        print(f"  {t[:40]:42} -> {pred}  {dict(zip(model.config.id2label, [round(x,2) for x in probs]))}")
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:300])

print()
print("=" * 70)
print("2) clapAI/MultiLingualSentiment — Russian availability")
try:
    ds = load_dataset("clapAI/MultiLingualSentiment", split="train", streaming=True)
    langs = {}
    labels = set()
    domains = {}
    n = 0
    ru_samples = []
    for row in ds:
        n += 1
        langs[row["language"]] = langs.get(row["language"], 0) + 1
        labels.add(row["label"])
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
        if row["language"] == "ru" and len(ru_samples) < 3:
            ru_samples.append((row["label"], row["domain"], row["text"][:90]))
        if n >= 20000:
            break
    print("scanned", n, "rows")
    print("labels:", labels)
    print("languages:", langs)
    print("n domains:", len(domains))
    print("RU samples:")
    for s in ru_samples:
        print("  ", s)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:300])

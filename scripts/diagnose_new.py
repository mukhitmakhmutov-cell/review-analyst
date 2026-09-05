"""Confirm the 2 new misclassifications + ablations, and verify the KZ
generator fix is live on the running server.
"""
import json
import urllib.request
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
MODEL_DIR = str(BASE / "model" / "multilingual")
CLASSES = ["positive", "neutral", "negative"]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def predict(texts):
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        return torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

cases = [
    ("RU pos (as-is)", "Персонал ресторана оперативно поменял грязные салфетки без напоминания"),
    ("RU pos (no 'грязные')", "Персонал ресторана оперативно поменял салфетки без напоминания"),
    ("EN mild pos (as-is)", "Prices at the grocery store were competitive and the checkout lines moved quickly"),
    ("EN strong pos", "Prices at the grocery store were great and the checkout lines were super fast"),
]
texts = [c[1] for c in cases]
probs = predict(texts)
print("=== 2 new cases + ablations ===")
for (name, t), p in zip(cases, probs):
    lab = CLASSES[int(p.argmax())]
    print(f"{name:22} {lab:9} pos={p[0]:.2f} neu={p[1]:.2f} neg={p[2]:.2f}  {t[:48]}")

print("\n=== /generate-example (KZ fix) ===")
req = urllib.request.Request("http://127.0.0.1:8000/generate-example", data=b"",
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    KZ = set("ұүәғқңөіһ")
    for i, rev in enumerate(data["reviews"]):
        tag = "KZ" if sum(1 for ch in rev.lower() if ch in KZ) >= 3 else ("RU" if any("\u0400" <= c <= "\u04FF" for c in rev) else "EN")
        print(f"{i+1:2}. [{tag}] {rev[:70]}")
except Exception as e:
    print("generator error:", e)

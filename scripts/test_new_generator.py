"""Test the NEW generator prompt end-to-end: generate 12 reviews (4 KZ
entertainment + 4 RU business + 4 EN business) via the LLM, then classify all
12 with the model. Verify KZ (entertainment-domain) + RU/EN (business) all
classify correctly.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
load_dotenv(BASE / ".env")
client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ.get("LLM_MODEL", "unknown")

MODEL_DIR = str(BASE / "model" / "multilingual")
CLASSES = ["positive", "neutral", "negative"]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

PROMPT = (
    "Напиши 12 отзывов клиентов, по одному в строке, без нумерации, без кавычек, "
    "без меток языков, без пояснений.\n"
    "Первые 4 — СТРОГО на казахском языке (қазақ тілінде), о кино, сериалах, музыке "
    "и концертах (развлечения). Используй казахские слова (мысалы: фильм, серия, ән, "
    "концерт, актер, сюжет, дыбыс, көрермен), НЕ на русском.\n"
    "Следующие 4 — на русском, о ресторанах, салонах и магазинах (услуги).\n"
    "Последние 4 — на английском, о ресторанах, салонах и магазинах.\n"
    "Смесь тональности: 4 негативных, 4 нейтральных, 4 позитивных, распределённых по языкам.\n"
    "Каждый отзыв 2-3 предложения, конкретный (персонал, еда/контент, цены, ожидание, чистота)."
)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=1600,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
text = resp.choices[0].message.content.strip()
reviews = [ln.strip(" \t-•*") for ln in text.splitlines() if ln.strip()]
print(f"generated {len(reviews)} reviews\n")

with torch.no_grad():
    enc = tok(reviews, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

KZ_CHARS = set("ұүәғқңөіһ")
def lang_of(t):
    if sum(1 for ch in t.lower() if ch in KZ_CHARS) >= 3:
        return "KZ"
    return "RU" if any("\u0400" <= ch <= "\u04FF" for ch in t) else "EN"

print(f"{'#':2} {'lang':3} {'pred':9} {'pos':>5} {'neu':>5} {'neg':>5}  text")
print("-" * 100)
for i, (rev, p) in enumerate(zip(reviews, probs)):
    lab = CLASSES[int(p.argmax())]
    print(f"{i+1:2} {lang_of(rev):3} {lab:9} {p[0]:5.2f} {p[1]:5.2f} {p[2]:5.2f}  {rev[:55]}")

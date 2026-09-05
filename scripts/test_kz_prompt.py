"""Test whether the LLM can be made to produce Kazakh, trying 3 prompt strategies.
Detects Kazakh by the presence of Kazakh-only Cyrillic letters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
load_dotenv(BASE / ".env")
client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ.get("LLM_MODEL", "unknown")

# Kazakh-only letters (not in Russian)
KZ_CHARS = set("ұүәғқңөіһ")

def is_kazakh(text):
    return sum(1 for ch in text.lower() if ch in KZ_CHARS) >= 3

def gen(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content.strip()

prompts = {
    "A_current": (
        "Generate 12 realistic customer reviews about different places "
        "(restaurants, salons, stores, services). Write them in THREE languages: "
        "first 4 in KAZAKH, next 4 in RUSSIAN, last 4 in ENGLISH. "
        "Overall sentiment mix: 4 negative, 4 neutral, 4 positive, "
        "spread across the languages. Each review 2-3 sentences, specific "
        "(mention staff, food, prices, wait time, cleanliness). "
        "Output one review per line, no numbering, no quotes, no language "
        "labels, no explanations."
    ),
    "B_forceful": (
        "Напиши 12 отзывов клиентов о разных местах (рестораны, салоны, магазины, услуги). "
        "ВАЖНО: первые 4 отзыва — СТРОГО на казахском языке (қазақ тілінде), используй казахские слова "
        "(мысалы: тамақ, қызмет, баға, тәтті, жақсы, нашар, күту), НЕ на русском. "
        "Следующие 4 — на русском. Последние 4 — на английском. "
        "Смесь тональности: 4 негативных, 4 нейтральных, 4 позитивных. "
        "Каждый отзыв 2-3 предложения. По одному отзыву в строке, без нумерации, без кавычек, "
        "без меток языков, без пояснений."
    ),
    "C_fewshot": (
        "Generate 12 customer reviews, one per line, no numbering/quotes/labels. "
        "First 4 MUST be in KAZAKH (қазақ тілінде) — like these examples:\n"
        "«Қызмет өте жақсы, персонал көңілді, баға қолайлы, қайта келермін.»\n"
        "«Тамақ суық болды, күту ұзақ еді, қызмет нашар, қайта бармаймын.»\n"
        "«Орын таза, бірақ баға сәл жоғары, жалпы орташа.»\n"
        "Next 4 in RUSSIAN, last 4 in ENGLISH. Sentiment mix: 4 neg, 4 neu, 4 pos spread across. "
        "Each 2-3 sentences, specific (staff, food, prices, wait, cleanliness)."
    ),
}

for name, prompt in prompts.items():
    print(f"\n===== {name} =====")
    try:
        out = gen(prompt)
        lines = [ln.strip(" \t-•*") for ln in out.splitlines() if ln.strip()]
        print(f"({len(lines)} lines)")
        for i, ln in enumerate(lines):
            tag = "KZ " if is_kazakh(ln) else "   "
            print(f"{tag}{i+1:2}. {ln[:75]}")
    except Exception as e:
        print("error:", e)

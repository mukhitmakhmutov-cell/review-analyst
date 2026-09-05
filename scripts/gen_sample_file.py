"""Use the local LLM to generate a realistic sample reviews file for testing the parser.

Produces data/sample_reviews.csv with a 'review' column and mixed sentiments,
so the user can drag it into the app and watch the file-parsing + ML + LLM flow.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
LLM_MODEL = os.environ.get("LLM_MODEL", "unknown")

prompt = (
    "Generate 12 realistic customer reviews about different places "
    "(restaurants, salons, stores, services). Mix sentiments: "
    "4 negative, 4 neutral, 4 positive. Write in ENGLISH, "
    "each review 2-3 sentences, specific (mention staff, food, "
    "prices, wait time, cleanliness). First output line must be the "
    "header 'review', then one line per review, each wrapped in double "
    "quotes. No explanations, CSV only."
)
resp = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=2000,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
text = resp.choices[0].message.content.strip()
# strip possible markdown fences
if text.startswith("```"):
    text = text.strip("`")
    if text.lower().startswith("csv"):
        text = text[3:]
    text = text.strip()

out = BASE_DIR / "data" / "sample_reviews.csv"
out.write_text(text, encoding="utf-8")
print("Saved:", out)
print("---")
print(text)

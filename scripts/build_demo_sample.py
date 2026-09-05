"""Build a curated multilingual (EN/RU/KZ) sample and verify each review
classifies correctly against the running API. Prints predictions so we can
pick a demo set that looks right. KZ reviews are pulled from the real dataset.
"""
import json
import re
import urllib.request
from datasets import load_dataset

# --- KZ: pull clearly-labeled reviews from the real dataset ---
kz = load_dataset("R3iwan/entertainment-reviews-kazakh")
split = kz[list(kz.keys())[0]]
kz_by = {"negative": [], "neutral": [], "positive": []}
for row in split:
    lab = str(row["sentiment"]).strip().lower()
    if lab not in kz_by or len(kz_by[lab]) >= 6:
        continue
    txt = row["text"]
    if isinstance(txt, (list, tuple)):
        txt = txt[0] if txt else ""
    t = str(txt).strip()
    t = re.sub(r"^\[\d+\]\s*", "", t)
    if 20 < len(t) < 200:
        kz_by[lab].append(t)

# --- candidate EN + RU (RU avoids the food 'вкусный/наваристый' pattern) ---
en = {
    "positive": "Absolutely loved this place! The food was delicious and the staff was incredibly friendly. Will definitely come back.",
    "negative": "Terrible experience. The service was slow, the food was cold, and the staff was rude. Never coming back.",
    "neutral": "The restaurant is located on the main street and is open from 9am to 9pm. Parking is available nearby.",
}
ru = {
    "positive": "Отличный сервис, вежливый персонал, всё понравилось, рекомендую всем",
    "negative": "Очень плохой сервис, ждали час, персонал грубый, больше не приду",
    "neutral": "Магазин находится в центре города, работает с 10 до 20, есть парковка рядом",
}

def api_predict(reviews):
    body = json.dumps({"reviews": reviews, "generate_report": False}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8000/analyze", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))

# assemble a candidate demo set: 1 pos + 1 neg + 1 neu per language
demo = []
for lang, src in (("en", en), ("ru", ru), ("kz", kz_by)):
    for lab in ("positive", "neutral", "negative"):
        val = src[lab]
        if isinstance(val, (list, tuple)):
            val = val[0]
        demo.append((lang, lab, val))

print("=== candidate demo set (lang, expected, text) ===")
for lang, lab, text in demo:
    print(f"[{lang}/{lab}] {text[:70]}")

print("\n=== API predictions ===")
res = api_predict([t for _, _, t in demo])
ok = 0
for (lang, lab, text), item in zip(demo, res["results"]):
    p = item["probabilities"]
    mark = "OK " if item["sentiment"] == lab else "XX "
    ok += (item["sentiment"] == lab)
    print(f"{mark}{lang:3}/{lab:9} -> {item['sentiment']:9} "
          f"pos={p.get('positive',0):.2f} neu={p.get('neutral',0):.2f} neg={p.get('negative',0):.2f}  {text[:40]}")
print(f"\n{ok}/{len(demo)} correct")

"""Diagnose the Kazakh misclassification.

1) Test the two screenshot KZ business reviews + in-domain KZ entertainment
   reviews (from the training domain) to separate 'domain mismatch' from
   'model is just bad at KZ'.
2) Call the running /generate-example endpoint and show what languages it
   actually returns (is the 'Kazakh' slot really Kazakh?).
"""
import json
import urllib.request
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\model\multilingual"
CLASSES = ["positive", "neutral", "negative"]  # head order

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def predict(texts):
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        return torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

# --- in-domain KZ: pull from the entertainment dataset (training domain) ---
kz = load_dataset("R3iwan/entertainment-reviews-kazakh")
split = kz[list(kz.keys())[0]]
indomain = {"positive": [], "neutral": [], "negative": []}
for row in split:
    lab = str(row["sentiment"]).strip().lower()
    if lab in indomain and len(indomain[lab]) < 2:
        txt = row["text"]
        if isinstance(txt, (list, tuple)):
            txt = txt[0] if txt else ""
        t = str(txt).strip()
        import re
        t = re.sub(r"^\[\d+\]\s*", "", t)
        if 20 < len(t) < 200:
            indomain[lab].append(t)

cases = []
# the two screenshot (business-domain) KZ reviews
cases.append(("KZ-biz", "POS?", "Сервисмен қыз ұсынған сусын суық болды, бірақ тағамдар тез әкеліп, дәмі өте жақсы еді. Бөлме таза болғанымен, баға сапасына қарағанда сәл қымбаттау. Жалпы алғанда қайтуға болатын орын."))
cases.append(("KZ-biz", "POS", "Бұл дүкенде киімдердің сапасы өте жоғары, әсіресе жүкпелер мен көйлектер. Кеңсеге кіргеннен бастап сапа мен тазалылық сезіледі. Тек бағалар сәл жоғары, бірақ сапа оған лайықты."))
# in-domain KZ entertainment reviews
for lab in ("positive", "neutral", "negative"):
    for t in indomain[lab][:2]:
        cases.append(("KZ-ent", lab, t))

print("=== MODEL PREDICTIONS (KZ) ===")
print(f"{'type':8} {'hint':9} {'pred':9} {'pos':>5} {'neu':>5} {'neg':>5}  text")
print("-" * 105)
texts = [c[2] for c in cases]
probs = predict(texts)
for (typ, hint, text), p in zip(cases, probs):
    lab = CLASSES[int(p.argmax())]
    print(f"{typ:8} {hint:9} {lab:9} {p[0]:5.2f} {p[1]:5.2f} {p[2]:5.2f}  {text[:55]}")

# --- generator output ---
print("\n=== /generate-example OUTPUT ===")
req = urllib.request.Request("http://127.0.0.1:8000/generate-example", data=b"",
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    for i, rev in enumerate(data["reviews"]):
        print(f"{i+1:2}. {rev[:80]}")
except Exception as e:
    print("generator error:", e)

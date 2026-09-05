"""Isolate why 'Кухня чистая и уютная, суп из говядины был очень наваристым и вкусным'
is read as negative. Test sentence fragments + paraphrases on the BASE head.
Head order: 0=positive, 1=neutral, 2=negative.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
CLASSES = ["positive", "neutral", "negative"]

texts = [
    "Кухня чистая и уютная, суп из говядины был очень наваристым и вкусным",  # full
    "Кухня чистая и уютная",                                                  # frag A
    "суп из говядины был очень наваристым и вкусным",                         # frag B
    "суп был вкусным",                                                        # simple pos
    "Кухня чистая, суп вкусный, всё понравилось",                             # paraphrase
    "Отличная кухня, вкусный суп, уютная атмосфера",                          # strong pos
    "Кухня грязная, суп невкусный",                                           # neg control
    "Наваристый суп",                                                         # rare word alone
]
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE_NAME)
model = AutoModelForSequenceClassification.from_pretrained(BASE_NAME).to(device).eval()
with torch.no_grad():
    enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

for t, p in zip(texts, probs):
    lab = CLASSES[int(p.argmax())]
    print(f"{lab:9} pos={p[0]:.2f} neu={p[1]:.2f} neg={p[2]:.2f}  {t}")

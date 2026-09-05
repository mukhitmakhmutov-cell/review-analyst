"""Diagnose WHY the multilingual model misclassifies general-domain Russian.

Tests the 3 failing screenshot reviews (salon/restaurant/store) against
in-domain clothing reviews (rureviews domain) and clearly-labeled general
Russian reviews, to separate 'domain mismatch' from 'undertraining'.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\model\multilingual"
CLASSES = ["positive", "neutral", "negative"]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

cases = [
    # (label_hint, domain, text)
    ("NEG", "salon(shot1)", "Цена за стрижку оказалась вдвое выше указанной в меню, а качество работы мастера нас разочаровало"),
    ("POS", "rest(shot2)", "Кухня чистая и уютная, суп из говядины был очень наваристым и вкусным"),
    ("POS", "store(shot3)", "Персонал магазина вежливый, быстро помог найти нужную деталь и предложил скидку"),
    ("POS", "clothing(in-domain)", "Платье отличное, ткань приятная, размер подошёл идеально, буду ещё заказывать"),
    ("NEG", "clothing(in-domain)", "Качество плохое, пошив ужасный, не стоит за такие деньги"),
    ("NEG", "general", "Очень плохой сервис, ждали час, персонал грубый, больше не приду"),
    ("POS", "general", "Всё супер, отличный сервис, вежливый персонал, рекомендую всем"),
    ("NEG", "general", "Заказ пришёл битый, продавец на меня наорал, деньги вернули только через неделю"),
    ("POS", "general", "Благодарю за отличный отдых, всё понравилось, персонал внимательный"),
]

texts = [c[2] for c in cases]
with torch.no_grad():
    enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

print(f"{'hint':4} {'domain':20} {'pred':9} {'pos':>6} {'neu':>6} {'neg':>6}  text")
print("-" * 100)
for (hint, dom, text), p in zip(cases, probs):
    pred = CLASSES[int(p.argmax())]
    mark = "OK " if pred == hint.lower() else "MISS"
    print(f"{hint:4} {dom:20} {pred:9} {p[0]:6.2f} {p[1]:6.2f} {p[2]:6.2f}  [{mark}] {text[:55]}")

"""Compare the BASE pretrained head vs the FINE-TUNED model on the 9 diagnostic
cases, to see whether fine-tuning introduced the errors or the base model has them.
Base head order (config): 0=positive, 1=neutral, 2=negative.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
FT_DIR = r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\model\multilingual"
CLASSES = ["positive", "neutral", "negative"]  # head order

cases = [
    ("NEG", "Цена за стрижку оказалась вдвое выше указанной в меню, а качество работы мастера нас разочаровало"),
    ("POS", "Кухня чистая и уютная, суп из говядины был очень наваристым и вкусным"),
    ("POS", "Персонал магазина вежливый, быстро помог найти нужную деталь и предложил скидку"),
    ("POS", "Платье отличное, ткань приятная, размер подошёл идеально, буду ещё заказывать"),
    ("NEG", "Качество плохое, пошив ужасный, не стоит за такие деньги"),
    ("NEG", "Очень плохой сервис, ждали час, персонал грубый, больше не приду"),
    ("POS", "Всё супер, отличный сервис, вежливый персонал, рекомендую всем"),
    ("NEG", "Заказ пришёл битый, продавец на меня наорал, деньги вернули только через неделю"),
    ("POS", "Благодарю за отличный отдых, всё понравилось, персонал внимательный"),
]
texts = [c[1] for c in cases]
device = "cuda" if torch.cuda.is_available() else "cpu"


def run(model, tok):
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        return torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()


base_tok = AutoTokenizer.from_pretrained(BASE_NAME)
base = AutoModelForSequenceClassification.from_pretrained(BASE_NAME).to(device).eval()
ft_tok = AutoTokenizer.from_pretrained(FT_DIR)
ft = AutoModelForSequenceClassification.from_pretrained(FT_DIR).to(device).eval()

bp = run(base, base_tok)
fp = run(ft, ft_tok)

print(f"{'hint':4} {'base':9} {'ft':9} {'base pos/neu/neg':18} {'ft pos/neu/neg':18}  text")
print("-" * 110)
for (hint, text), b, f in zip(cases, bp, fp):
    bl = CLASSES[int(b.argmax())]
    fl = CLASSES[int(f.argmax())]
    bmark = "ok" if bl == hint.lower() else "XX"
    fmark = "ok" if fl == hint.lower() else "XX"
    print(f"{hint:4} {bl:4}[{bmark}] {fl:4}[{fmark}]  "
          f"{b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}        {f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}   {text[:45]}")

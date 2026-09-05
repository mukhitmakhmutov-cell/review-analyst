"""Inspect the saved model's label config and raw predictions to find the
label-ordering mismatch between the trained head and config.id2label."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\model\multilingual"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

print("config.num_labels:", model.config.num_labels)
print("config.id2label:  ", model.config.id2label)
print("config.label2id:  ", model.config.label2id)
print()

cases = [
    ("EN pos", "I absolutely love this place, the food was amazing and staff was great!"),
    ("EN neg", "Terrible experience, rude staff, cold food, never coming back."),
    ("EN neu", "The restaurant is located on the main street, open from 9 to 9."),
    ("RU pos(shot2)", "Кухня чистая и уютная, суп из говядины был очень наваристым и вкусным"),
    ("RU neg(shot1)", "Цена за стрижку оказалась вдвое выше указанной в меню, а качество работы мастера нас разочаровало"),
]
texts = [c[1] for c in cases]
with torch.no_grad():
    enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, -1).cpu().numpy()

print("raw logits (order = neuron index):")
for (name, _), lg, pr in zip(cases, logits.cpu().numpy(), probs):
    print(f"  {name:14} logits={ [round(x,2) for x in lg] }  probs={ [round(x,2) for x in pr] }  argmax={int(pr.argmax())}")

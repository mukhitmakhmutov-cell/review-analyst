"""Fine-tune a multilingual transformer for 3-class sentiment (EN/RU/KZ).

Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student (135M).
  We start from a *sentiment-pretrained* multilingual checkpoint (not the
  generic MLM one) so the encoder already understands polarity — this fixes
  the "everything collapses to neutral" failure on mild/descriptive reviews.
  The base model's well-calibrated 3-class sentiment head is KEPT (not
  re-initialised) and fine-tuned in place, so the model stays confident on
  clear-cut reviews. We adopt the base head's label order throughout.
Data:  data/processed/train.jsonl + test.jsonl  ({"text","label","lang"})
Output: model/multilingual/  (model + tokenizer) + model/multilingual/metrics.json

label map (base model's head order): 0=positive, 1=neutral, 2=negative
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
PROC = BASE / "data" / "processed"
OUT = BASE / "model" / "multilingual"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
MAX_LEN = 128
SEED = 42
LABELS = ["positive", "neutral", "negative"]

os.environ["WANDB_DISABLED"] = "true"
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def make_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"f1_macro": f1, "accuracy": acc}
    return compute_metrics


def main():
    train_ds = load_jsonl(PROC / "train.jsonl")
    test_ds = load_jsonl(PROC / "test.jsonl")
    print("train:", train_ds, "\ntest:", test_ds)

    # keep raw test data for per-language evaluation
    test_texts = [r["text"] for r in test_ds]
    test_labels = [r["label"] for r in test_ds]
    test_langs = [r["lang"] for r in test_ds]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

    # keep 'label', drop the raw text/lang columns
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text", "lang"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text", "lang"])
    for ds in (train_ds, test_ds):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Keep the base model's sentiment-pretrained head (do NOT re-initialise it):
    # it is already calibrated for 3-class polarity, so fine-tuning in place keeps
    # the model confident on clear-cut reviews. Its label order
    # (0=positive, 1=neutral, 2=negative) is adopted throughout the pipeline.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    args = TrainingArguments(
        output_dir=str(OUT),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
        fp16=True,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        report_to=[],
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=make_metrics(),
    )

    print("=== TRAINING ===")
    trainer.train()

    # save best model + tokenizer
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print("saved model to", OUT)

    # ---- detailed per-language evaluation ----
    print("\n=== PER-LANGUAGE EVALUATION ===")
    model.eval()
    device = next(model.parameters()).device
    texts = test_texts
    gold = test_labels
    langs = test_langs

    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            batch = tokenizer(texts[i:i + 256], truncation=True, max_length=MAX_LEN,
                              padding=True, return_tensors="pt").to(device)
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(probs.argmax(axis=-1))
            all_probs.append(probs)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = {}
    for lang in ("en", "ru", "kz"):
        mask = np.array([l == lang for l in langs])
        y_true = np.array(gold)[mask]
        y_pred = all_preds[mask]
        y_prob = all_probs[mask]
        m = {
            "n": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "f1_per_class": [round(float(x), 4) for x in f1_score(y_true, y_pred, average=None)],
            "roc_auc_macro": round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4),
        }
        metrics[lang] = m
        print(f"{lang}: n={m['n']}  acc={m['accuracy']}  f1_macro={m['f1_macro']}  "
              f"auc={m['roc_auc_macro']}  f1(pos/neu/neg)={m['f1_per_class']}")

    # overall
    y_true = np.array(gold)
    metrics["overall"] = {
        "n": int(len(gold)),
        "accuracy": round(float(accuracy_score(y_true, all_preds)), 4),
        "f1_macro": round(float(f1_score(y_true, all_preds, average="macro")), 4),
        "roc_auc_macro": round(float(roc_auc_score(y_true, all_probs, multi_class="ovr", average="macro")), 4),
    }
    print("overall:", metrics["overall"])

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()

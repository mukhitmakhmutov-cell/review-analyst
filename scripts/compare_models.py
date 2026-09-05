import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\data\raw\yelp_reviews.csv")
X = df["review"]; y = df["sentiment"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def report(name, model, proba=True):
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_t = time.time() - t0
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    if proba and hasattr(model, "predict_proba"):
        auc = roc_auc_score(yte, model.predict_proba(Xte), multi_class="ovr")
    else:
        auc = float("nan")
    print(f"{name:32s} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} fit={fit_t:.1f}s")
    return f1

print("=== Baseline candidates ===")
report("NB unigram (baseline)", Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, max_features=50_000)),
    ("clf", MultinomialNB()),
]))
report("LogReg unigram C=1", Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, max_features=50_000)),
    ("clf", LogisticRegression(max_iter=1000, C=1.0)),
]))

print("=== Improved candidates ===")
report("LogReg bigram C=4", Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50_000)),
    ("clf", LogisticRegression(max_iter=1000, C=4.0)),
]))
report("LogReg bigram C=8", Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50_000)),
    ("clf", LogisticRegression(max_iter=1000, C=8.0)),
]))
report("LinearSVC bigram C=0.5 (calib)", Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50_000)),
    ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=2000), cv=3, method="sigmoid")),
]))

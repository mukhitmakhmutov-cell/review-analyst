import time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

NEGATION_WORDS = {
    "not", "no", "never", "nothing", "nobody", "without", "dont", "don't",
    "cant", "can't", "wont", "won't", "isnt", "isn't", "wasnt", "wasn't",
    "hardly", "barely",
}


def build_features(texts):
    t = texts.fillna("")
    words = t.str.split()
    lower = t.str.lower()
    return pd.DataFrame({
        "text_length": t.str.len(),
        "word_count": words.str.len(),
        "avg_word_length": t.str.len() / words.str.len().clip(lower=1),
        "exclamation_count": t.str.count("!"),
        "question_count": t.str.count(r"\?"),
        "uppercase_ratio": t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
        "negation_count": lower.apply(lambda s: sum(w.strip(".,!?\"'") in NEGATION_WORDS for w in s.split())),
        "has_numbers": t.str.contains(r"\d").astype(int),
    })


class CombinedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf, scaler):
        self.tfidf = tfidf
        self.scaler = scaler

    def fit(self, X, y=None):
        texts = pd.Series(X)
        self.tfidf.fit(texts)
        self.scaler.fit(build_features(texts))
        return self

    def transform(self, X):
        texts = pd.Series(X)
        tf = self.tfidf.transform(texts)
        hf = self.scaler.transform(build_features(texts)).astype(np.float32)
        return sparse.hstack([tf, hf])


df = pd.read_csv(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\data\raw\yelp_reviews.csv")
X = df["review"]; y = df["sentiment"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = [
    ("feats", CombinedTransformer(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50_000),
        StandardScaler(),
    )),
]
Xtr_t = pipe[0][1].fit_transform(Xtr)
Xte_t = pipe[0][1].transform(Xte)
print("feature matrix:", Xtr_t.shape)

le = LabelEncoder().fit(ytr)
t0 = time.time()
m = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.9,
    colsample_bytree=0.8, eval_metric="mlogloss", tree_method="hist",
    device="cuda", random_state=42, n_jobs=4,
)
m.fit(Xtr_t, le.transform(ytr))
print("GPU fit 300 trees + features:", round(time.time() - t0, 1), "s")
p = m.predict_proba(Xte_t)
pred = le.inverse_transform(m.predict(Xte_t))
print("acc", round(accuracy_score(yte, pred), 4))
print("f1macro", round(f1_score(yte, pred, average="macro"), 4))
print("rocauc", round(roc_auc_score(yte, p, multi_class="ovr"), 4))
print("confusion:"); print(confusion_matrix(yte, pred))

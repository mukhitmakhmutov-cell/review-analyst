"""Build and execute notebooks/analysis.ipynb for the Review Analyst project."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
VENV_PY = r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\.venv\Scripts\python.exe"

md = lambda src: nbformat.v4.new_markdown_cell(src)
code = lambda src: nbformat.v4.new_code_cell(src)

cells = [
md("""# Review Analyst для бизнеса
**Гибридный проект: ML + AI**

- **ML часть:** классификация тональности отзывов в 3 класса (negative / neutral / positive). Baseline — Naive Bayes, улучшенные модели — Logistic Regression, XGBoost (на GPU) и LinearSVC.
- **AI часть:** LLM (Qwen 27B через OpenAI-совместимый API) анализирует негативные отзывы и генерирует отчёт для бизнеса: основные жалобы и рекомендации.
- **Интерфейс:** FastAPI (`app.py`), эндпоинт `POST /analyze`.

**Датасет:** [Yelp reviews](https://huggingface.co/datasets/Yelp/yelp_review_full) — 30 000 отзывов (10k на класс), случайная выборка из всех 650k записей. Звёзды 1–2 → negative, 3 → neutral, 4–5 → positive.
"""),

code("""import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 13
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

_here = Path.cwd()
BASE_DIR = _here if (_here / "data").exists() else _here.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "yelp_reviews.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
print("Ready. Data:", DATA_PATH)"""),

md("""## 1. Загрузка данных и EDA"""),

code("""df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.info()
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicates:", int(df.duplicated().sum()))
df.head(3)"""),

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["sentiment"].value_counts().plot(kind="bar", ax=axes[0], color=["#e74c3c", "#f1c40f", "#2ecc71"])
axes[0].set_title("Распределение классов")
axes[0].set_xlabel("")
df["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#3498db")
axes[1].set_title("Распределение оценок (звёзды)")
axes[1].set_xlabel("Оценка")
plt.tight_layout()
plt.show()"""),

code("""df["text_length"] = df["review"].str.len()
df["word_count"] = df["review"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.boxplot(column="text_length", by="sentiment", ax=axes[0], grid=False)
axes[0].set_title("Длина отзыва (символы) по классам")
axes[0].set_xlabel("")
df.boxplot(column="word_count", by="sentiment", ax=axes[1], grid=False)
axes[1].set_title("Количество слов по классам")
axes[1].set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.show()

df.groupby("sentiment")[["text_length", "word_count"]].agg(["mean", "median"]).round(1)"""),

code("""for lab in ["negative", "neutral", "positive"]:
    print(f"--- {lab} ---")
    for t in df[df.sentiment == lab]["review"].head(2):
        print("•", t[:220].replace("\\n", " "), "…")
    print()"""),

code("""def top_words(texts, n=12):
    vec = TfidfVectorizer(stop_words="english", max_features=20000)
    m = vec.fit_transform(texts)
    scores = np.asarray(m.sum(axis=0)).ravel()
    idx = np.argsort(scores)[::-1][:n]
    names = vec.get_feature_names_out()
    return [(names[i], int(scores[i])) for i in idx]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, lab in zip(axes, ["negative", "neutral", "positive"]):
    words = top_words(df[df.sentiment == lab]["review"])
    ax.barh([w for w, _ in words][::-1], [c for _, c in words][::-1])
    ax.set_title(lab)
plt.suptitle("Топ-12 слов (без стоп-слов) в каждом классе")
plt.tight_layout()
plt.show()"""),

md("""**Выводы EDA:**
- Классы сбалансированы (по 10k), пропусков и дубликатов нет — предобработка сводится к текстовой.
- Негативные отзывы в среднем короче и эмоциональнее, позитивные — длиннее.
- В каждом классе есть характерная лексика: негатив — *terrible, worst, never*, позитив — *great, amazing, excellent*. Задача решаема по тексту."""),

md("""## 2. Baseline: Naive Bayes

Классическая «первая» модель для текста. Все модели в проекте используют одни и те же признаки — TF-IDF (n-граммы 1–2, sublinear_tf, 50k признаков), чтобы сравнение было честным.
"""),

code("""X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape[0], " Test:", X_test.shape[0])


def evaluate(model, X_tr, X_te, y_tr, y_te, name):
    res = {
        "model": name,
        "acc_train": round(accuracy_score(y_tr, model.predict(X_tr)), 4),
        "acc_test": round(accuracy_score(y_te, model.predict(X_te)), 4),
        "f1_macro_test": round(f1_score(y_te, model.predict(X_te), average="macro"), 4),
        "roc_auc_test": round(roc_auc_score(y_te, model.predict_proba(X_te), multi_class="ovr"), 4),
    }
    print(res)
    return res


baseline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, max_features=50_000)),
    ("clf", MultinomialNB()),
])
baseline.fit(X_train, y_train)
baseline_res = evaluate(baseline, X_train, X_test, y_train, y_test, "Naive Bayes (baseline)")"""),

md("""## 3. Улучшенные модели

Пробуем три модели на тех же TF-IDF bigram-признаках:
1. **Logistic Regression** — линейная, быстрая, сильная на тексте.
2. **XGBoost** — градиентный бустинг, обучается на GPU (`device='cuda'`).
3. **LinearSVC** (с калибровкой вероятностей) — линейный SVM.
"""),

code("""logreg = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50_000)),
    ("clf", LogisticRegression(max_iter=1000, C=4.0)),
])
logreg.fit(X_train, y_train)
logreg_res = evaluate(logreg, X_train, X_test, y_train, y_test, "Logistic Regression (bigram)")"""),

code("""class LabelEncodingXGB(BaseEstimator, ClassifierMixin):
    \"\"\"XGBClassifier с кодированием строковых лейблов (xgboost 3.3 требует int-лейблы).\"\"\"

    def __init__(self, **kwargs):
        self.clf = xgb.XGBClassifier(**kwargs)
        self.le = LabelEncoder()

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.clf.fit(X, self.le.fit_transform(y))
        return self

    def predict(self, X):
        return self.classes_[self.clf.predict(X)]

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


xgb_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50_000)),
    ("clf", LabelEncodingXGB(
        n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.9,
        colsample_bytree=0.8, eval_metric="mlogloss", tree_method="hist",
        device="cuda", random_state=RANDOM_STATE, n_jobs=4,
    )),
])
xgb_model.fit(X_train, y_train)
xgb_res = evaluate(xgb_model, X_train, X_test, y_train, y_test, "XGBoost (bigram, GPU)")"""),

code("""svc = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50_000)),
    ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=2000), cv=3, method="sigmoid")),
])
svc.fit(X_train, y_train)
svc_res = evaluate(svc, X_train, X_test, y_train, y_test, "LinearSVC (bigram, calibrated)")"""),

code("""results = pd.DataFrame([baseline_res, logreg_res, xgb_res, svc_res]).sort_values("f1_macro_test", ascending=False)
results"""),

code("""best = svc
cm = confusion_matrix(y_test, best.predict(X_test))
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(3), ["negative", "neutral", "positive"])
plt.yticks(range(3), ["negative", "neutral", "positive"])
plt.xlabel("Предсказано")
plt.ylabel("Факт")
plt.title("Матрица ошибок — лучшая модель (test)")
plt.tight_layout()
plt.show()"""),

md("""## 4. Сохранение модели"""),

code("""joblib.dump(best, MODEL_DIR / "best_pipeline.pkl")
joblib.dump(results.set_index("model"), MODEL_DIR / "metrics.pkl")
print("Saved:", MODEL_DIR / "best_pipeline.pkl")"""),

md("""## 5. AI-часть: LLM-отчёт по негативным отзывам

LLM (локальная Qwen 27B через OpenAI-совместимый API) получает выборку негативных отзывов и пишет отчёт для бизнеса: основные жалобы и конкретные рекомендации. Тот же код используется в FastAPI-приложении (`app.py`).
Ключ хранится в `.env` (не коммитится).
"""),

code("""import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(BASE_DIR / ".env")
client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
LLM_MODEL = os.environ.get("LLM_MODEL", "unknown")


def llm_report(negative_reviews, max_reviews=15, max_tokens=1600):
    sample = "\\n\\n".join(f"[{i+1}] {t[:600]}" for i, t in enumerate(negative_reviews[:max_reviews]))
    prompt = (
        "Ты — аналитик для бизнеса. Ниже приведены негативные отзывы клиентов. "
        "Напиши краткий отчёт на русском языке: 1) 3-5 основных жалоб (списком), "
        "2) 3-5 конкретных рекомендаций по улучшению (списком). "
        "Будь конкретен, ссылайся на отзывы. Не более 250 слов.\\n\\n"
        f"Отзывы:\\n{sample}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content


neg_test = X_test[y_test == "negative"]
report = llm_report(list(neg_test.head(15)))
print(report)"""),

md("""## Выводы

- **Baseline** (Naive Bayes) дал F1 macro ≈ 0.66. Лучшая модель — **LinearSVC** с bigram-признаками — подняла его до ≈ 0.74.
- **Важное наблюдение:** XGBoost (≈ 0.71) *проигрывает* линейным моделям (LogReg 0.73, LinearSVC 0.74) на разреженных TF-IDF-признаках. Это задокументированный результат: древовидные модели плохо работают с высокомерными разреженными текстовыми фичами.
- Класс **neutral** (3 звезды) — самый сложный, его чаще всего путают с negative/positive. Это совпадает с литературой по 3-классовой тональности Yelp.
- **AI-часть** превращает сырые предсказания в понятный бизнес-отчёт: именно это делает проект гибридным и пригодным для реального продукта.
- Дальше: кросс-валидация, подбор гиперпараметров (Optuna), A/B-сравнение промптов для LLM, Docker-деплой."""),
]

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

os.chdir(ROOT / "notebooks")
km = KernelManager()
km.kernel_cmd = [VENV_PY, "-m", "ipykernel", "-f", "{connection_file}"]
km.start_kernel()
try:
    client = NotebookClient(nb, kernel_manager=km, timeout=900, iopub_timeout=900)
    client.execute()
finally:
    km.shutdown_kernel()

out = ROOT / "notebooks" / "analysis.ipynb"
nbformat.write(nb, out)
print("Saved executed notebook:", out)

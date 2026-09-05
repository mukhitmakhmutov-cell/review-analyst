"""Build the 5-minute presentation (presentation/presentation.pptx).

Follows the course template from final_project.ipynb:
  1. Название и проблема
  2. Данные
  3. Подход и модели
  4. Демо / интерфейс
  5. Выводы и что дальше
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "presentation" / "presentation.pptx"
OUT.parent.mkdir(exist_ok=True)
SHOTS = BASE_DIR / "screenshots"

# palette (matches the app)
ACC = RGBColor(0x4F, 0x46, 0xE5)
ACC2 = RGBColor(0x7C, 0x3A, 0xED)
INK = RGBColor(0x17, 0x1C, 0x2B)
MUTED = RGBColor(0x6B, 0x72, 0x80)
BG = RGBColor(0xF4, 0xF6, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NEG = RGBColor(0xE7, 0x4C, 0x3C)
NEU = RGBColor(0xF1, 0xC4, 0x0F)
POS = RGBColor(0x2E, 0xCC, 0x71)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def add_bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: list of paragraphs; each paragraph is list of (text, size, color, bold)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for (text, size, color, bold) in para:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = "Segoe UI"
    return tb


def header(slide, title, subtitle=None):
    add_rect(slide, 0, 0, SW, Inches(1.15), ACC)
    add_text(slide, Inches(0.6), Inches(0.18), Inches(12), Inches(0.8),
             [[(title, 30, WHITE, True)]], anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(slide, Inches(0.6), Inches(1.25), Inches(12), Inches(0.5),
                 [[(subtitle, 15, MUTED, False)]])


# ---------------- Slide 1: Название и проблема ----------------
s = prs.slides.add_slide(BLANK)
add_bg(s, WHITE)
add_rect(s, 0, 0, SW, Inches(2.6), ACC)
add_text(s, Inches(0.7), Inches(0.55), Inches(12), Inches(1.0),
         [[("Review Analyst", 44, WHITE, True)]])
add_text(s, Inches(0.7), Inches(1.55), Inches(12), Inches(0.9),
         [[("Гибридный проект ML + AI: анализ тональности отзывов", 20, WHITE, False)]])
add_text(s, Inches(0.7), Inches(3.0), Inches(12), Inches(3.5), [
    [("Проблема", 22, ACC, True)],
    [("", 6, INK, False)],
    [("Бизнес получает тысячи отзывов, но не может быстро понять, ", 17, INK, False),
     ("что именно не так", 17, NEG, True),
     (" и что делать.", 17, INK, False)],
    [("", 6, INK, False)],
    [("Решение", 22, ACC, True)],
    [("", 6, INK, False)],
    [("ML-модель классифицирует каждый отзыв (негатив / нейтрал / позитив), ", 17, INK, False),
     ("а LLM превращает негатив в готовый бизнес-отчёт", 17, ACC2, True),
     (" с конкретными рекомендациями.", 17, INK, False)],
])
add_text(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.5),
         [[("Data Science & AI · Итоговый проект · 2026", 13, MUTED, False)]])

# ---------------- Slide 2: Данные ----------------
s = prs.slides.add_slide(BLANK)
add_bg(s)
header(s, "Данные", "Yelp reviews (HuggingFace) · 3 класса тональности")
# stat cards
cards = [
    ("30 000", "отзывов", ACC),
    ("3", "класса", ACC2),
    ("10 000", "на класс (баланс)", POS),
    ("EN", "язык отзывов", MUTED),
]
cx = Inches(0.7)
for val, lab, col in cards:
    add_rect(s, cx, Inches(2.0), Inches(2.9), Inches(1.5), WHITE, line=RGBColor(0xE5, 0xE8, 0xF0))
    add_text(s, cx, Inches(2.15), Inches(2.9), Inches(0.8),
             [[(val, 30, col, True)]], align=PP_ALIGN.CENTER)
    add_text(s, cx, Inches(2.95), Inches(2.9), Inches(0.5),
             [[(lab, 13, MUTED, False)]], align=PP_ALIGN.CENTER)
    cx += Inches(3.05)
add_text(s, Inches(0.7), Inches(3.9), Inches(12), Inches(3.0), [
    [("Ключевые наблюдения из EDA", 20, ACC, True)],
    [("", 6, INK, False)],
    [("•  1–2★ → негатив,  3★ → нейтрал,  4–5★ → позитив (3-классовая задача)", 16, INK, False)],
    [("•  Длина отзывов: медиана ~130 слов, есть очень короткие и очень длинные", 16, INK, False)],
    [("•  Классы сбалансированы → можно оптимизировать F1 macro без весов", 16, INK, False)],
    [("•  Случайная выборка из 650k — чтобы не было кластеров по бизнесам", 16, INK, False)],
])

# ---------------- Slide 3: Подход и модели ----------------
s = prs.slides.add_slide(BLANK)
add_bg(s)
header(s, "Подход и модели", "Baseline → улучшение → сравнение")
# pipeline
add_rect(s, Inches(0.7), Inches(1.9), Inches(12), Inches(0.9), WHITE, line=RGBColor(0xE5, 0xE8, 0xF0))
add_text(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(0.7),
         [[("Текст  →  TF-IDF (биграммы)  →  Классификатор  →  вероятности по 3 классам", 16, INK, True)]],
         anchor=MSO_ANCHOR.MIDDLE)
# metrics table
rows = [
    ("Модель", "F1 macro", "ROC AUC", True),
    ("Naive Bayes (униграммы, baseline)", "0.661", "0.843", False),
    ("XGBoost (биграммы, GPU)", "0.706", "0.872", False),
    ("Logistic Regression (биграммы)", "0.735", "0.891", False),
    ("LinearSVC (биграммы, calibrated)", "0.744", "0.894", True),
]
ty = Inches(3.1)
row_h = Inches(0.62)
for i, (m, f1, auc, best) in enumerate(rows):
    bg = RGBColor(0xEE, 0xF0, 0xFF) if best else (WHITE if i % 2 else RGBColor(0xFB, 0xFB, 0xFE))
    add_rect(s, Inches(0.7), ty, Inches(12), row_h, bg,
             line=ACC if best else RGBColor(0xE5, 0xE8, 0xF0))
    add_text(s, Inches(0.9), ty, Inches(7.5), row_h,
             [[(m, 15, ACC if best else INK, best)]], anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(8.6), ty, Inches(1.8), row_h,
             [[(f1, 15, ACC if best else INK, best)]], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(10.6), ty, Inches(1.8), row_h,
             [[(auc, 15, ACC if best else INK, best)]], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    ty += row_h
add_text(s, Inches(0.7), Inches(6.5), Inches(12), Inches(0.8), [
    [("Вывод: ", 15, ACC, True),
     ("на разреженных TF-IDF линейные модели обгоняют градиентный бустинг — ", 15, INK, False),
     ("XGBoost переобучается на высокоразмерных бинарных признаках.", 15, INK, False)],
])

# ---------------- Slide 4: Демо / интерфейс ----------------
s = prs.slides.add_slide(BLANK)
add_bg(s)
header(s, "Демо", "FastAPI + веб-интерфейс · загрузка файла → ML → LLM-отчёт")
# left: ui screenshot
s.shapes.add_picture(str(SHOTS / "ui.png"), Inches(0.5), Inches(1.7), height=Inches(5.4))
# right: demo screenshot (taller, crop width)
s.shapes.add_picture(str(SHOTS / "demo.png"), Inches(4.6), Inches(1.55), height=Inches(5.7))
add_text(s, Inches(9.0), Inches(1.9), Inches(3.9), Inches(5.0), [
    [("Как это работает", 18, ACC, True)],
    [("", 6, INK, False)],
    [("1.  Пользователь вставляет отзывы или перетаскивает файл (.txt/.csv/.jsonl/.json)", 14, INK, False)],
    [("", 4, INK, False)],
    [("2.  ML-модель классифицирует каждый отзыв и даёт вероятности", 14, INK, False)],
    [("", 4, INK, False)],
    [("3.  LLM читает негатив и пишет отчёт: жалобы + рекомендации", 14, INK, False)],
    [("", 4, INK, False)],
    [("4.  Всё рендерится в браузере (Markdown)", 14, INK, False)],
])

# ---------------- Slide 5: Выводы и что дальше ----------------
s = prs.slides.add_slide(BLANK)
add_bg(s)
header(s, "Выводы и что дальше")
add_text(s, Inches(0.7), Inches(1.7), Inches(12), Inches(5.2), [
    [("Что получилось", 20, ACC, True)],
    [("", 5, INK, False)],
    [("•  Рабочая модель: F1 macro 0.744, ROC AUC 0.894 (лучшая из 4)", 16, INK, False)],
    [("•  Гибридный продукт: ML-классификация + LLM-отчёт на русском", 16, INK, False)],
    [("•  Веб-интерфейс с загрузкой файлов и валидацией", 16, INK, False)],
    [("", 8, INK, False)],
    [("Что было сложно", 20, ACC, True)],
    [("", 5, INK, False)],
    [("•  LLM — thinking-модель: без отключения «размышлений» content приходит пустым", 16, INK, False)],
    [("•  XGBoost на GPU: пришлось обёртывать метки (int) и подбирать параметры", 16, INK, False)],
    [("", 8, INK, False)],
    [("Что улучшил бы при большем времени", 20, ACC, True)],
    [("", 5, INK, False)],
    [("•  Обучить модель на русскоязычных отзывах (сейчас ML — только EN)", 16, INK, False)],
    [("•  Fine-tuning небольшой трансформер для более высокой точности", 16, INK, False)],
    [("•  RAG: LLM ссылалась бы на базу знаний о заведении", 16, INK, False)],
])

prs.save(OUT)
print("saved", OUT)

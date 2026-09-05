"""Capture screenshots of the running Review Analyst app using headless Edge.

Produces:
  screenshots/ui.png    - the input interface (before analysis)
  screenshots/demo.png  - full results: summary, per-review cards, LLM report
"""
import csv
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE = BASE_DIR / "data" / "sample_reviews.csv"
OUT_DIR = BASE_DIR / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

reviews = []
with open(SAMPLE, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        reviews.append(r["review"].strip())
print("loaded", len(reviews), "reviews")

URL = "http://127.0.0.1:8000"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(600)

    # 1) input interface
    page.screenshot(path=str(OUT_DIR / "ui.png"), full_page=True)
    print("saved ui.png")

    # 2) load reviews + run analysis
    page.fill("#in", "\n".join(reviews))
    page.wait_for_timeout(300)
    page.click("#go")
    page.wait_for_function(
        "() => { const s=document.getElementById('status').textContent;"
        " const o=document.getElementById('out').innerHTML;"
        " return s.includes('Готово') || o.includes('class=\"err\"'); }",
        timeout=90000,
    )
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT_DIR / "demo.png"), full_page=True)
    print("saved demo.png")

    browser.close()
print("done")

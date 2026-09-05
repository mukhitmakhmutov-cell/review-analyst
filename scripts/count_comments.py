import re
from pathlib import Path

base = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst")
for f in [
    "app.py",
    "scripts/download_data.py",
    "scripts/compare_models.py",
    "scripts/build_notebook.py",
    "scripts/gen_sample_file.py",
]:
    p = base / f
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    code = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    comments = [l for l in lines if l.strip().startswith("#")]
    docstrings = text.count('"""') // 2
    print(f"{f:32} lines={len(lines):4} code={len(code):4} #comments={len(comments):3} docstrings={docstrings}")

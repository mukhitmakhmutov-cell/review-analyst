import json

nb = json.load(open(
    r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\notebooks\analysis.ipynb",
    encoding="utf-8",
))
for idx in (8, 21):
    print("=" * 70)
    print(f"CELL {idx} (markdown):")
    print("=" * 70)
    print("".join(nb["cells"][idx]["source"]))
    print()

"""Remap label order in the existing train/test JSONL from
0=negative,1=neutral,2=positive  ->  0=positive,1=neutral,2=negative.

new_label = 2 - old_label  (negative<->positive swap, neutral unchanged).
In-place; backs up the originals first.
"""
import json
from pathlib import Path

PROC = Path(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\data\processed")

for name in ("train.jsonl", "test.jsonl"):
    p = PROC / name
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    for r in rows:
        r["label"] = 2 - r["label"]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = {0: 0, 1: 0, 2: 0}
    for r in rows:
        dist[r["label"]] += 1
    print(f"{name}: {len(rows)} rows  dist(pos/neu/neg)={dist[0]}/{dist[1]}/{dist[2]}")

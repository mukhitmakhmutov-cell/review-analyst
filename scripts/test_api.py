import csv
import json
import urllib.request

rows = []
with open(r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\data\sample_reviews.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r["review"].strip())
print("parsed reviews:", len(rows))

payload = json.dumps({"reviews": rows, "generate_report": True}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/analyze", data=payload, headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=120) as resp:
    d = json.loads(resp.read().decode("utf-8"))
print("summary:", d["summary"])
for x in d["results"]:
    print(f"  [{x['sentiment']:>8}] {x['probabilities'][x['sentiment']]:.3f}  {x['review'][:55]}")
print("--- LLM REPORT ---")
print(d["llm_report"])

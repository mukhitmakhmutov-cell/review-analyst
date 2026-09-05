"""POST the 3 screenshot reviews to the running app and print ML predictions."""
import json
import urllib.request

reviews = [
    "Цена за стрижку оказалась вдвое выше указанной в меню, а качество работы мастера нас разочаровало",
    "Кухня чистая и уютная, суп из говядины был очень наваристым и вкусным",
    "Персонал магазина вежливый, быстро помог найти нужную деталь и предложил скидку",
]
body = json.dumps({"reviews": reviews, "generate_report": False}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/analyze",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as r:
    data = json.loads(r.read().decode("utf-8"))

for item in data["results"]:
    p = item["probabilities"]
    print(f"pred={item['sentiment']:9} pos={p.get('positive',0):.2f} neu={p.get('neutral',0):.2f} "
          f"neg={p.get('negative',0):.2f}  {item['review'][:50]}")
print("\nsummary:", data["summary"])

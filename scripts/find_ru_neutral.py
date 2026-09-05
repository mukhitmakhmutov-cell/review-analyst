"""Find a Russian NEUTRAL review the model classifies as neutral."""
import json
import urllib.request

def api_predict(reviews):
    body = json.dumps({"reviews": reviews, "generate_report": False}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8000/analyze", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))

cands = [
    "Магазин расположен в центре города. Часы работы: с 10:00 до 20:00.",
    "Услуга оказана в срок. Менеджер ответил на все вопросы.",
    "Заказ оформлен онлайн, доставка заняла два дня.",
    "Офис находится на втором этаже, вход со двора.",
    "Приложение работает стабильно, обновлений за месяц не было.",
]
res = api_predict(cands)
for t, item in zip(cands, res["results"]):
    p = item["probabilities"]
    mark = "OK " if item["sentiment"] == "neutral" else "XX "
    print(f"{mark}{item['sentiment']:9} pos={p.get('positive',0):.2f} neu={p.get('neutral',0):.2f} "
          f"neg={p.get('negative',0):.2f}  {t}")

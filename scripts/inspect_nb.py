import json

nb = json.load(open(
    r"C:\Users\79021\OneDrive - РУТ (МИИТ)\Рабочий стол\mukhit\review_analyst\notebooks\analysis.ipynb",
    encoding="utf-8",
))
print("total cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]).strip()
    first = src.split("\n")[0][:78]
    if c["cell_type"] == "markdown":
        print(f"{i:2d} MD   | {first}")
    else:
        has_out = any(
            o.get("output_type") in ("display_data", "stream", "execute_result")
            for o in c.get("outputs", [])
        )
        print(f"{i:2d} CODE | {first}  [out:{'Y' if has_out else 'n'}]")

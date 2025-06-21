import json
from collections import OrderedDict

with open("WLASL_v0.3.json") as f:
    data = json.load(f)

with open("WLASL_v0.3.json") as f:
    data = json.load(f)

actions = list(OrderedDict.fromkeys(entry["gloss"] for entry in data))[:50]
print(actions)
print(len(actions))

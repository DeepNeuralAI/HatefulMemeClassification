import jsonlines
import json

with open("csvjson.json", 'r', encoding='utf-8') as jsonf:
    data = json.load(jsonf)

with jsonlines.open('train_miso.jsonl', 'w') as writer:
    writer.write_all(data)
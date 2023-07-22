import json

with open('dataset_infos.json') as f:
    text = json.load(f)
    with open('new.json', 'w')as fw:
        json.dump(text, fw, indent=4)
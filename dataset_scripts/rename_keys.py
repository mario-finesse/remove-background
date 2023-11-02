import json

captions_path = "data/finesse_kp_dataset.json"
with open(captions_path, 'r') as json_file:
    data = json.load(json_file)

captions = {}
for img_path, caption in data.items():
    pair_name = img_path.split("/")[-1].replace(".png", "")
    captions[pair_name] = caption

with open("data/finesse_captions.json", 'w') as file:
    json.dump(captions, file)
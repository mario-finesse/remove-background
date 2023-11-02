import json
import os

captions_path = "data/captions/finesse_captions.json"
with open(captions_path, 'r') as json_file:
    captions = json.load(json_file)

finesse_kp_background_dataset = {}
images_background_path = "/data/clogen/ed_datasets/diff_backgrounds"
for background in os.listdir(images_background_path):
    images_path = os.path.join(images_background_path, background)
    png_files = [file for file in os.listdir(images_path) if file.endswith('.png')]
    for i, file_name in enumerate(png_files):
        print(f"{i}/{len(png_files)}: Creating caption for {file_name}")
        # 1. Get the caption
        pair_name = file_name.rsplit('_', 1)[0]
        # 2. Get the background from image_name
        background = file_name.rsplit('_', 1)[1].rsplit('.', 1)[0]
        # 3. Create new caption
        
        original_caption = captions[pair_name]
        # new_caption = original_caption["caption"] + f", {background} background"
        
        if background == "585963": #dark gray
            # new_caption = original_caption["caption"] + "model on 585963 background"
            continue
        elif background == "987489": #pink
            new_caption = original_caption["caption"] + ", pink background"
        elif background == "e5e6eb": #shopify gray
            new_caption = original_caption["caption"] + ", gray background"
        
        print(f"{i}/{len(png_files)}: New caption created: {new_caption}")
        # 4. Add caption to dataset
        image_path = os.path.join(images_path, file_name)
        finesse_kp_background_dataset[image_path] = {"caption": new_caption}

with open("data/captions/finesse_kp_background_dataset_2.json", 'w') as file:
    json.dump(finesse_kp_background_dataset, file)
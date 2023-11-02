import os
import shutil

images_path = "/data/clogen/ed_datasets/finesse_prods"
output_folder_path = "demo_datasets/finesse_model_images"
for image_name in os.listdir(images_path):
    if "finesse" in image_name:
        source_file_path = os.path.join(images_path, image_name)
        print(f"Copying image {source_file_path}")
        destination_file_path = os.path.join(output_folder_path, image_name)
        shutil.copy(source_file_path, destination_file_path)


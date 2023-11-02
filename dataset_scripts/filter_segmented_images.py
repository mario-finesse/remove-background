import os
import shutil

"""
Given the filtered dataset of images that were segmented correctly (filtered_dataset_path), 
move all of the original images (gray_image_folder_path) to a new folder (output_folder_path)
so we can have a filtered dataset.
"""

filtered_dataset_path = "data/image_background/filtered 10d72c"
segmented_image_folder_path = "data/output/Slazzer"
output_folder_path = "data/output/filtered Slazzer"

png_files = [file for file in os.listdir(filtered_dataset_path) if file.endswith('.png')]
for i, image_name in enumerate(png_files):
    source_file_path = os.path.join(segmented_image_folder_path, image_name)
    destination_path = os.path.join(output_folder_path, image_name)
    print(f"{i}/{len(png_files)}: Copying image from {source_file_path} to {destination_path}")
    shutil.copy(source_file_path, destination_path)

import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_image(seg_image_name:str) -> None:
    print(f"Processing image {seg_image_name}")
    
    image_path = os.path.join(images_path, seg_image_name)
    raw_image = Image.open(image_path)
    local_background_image = background_image.resize(raw_image.size)
    combined_image = Image.composite(raw_image, local_background_image, raw_image)
    output_path = os.path.join(output_folder_path, seg_image_name)
    combined_image.save(output_path)
    
    print(f"Image {seg_image_name} added background {background}")

images_path = "data/output/filtered Slazzer"
background_image_path = "data/background/585963.png"
output_folder_path = "data/image_background/585963"

background = os.path.basename(background_image_path)[:-4]
background_image = Image.open(background_image_path)

segmented_images = [file for file in os.listdir(images_path) if file.endswith('.png')]

num_processes = 8
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    list(tqdm(executor.map(process_image, segmented_images), total=len(segmented_images)))
